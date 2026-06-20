import os

BASE_DIR = "/scratch/s5208475/NLP"
CACHE_DIR = f"{BASE_DIR}/hf_cache"
TMP_DIR = f"{BASE_DIR}/tmp"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"
os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"
os.environ["TMPDIR"] = TMP_DIR
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset
from transformers import pipeline, logging, BitsAndBytesConfig
import pandas as pd
import torch
import re
import json
import random
import warnings

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

ds = load_dataset("Randolphzeng/Mr-Ben")
df = pd.DataFrame(ds['train'])

MODEL_LLAMa = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_QWEN25 = "Qwen/Qwen2.5-7B-Instruct"
MODEL_QWEN36 = "Qwen/Qwen3.6-27B"

def parse_options(options_str):
    pattern = r'([A-D]):\s*(.*?)(?=\s+[A-D]:|$)'
    matches = re.findall(pattern, options_str)
    return {k: v.strip() for k, v in matches}

def run_model(model_name, selected_indices, subject_key, subject_label, subject_df, out_dir, mode):
    model_short = model_name.replace("/", "_")

    # Check all exist before loading
    all_exist = True
    for i, idx in enumerate(selected_indices, 1):
        filename = f"{model_short}_{mode}_{subject_key}_{i}.json"
        filepath = os.path.join(out_dir, filename)
        if not os.path.exists(filepath):
            all_exist = False
            break
    if all_exist:
        print(f"  All {len(selected_indices)} files done for {model_name} on {subject_label}, skipping load", flush=True)
        return

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    llm = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        model_kwargs={"quantization_config": quant, "low_cpu_mem_usage": True}
    )

    for i, idx in enumerate(selected_indices, 1):
        filename = f"{model_short}_{mode}_{subject_key}_{i}.json"
        filepath = os.path.join(out_dir, filename)
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)", flush=True)
            continue

        row = subject_df.iloc[idx]
        options = parse_options(row['Options'])
        question = row['Question']

        prompt = f"""Please generate a step-by-step analysis for the following Question in the subject {subject_label}.
Question: {question}
Choice_A: {options.get('A', 'N/A')}
Choice_B: {options.get('B', 'N/A')}
Choice_C: {options.get('C', 'N/A')}
Choice_D: {options.get('D', 'N/A')}

Here is the desired format, please analyse each candidate choice sequentially and then jointly decide which option is the solution in the final step.
Please ensure every newline character follows a step indicator:
Step 1: [The first reasoning of the step by step analysis on the candidate choices here]
Step 2: [The second reasoning of the step by step analysis on the candidate choices here]
...
Step n: [Conclude your analysis and decide which choice to make here]
Solution: Choice_A/B/C/D
Please follow this format without any additional introductory or concluding statements."""

        max_tokens = 4000 if "Qwen3" in model_name else 1000
        messages = [{"role": "user", "content": prompt}]
        with torch.inference_mode():
            response = llm(messages, max_new_tokens=max_tokens, num_return_sequences=1, do_sample=False, repetition_penalty=1.1)

        generated = response[0]['generated_text']
        if isinstance(generated, list):
            steps_text = generated[-1]['content'].strip()
        else:
            steps_text = generated.replace(prompt, "").strip()

        # Strip thinking blocks, keep only final answer
        raw_text = re.sub(r'<think>.*?</think>\s*', '', steps_text, flags=re.DOTALL).strip()
        if not raw_text:
            raw_text = steps_text

        # Find the last draft (skip thinking drafts)
        last_one = raw_text.rfind("Step 1:")
        if last_one != -1:
            raw_text = raw_text[last_one:]

        # Extract solution (multiple formats)
        solution = "N/A"
        for pattern in [r'Solution:\s*(Choice_[A-D])', r'\\boxed\{([A-D])\}', r'(?:answer|correct).*?(?:Choice\s*)?[\( ]*([A-D])[\) ]*']:
            m = re.search(pattern, raw_text, flags=re.IGNORECASE)
            if m:
                ans = m.group(1)
                if ans.startswith("Choice_"):
                    solution = ans
                else:
                    solution = f"Choice_{ans}"
                break

        # Truncate at solution line or boxed answer
        trunc = re.split(r'\n\s*(?:Solution:|\\boxed\{)', raw_text.strip())[0]
        clean_text = trunc

        # Split into steps
        steps = re.split(r'\n\n?\s*(?=Step (?:\d+|n):)', clean_text)
        steps = [s.strip() for s in steps if s.strip()]
        steps = [re.sub(r'(?<=:)\s*\n\s*', ' ', s) for s in steps]

        data = {
            "model_name": model_name,
            "mode": mode,
            "subject": subject_label,
            "question_index": int(idx),
            "question_uuid": row['Question_UUID'],
            "question": question,
            "options": options,
            "analysis": steps,
            "solution": solution,
            "ground_truth": row['Ground_Truth_Answer']
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {filepath}", flush=True)

if __name__ == "__main__":
    for MODE in ["shared", "independent"]:
        print(f"\n{'='*60}\nStarting {MODE} mode\n{'='*60}", flush=True)
        subjects = {
            "high_school_biology": "High School Biology",
            "college_biology": "College Biology",
            "college_medicine": "College Medicine"
        }
        subject_keys = ["high_school_biology", "college_biology", "college_medicine"]
        num_per_subject = 50
        RANDOM_SEED = 42
        output_base = f"{BASE_DIR}/model_traces"
        model_order = [MODEL_LLAMa, MODEL_QWEN25, MODEL_QWEN36]

        for subject_key in subject_keys:
            subject_label = subjects[subject_key]
            subject_df = df[df['Subject'] == subject_key].reset_index(drop=True)

            unique_df = subject_df.drop_duplicates(subset='Question_UUID').reset_index()
            all_indices = unique_df['index'].tolist()
            rng = random.Random(RANDOM_SEED + subject_keys.index(subject_key))
            rng.shuffle(all_indices)

            shared_indices = all_indices[:50]
            remaining = all_indices[50:]
            indep_sizes = {
                "high_school_biology": [20, 20, 20],
                "college_biology": [62, 61, 62],
                "college_medicine": [68, 69, 68]
            }
            sizes = indep_sizes[subject_key]
            independent_chunks = []
            offset = 0
            for s in sizes:
                independent_chunks.append(remaining[offset:offset+s])
                offset += s

            for model_idx, model_name in enumerate(model_order):
                if MODE == "shared":
                    selected_indices = shared_indices
                    out_dir = os.path.join(output_base, "shared", subject_key)
                else:
                    selected_indices = independent_chunks[model_idx]
                    model_short = model_name.replace("/", "_")
                    out_dir = os.path.join(output_base, "independent", model_short, subject_key)
                os.makedirs(out_dir, exist_ok=True)

                print(f"\n=== {model_name} on {subject_label} ({MODE}, {len(selected_indices)} questions) ===", flush=True)
                run_model(model_name, selected_indices, subject_key, subject_label, subject_df, out_dir, MODE)