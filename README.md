# Model Trace Generation

## Requirements
To run the model trace generation, follow the next steps.

Connect to Hábrók via SSH and set up the environment:

```
cd /scratch/$USER
git clone <your-repo>
cd <your-repo>
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python3.11 -m venv trace_generation
source trace_generation/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers datasets pandas "regex>=2025.10.22"
python -m pip install accelerate
python -m pip install flash-attn --no-build-isolation
python -m pip install bitsandbytes
python -m pip install huggingface_hub datasets pandas
python -m pip install --upgrade transformers
```

Set your HuggingFace token and Git config:

```
export HF_TOKEN="your_token_here"
git config --global user.name "Your Name"
git config --global user.email "your.email@student.rug.nl"
```

## Instructions

After setting up, you can run the code with the following commands:

Create a job script and submit it:

```
sbatch job.sbatch
```

Monitor your job:

```
squeue -j <job_id>
squeue -u $USER
tail -f slurm-<job_id>.out
```

View the output traces (where X is your s-/p-number):

```
ls /scratch/sXXXXXXX/NLP/model_traces/
```

## Models

The three models used are:
- `meta-llama/Llama-3.1-8b-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen3.6-27B`

## Output Structure

The traces are saved under `model_traces/` in two modes:

- **shared/** — 50 questions per topic (high_school_biology, college_biology, college_medicine), all three models generate on the same questions
- **independent/** — remaining questions split per model, each model gets a disjoint chunk

Each trace is a JSON file containing the model name, subject, question, options, step-by-step analysis, and solution.

## Acknowledgements

This code was made with the help of the following people/resources:
- Group 11 from the NLP course
- Oscar Ferrer Domingo (Group 12 from the NLP course)
- Martijn Schippers (Group 12 from the NLP course)
- [OpenCode](https://opencode.ai)
- [Hábrók HPC cluster](https://wiki.hpc.rug.nl/habrok/introduction) (University of Groningen) 
  (with a total runtime of +/- 16 hours)
