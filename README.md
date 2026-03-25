# MedS-Ins
[NPJ Digital Medicine] The official codes for "Towards Evaluating and Building Versatile Large Language Models for Medicine"

[Paper (arxiv)](https://arxiv.org/abs/2408.12547) | [Paper (npj digital medicine)](https://www.nature.com/articles/s41746-024-01390-4) | [Leaderboard](https://henrychur.github.io/MedS-Bench/)

Datasets: [MedS-Ins](https://huggingface.co/datasets/Henrychur/MedS-Ins), [MedS-Bench](https://huggingface.co/datasets/Henrychur/MedS-Bench)

Models: [MMedS-Llama3-8B](https://huggingface.co/Henrychur/MMedS-Llama-3-8B)

---

## What's New in This Fork

This fork adds `Evaluation/eval_vllm.py`, a unified evaluation script that runs any vLLM-served model against MedS-Bench tasks without modifying any original dataset files or scoring logic.

**Key additions:**
- OpenAI-compatible API client replacing HuggingFace model loading — works with vLLM, SGLang, or any OpenAI-compatible server
- Thinking model support via `chat_template_kwargs` (tested on Qwen3.5)
- Parallel inference via `ThreadPoolExecutor` for fast throughput
- Task-aware scoring: token F1 for NER tasks, accuracy for classification tasks
- Per-sample prediction/expected logging during runs
- Structured `Answer:` output format enforced in system prompt for reliable parsing
- Task-specific prompt reinforcements (e.g. verbatim entity extraction for NER)

---

## Introduction

In this study, we introduce **MedS-Bench**, a comprehensive benchmark designed to evaluate the performance of large language models (LLMs) in clinical contexts. Unlike traditional benchmarks that focus solely on multiple-choice question answering, **MedS-Bench** covers 11 critical clinical tasks, including clinical report summarization, treatment recommendations, diagnosis, named entity recognition, and medical concept explanation, among others. We evaluated six leading LLMs, such as MEDITRON, Mistral, InternLM 2, Llama 3, GPT-4, and Claude-3.5, using few-shot prompting. Our findings reveal that even the most advanced models face significant challenges in these complex tasks.

To overcome these limitations, we developed **MedS-Ins**, a large-scale instruction tuning dataset tailored for medicine. **MedS-Ins** comprises 58 medically oriented language corpora, totaling 5M instances with 19K instructions across 122 tasks. To demonstrate the dataset's utility, we conducted a proof-of-concept experiment by performing instruction tuning on a lightweight, open-source medical language model. The resulting model, **MMedIns-Llama 3**, significantly outperformed existing models across nearly all clinical tasks.

---

## Repository Structure

```
MedS-Ins/
├── Evaluation/
│   ├── eval.py                  # Original HuggingFace eval script
│   ├── eval_vllm.py             # vLLM eval script (this fork)
│   ├── EvalDataset.py           # Original dataset loader (untouched)
│   ├── RewriteDefinitions.json  # Task definition overrides
│   └── prediction_results/      # CSV outputs (gitignored)
├── Metrics/                     # Per-task scoring implementations
│   ├── NamedEntityRecognition.py
│   ├── Diagnosis.py
│   ├── TreatmentPlanning.py
│   └── ...
├── Inference/                   # Original inference utilities
├── data_preparing/              # Dataset preparation scripts
├── DATA_Contributor/            # Guidelines for contributing new tasks
├── examples/                    # Example eval result CSVs
└── assets/                      # Figures for README
```

---

## vLLM Evaluation

### 1. Serve your model

```bash
vllm serve Qwen/Qwen3.5-4B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32000 \
  --served-model-name qwen-4B \
  --reasoning-parser qwen3
```

### 2. Download benchmark data

```bash
python3 - << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Henrychur/MedS-Bench",
    repo_type="dataset",
    local_dir="./MedS-Bench-data",
    allow_patterns=["NER/*", "Diagnosis/*", "Treatment_planning/*"]
)
EOF
```

### 3. Run eval

**Classification tasks** (accuracy scoring):

```bash
cd Evaluation

python eval_vllm.py \
  --model_name qwen-4B \
  --api_base http://localhost:8000/v1 \
  --benchmark_task DDXPlus \
  --data_file ../MedS-Bench-data/Diagnosis/task130_DDXPlus_text_classification_test.json \
  --max_samples 500 \
  --max_tokens 16384 \
  --temperature 1.0 \
  --top_p 0.95 \
  --presence_penalty 1.5 \
  --top_k 20 \
  --thinking_budget 8192 \
  --enable_thinking \
  --workers 32
```

**NER tasks** (token F1 scoring):

```bash
python eval_vllm.py \
  --model_name qwen-4B \
  --api_base http://localhost:8000/v1 \
  --benchmark_task bc4chem \
  --data_file ../MedS-Bench-data/NER/task125_test_bc4chem_named_enetity_recognition.json \
  --max_samples 500 \
  --max_tokens 16384 \
  --temperature 1.0 \
  --top_p 0.95 \
  --presence_penalty 1.5 \
  --top_k 20 \
  --thinking_budget 8192 \
  --enable_thinking \
  --workers 32
```

The script auto-detects task type from `--benchmark_task`. NER mode activates for: `bc4chem`, `bc5chem`, `bc5disease`, `species800`.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `qwen-4B` | Served model name in vLLM |
| `--api_base` | `http://localhost:8000/v1` | API base URL |
| `--benchmark_task` | required | Task name, used for output filename and mode detection |
| `--data_file` | required | Path to task JSON |
| `--max_samples` | 500 | Instances to evaluate |
| `--max_tokens` | 16384 | Max tokens per response |
| `--temperature` | 1.0 | Sampling temperature |
| `--top_p` | 0.95 | Top-p |
| `--top_k` | 20 | Top-k |
| `--presence_penalty` | 1.5 | Presence penalty |
| `--enable_thinking` | True | Enable thinking mode (`--no_thinking` to disable) |
| `--thinking_budget` | 8192 | Max thinking tokens |
| `--workers` | 32 | Parallel inference workers |

### Output

Results saved to `Evaluation/prediction_results/eval_{task}_{model}.csv`.

| Task type | CSV columns | Metric |
|---|---|---|
| Classification | `task_id, input, GT, raw_output, parsed_output, correct` | Accuracy |
| NER | `task_id, input, GT, raw_output, parsed_output, precision, recall, f1` | Token F1 |

---

## Adding a New Task

1. Add your task JSON to the appropriate category folder under `MedS-Bench-data/` following the schema below.

2. If your task requires a new scoring mode beyond accuracy or NER token F1, add it to `Metrics/` following the existing pattern, then add detection logic in `eval_vllm.py`:

```python
# At the top of eval_vllm.py
NER_TASKS = {"bc4chem", "bc5chem", "bc5disease", "species800"}
YOUR_TASKS = {"your_task_name"}  # add your set

# Detection
is_ner = any(t in args.benchmark_task.lower() for t in NER_TASKS)
is_your_type = any(t in args.benchmark_task.lower() for t in YOUR_TASKS)
```

3. Add a task-specific format instruction and any prompt reinforcements in the `format_instruction` block.

4. Add the run command to this README.

### Task JSON Schema

```json
{
  "Contributors": [""],
  "Source": [""],
  "URL": [""],
  "Categories": [""],
  "Reasoning": [""],
  "Definition": [""],
  "Input_language": [""],
  "Output_language": [""],
  "Instruction_language": [""],
  "Domains": [""],
  "Positive Examples": [{"input": "", "output": "", "explanation": ""}],
  "Negative Examples": [{"input": "", "output": "", "explanation": ""}],
  "Instances": [{"id": "", "input": "", "output": [""]}]
}
```

---

## Citation

```bibtex
@misc{wu2024evaluatingbuildingversatilelarge,
      title={Towards Evaluating and Building Versatile Large Language Models for Medicine},
      author={Chaoyi Wu and Pengcheng Qiu and Jinxin Liu and Hongfei Gu and Na Li and Ya Zhang and Yanfeng Wang and Weidi Xie},
      year={2024},
      eprint={2408.12547},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.12547},
}
```