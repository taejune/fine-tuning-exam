# AGENTS.md - Coding Agent Guidelines

## Project Overview

LLM fine-tuning project using LoRA (Low-Rank Adaptation) with Hugging Face Transformers stack.
Trains small language models on custom instruction-following datasets.

**Tech Stack:**
- Python 3.13+
- PyTorch 2.9+
- Transformers 4.57+
- PEFT (Parameter-Efficient Fine-Tuning) 0.18+
- TRL (Transformer Reinforcement Learning) 0.26+
- Datasets 4.5+

---

## Commands

### Package Management (uv)

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package>

# Add dev dependency
uv add --dev <package>

# Run Python script
uv run python <script.py>
```

### Running Scripts

```bash
# Train model with LoRA
uv run python train_model.py

# Compare base vs LoRA model inference
uv run python infer_compare.py
```

### Testing

No test framework configured yet. When adding tests:

```bash
# Recommended: pytest
uv add --dev pytest

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_example.py

# Run single test function
uv run pytest tests/test_example.py::test_function_name

# Run with verbose output
uv run pytest -v

# Run tests matching pattern
uv run pytest -k "pattern"
```

### Linting (when configured)

No linter configured yet. Recommended setup:

```bash
# Add ruff for linting and formatting
uv add --dev ruff

# Lint check
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking (optional)
uv add --dev mypy
uv run mypy .
```

---

## Code Style Guidelines

### Imports

Order imports as follows:
1. Standard library imports
2. Third-party imports (torch, transformers, etc.)
3. Local imports

```python
# Standard library
import os

# Third-party
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

# Local
from my_module import my_function
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `model_id`, `lora_config` |
| Constants | UPPER_SNAKE_CASE | `BASE_MODEL_PATH`, `DEVICE` |
| Functions | snake_case | `formatting_func()` |
| Classes | PascalCase | `CustomTrainer` |
| Files | snake_case | `train_model.py` |

### Type Hints

Use type hints for function signatures:

```python
def formatting_func(example: dict[str, str]) -> str:
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
```

### ML/Training Conventions

**Model Loading:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Use torch_dtype, not dtype
    trust_remote_code=True,
)
```

**LoRA Configuration:**
```python
lora_config = LoraConfig(
    r=4,                    # Low rank
    lora_alpha=8,           # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Training Arguments:**
```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=20,
    fp16=False,  # CPU training
    bf16=False,
    report_to="none",
)
```

### Dataset Handling

**IMPORTANT:** Use `formatting_func` with SFTTrainer instead of `dataset.map()`:

```python
# CORRECT: Use formatting_func
def formatting_func(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    formatting_func=formatting_func,  # String output only
)

# AVOID: dataset.map() for tokenization with SFTTrainer
```

### Inference Pattern

```python
model.eval()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
```

---

## Project Structure

```
fine-tuning-exam/
├── pyproject.toml      # Project config and dependencies
├── uv.lock             # Lock file (do not edit manually)
├── train_model.py      # LoRA fine-tuning script
├── infer_compare.py    # Base vs LoRA inference comparison
├── train.jsonl         # Training data (instruction/output pairs)
├── AGENTS.md           # This file
└── .venv/              # Virtual environment
```

---

## Data Format

Training data uses JSONL format with instruction-output pairs:

```jsonl
{"instruction": "Explain the ls command", "output": "ls lists directory contents."}
{"instruction": "What does kubectl get pods do?", "output": "Lists pods in current namespace."}
```

---

## Error Handling

- Check CUDA/MPS availability before assuming GPU
- Use `torch.float32` for CPU training (fp16/bf16 disabled)
- Verify model paths exist before loading
- Handle tokenizer padding/truncation explicitly

---

## Notes for AI Agents

1. **Model paths are hardcoded** - Update `model_id`, `BASE_MODEL_PATH`, `LORA_PATH` for your environment
2. **CPU-only training** - Scripts default to CPU; modify for GPU if available
3. **No tests exist** - Add pytest tests when extending functionality
4. **No linting configured** - Consider adding ruff for code quality
5. **Small dataset** - `train.jsonl` has only 4 examples; extend for real training
