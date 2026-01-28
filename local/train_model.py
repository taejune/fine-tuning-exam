import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

model_id = "/Users/ham/workspace/test_dir/Qwen2.5-0.5B-Instruct"
device = "cpu"

# 1️⃣ tokenizer / model
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32
).to(device)

# 2️⃣ LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3️⃣ dataset (❗ map 절대 쓰지 마라)
dataset = load_dataset("json", data_files="train.jsonl")

# 4️⃣ formatting_func는 "문자열"만 반환해야 함
def formatting_func(example):
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

# 5️⃣ training args
training_args = TrainingArguments(
    output_dir="./intel-mac-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    save_steps=20,
    fp16=False,
    bf16=False,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    formatting_func=formatting_func
)

# 6️⃣ Trainer
trainer.train()
