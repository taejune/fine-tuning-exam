import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer  # SFTConfig 추가
from peft import LoraConfig

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# dtype 경고 해결: torch_dtype -> dtype
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="mps",
    dtype=torch.float16
)

# 2. 데이터셋 로드 (기존에 생성한 my_dataset.jsonl 사용)
dataset = load_dataset("json", data_files="my_dataset.jsonl", split="train")

# 3. LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. SFTConfig 설정 (max_seq_length 에러 해결의 핵심)
sft_config = SFTConfig(
    output_dir="./qwen-finetuned",
    max_length=512,              # 여기에 max_seq_length를 넣습니다.
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=20,
    logging_steps=1,
    save_strategy="no",
    packing=False                    # 추가적인 최신 설정
)

# 5. 트레이너 실행
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,                 # 수정된 sft_config 전달
)

print("학습 시작...")
trainer.train()

# 6. 간단한 테스트
print("\n--- 학습 후 테스트 ---")
prompt = "서버 상태가 어때?"
# Qwen2.5 인스트럭트 포맷 적용
formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
