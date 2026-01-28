import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 디바이스 설정 (CUDA 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 1. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# dtype: torch_dtype 파라미터명 사용, device_map을 auto로 변경
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # 자동으로 GPU에 배치
    torch_dtype=torch.bfloat16,  # 올바른 파라미터명
    trust_remote_code=True,
)

# 2. 데이터셋 로드
dataset = load_dataset("json", data_files="/workspace/data/my_dataset.jsonl", split="train")

# 3. LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. SFTConfig 설정
sft_config = SFTConfig(
    output_dir="/workspace/checkpoints/qwen-finetuned",
    max_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_steps=20,
    logging_steps=1,
    save_strategy="no",
    max_grad_norm=1.0,
    packing=False,
    # Linux/CUDA 추가 설정
    bf16=True,  # bfloat16 학습 활성화
    dataloader_num_workers=4,  # 데이터 로딩 병렬화
    dataloader_pin_memory=True,  # GPU 전송 최적화
)

# 5. 트레이너 실행
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,
)

print("학습 시작...")
trainer.train()

# 6. 학습 후 테스트
print("\n--- 학습 후 테스트 ---")
model.eval()
prompt = "서버 상태가 어때?"

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"질문: {prompt}")
print(f"답변: {response}")
