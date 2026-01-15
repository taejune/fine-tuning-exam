import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 경로
BASE_MODEL_PATH = "/Users/ham/workspace/test_dir/Qwen2.5-0.5B-Instruct"
LORA_PATH = "/Users/ham/workspace/test_dir/intel-mac-lora/checkpoint-2"

device = "cpu"

# tokenizer (공통)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

# ===============================
# 1️⃣ 베이스 모델
# ===============================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.float32
).to(device)

base_model.eval()

# ===============================
# 2️⃣ LoRA 모델 (베이스 + 어댑터)
# ===============================
lora_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.float32
).to(device)

lora_model = PeftModel.from_pretrained(
    lora_model,
    LORA_PATH
)

lora_model.eval()

base_params = dict(base_model.named_parameters())
lora_params = dict(lora_model.named_parameters())

diff = False
for k in lora_params:
    if "lora" in k:
        diff = True
        print("LoRA param:", k)
        break

print("LoRA params exist:", diff)


# ===============================
# 테스트 프롬프트
# ===============================
prompt = """### Instruction:
kubectl get pods는 무슨 명령인가?

### Response:
"""


inputs = tokenizer(prompt, return_tensors="pt")

# ===============================
# 베이스 모델 추론
# ===============================
print("\n===== BASE MODEL OUTPUT =====")
with torch.no_grad():
    base_out = base_model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
print(tokenizer.decode(base_out[0], skip_special_tokens=True))

# ===============================
# LoRA 모델 추론
# ===============================
print("\n===== LORA MODEL OUTPUT =====")
with torch.no_grad():
    lora_out = lora_model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
print(tokenizer.decode(lora_out[0], skip_special_tokens=True))
