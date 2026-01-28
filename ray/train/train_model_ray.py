"""
Ray Train + KubeRay 기반 LoRA Fine-tuning 스크립트

기존 단일 노드 트레이닝을 분산 학습으로 변환한 버전.
KubeRay RayJob으로 Kubernetes 클러스터에서 실행 가능.
"""

import os

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer


# ============================================
# Configuration (환경변수로 오버라이드 가능)
# ============================================
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "../../local/train.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/ray-lora-output")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"


def formatting_func(example: dict[str, str]) -> str:
    """데이터셋 포맷팅 함수 - SFTTrainer에서 사용"""
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""


def train_func():
    """
    각 Ray Worker에서 실행되는 트레이닝 함수.
    Ray Train이 자동으로 분산 환경을 설정.
    """
    # Ray Train context에서 worker 정보 가져오기
    context = train.get_context()
    world_size = context.get_world_size()
    rank = context.get_world_rank()

    print(f"[Worker {rank}/{world_size}] Starting training...")

    # 1️⃣ Tokenizer / Model 로드
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    # padding token 설정 (없는 경우)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # 2️⃣ LoRA Configuration
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    if rank == 0:
        model.print_trainable_parameters()

    # 3️⃣ Dataset 로드
    dataset = load_dataset("json", data_files=DATA_PATH)

    # 4️⃣ Training Arguments (분산 학습용 설정)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=20,
        fp16=False,
        bf16=False,
        report_to="none",
        # 분산 학습 설정 - Ray Train이 자동 처리
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
    )

    # 5️⃣ SFTTrainer 초기화
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
    )

    # 6️⃣ 트레이닝 실행
    train_result = trainer.train()

    # 7️⃣ 모델 저장 (rank 0에서만)
    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
        print(f"[Worker {rank}] Model saved to {OUTPUT_DIR}")

    # 8️⃣ Ray Train에 메트릭 리포트
    train.report({
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    })


def main():
    """Ray Train을 사용한 분산 학습 실행"""

    # Ray 초기화 (KubeRay에서는 자동으로 클러스터에 연결)
    if not ray.is_initialized():
        ray.init()

    print(f"Ray initialized. Dashboard: {ray.get_dashboard_url()}")
    print(f"Available resources: {ray.available_resources()}")

    # Scaling Config: Worker 수와 리소스 설정
    scaling_config = ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=USE_GPU,
        resources_per_worker={
            "CPU": 2,
            "GPU": 1 if USE_GPU else 0,
        },
    )

    # Checkpoint Config: 체크포인트 저장 설정
    checkpoint_config = CheckpointConfig(
        num_to_keep=2,  # 최근 2개 체크포인트만 유지
    )

    # Run Config: 실행 설정
    run_config = RunConfig(
        name="lora-fine-tuning",
        storage_path=OUTPUT_DIR,
        checkpoint_config=checkpoint_config,
    )

    # TorchTrainer: PyTorch 분산 학습을 위한 Ray Train API
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # 트레이닝 실행
    print("Starting distributed training...")
    result = trainer.fit()

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Results: {result.metrics}")
    print(f"Checkpoint path: {result.checkpoint}")
    print("=" * 50)

    return result


if __name__ == "__main__":
    main()
