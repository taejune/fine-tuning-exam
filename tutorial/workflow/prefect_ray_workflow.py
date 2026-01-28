"""
Prefect + Ray Train 파인튜닝 워크플로

워크플로 구성:
  1. validate_dataset   - 데이터셋 검증 (pre-task)
  2. run_ray_training   - Ray 분산 파인튜닝 (main task)
  3. evaluate_model     - 모델 평가 (post-task)

실행 방법 (로컬):
  prefect server start          # Prefect 서버 시작 (별도 터미널)
  python prefect_ray_workflow.py

실행 방법 (Kubernetes):
  k8s/prefect/ 디렉토리의 YAML 리소스를 배포한 후
  Prefect Worker가 자동으로 flow run을 실행합니다.
"""

import json
import os
from pathlib import Path

import ray
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

# ============================================
# Configuration
# ============================================
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", str(Path(__file__).parent.parent / "training" / "train.jsonl"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/prefect-ray-output")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"


# ============================================
# Pre-task: 데이터셋 검증
# ============================================
@task(name="validate-dataset", retries=2, retry_delay_seconds=5)
def validate_dataset(data_path: str) -> dict:
    """학습 데이터셋의 유효성을 검증한다."""
    logger = get_run_logger()
    logger.info(f"Validating dataset: {data_path}")

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    records = []
    required_keys = {"instruction", "output"}
    errors = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {i}: invalid JSON")
                continue

            missing = required_keys - record.keys()
            if missing:
                errors.append(f"Line {i}: missing keys {missing}")
                continue

            if not record["instruction"].strip() or not record["output"].strip():
                errors.append(f"Line {i}: empty instruction or output")
                continue

            records.append(record)

    if errors:
        for err in errors:
            logger.warning(err)

    if not records:
        raise ValueError("No valid records found in dataset")

    stats = {
        "total_lines": i,
        "valid_records": len(records),
        "errors": len(errors),
        "avg_instruction_len": sum(len(r["instruction"]) for r in records) / len(records),
        "avg_output_len": sum(len(r["output"]) for r in records) / len(records),
    }

    logger.info(f"Dataset validation passed: {stats['valid_records']}/{stats['total_lines']} records valid")
    return stats


# ============================================
# Main task: Ray 분산 파인튜닝
# ============================================
def _train_func():
    """각 Ray Worker에서 실행되는 학습 함수."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from trl import SFTTrainer

    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    print(f"[Worker {rank}/{world_size}] Starting training...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if rank == 0:
        model.print_trainable_parameters()

    def formatting_func(example: dict[str, str]) -> str:
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

    dataset = load_dataset("json", data_files=DATA_PATH)

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
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
    )

    result = trainer.train()

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
        print(f"[Worker {rank}] Model saved to {OUTPUT_DIR}")

    train.report({
        "train_loss": result.training_loss,
        "train_runtime": result.metrics.get("train_runtime", 0),
        "train_samples_per_second": result.metrics.get("train_samples_per_second", 0),
    })


@task(name="ray-fine-tuning", timeout_seconds=3600)
def run_ray_training() -> dict:
    """Ray Train을 사용하여 LoRA 분산 파인튜닝을 실행한다."""
    logger = get_run_logger()

    if not ray.is_initialized():
        ray.init()

    logger.info(f"Ray cluster: {ray.available_resources()}")
    logger.info(f"Model: {MODEL_ID}, Workers: {NUM_WORKERS}, GPU: {USE_GPU}")

    scaling_config = ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=USE_GPU,
        resources_per_worker={"CPU": 2, "GPU": 1 if USE_GPU else 0},
    )

    run_config = RunConfig(
        name="prefect-lora-fine-tuning",
        storage_path=OUTPUT_DIR,
        checkpoint_config=CheckpointConfig(num_to_keep=2),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=_train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    logger.info("Starting distributed training...")
    result = trainer.fit()

    metrics = {
        "train_loss": result.metrics.get("train_loss", None),
        "train_runtime": result.metrics.get("train_runtime", None),
        "train_samples_per_second": result.metrics.get("train_samples_per_second", None),
        "checkpoint_path": str(result.checkpoint) if result.checkpoint else None,
    }

    logger.info(f"Training completed: {metrics}")
    return metrics


# ============================================
# Post-task: 모델 평가
# ============================================
@task(name="evaluate-model", retries=1)
def evaluate_model(training_metrics: dict) -> dict:
    """학습된 모델의 간단한 추론 테스트를 수행한다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger = get_run_logger()
    logger.info("Starting model evaluation...")

    checkpoint_path = training_metrics.get("checkpoint_path") or OUTPUT_DIR

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    # 간단한 추론 테스트
    test_prompts = [
        "### Instruction:\nKubernetes에서 Pod 목록을 확인하는 명령어는?\n\n### Response:\n",
        "### Instruction:\n현재 디렉토리의 파일 목록을 보는 리눅스 명령어는?\n\n### Response:\n",
    ]

    results = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        results.append({"prompt": prompt.split("\n")[1], "response": response})
        logger.info(f"Prompt: {prompt.split(chr(10))[1][:50]}... -> {response[:80]}...")

    eval_result = {
        "num_test_prompts": len(test_prompts),
        "results": results,
        "train_loss": training_metrics.get("train_loss"),
    }

    # Prefect Artifact로 결과 기록
    markdown_body = f"""## Fine-tuning Results

| Metric | Value |
|--------|-------|
| Train Loss | {training_metrics.get('train_loss', 'N/A')} |
| Runtime (s) | {training_metrics.get('train_runtime', 'N/A')} |
| Samples/sec | {training_metrics.get('train_samples_per_second', 'N/A')} |

### Inference Samples

"""
    for r in results:
        markdown_body += f"**Q:** {r['prompt']}\n\n**A:** {r['response']}\n\n---\n\n"

    create_markdown_artifact(
        key="finetuning-report",
        markdown=markdown_body,
        description="LoRA Fine-tuning evaluation report",
    )

    logger.info("Evaluation complete")
    return eval_result


# ============================================
# Prefect Flow: 전체 파이프라인
# ============================================
@flow(name="lora-fine-tuning-pipeline", log_prints=True)
def fine_tuning_pipeline(
    data_path: str = DATA_PATH,
    model_id: str = MODEL_ID,
):
    """
    LoRA Fine-tuning 전체 파이프라인.

    Steps:
      1. validate_dataset  - 데이터셋 무결성 검증
      2. run_ray_training  - Ray 분산 파인튜닝
      3. evaluate_model    - 결과 평가 및 리포트
    """
    # Step 1: 데이터 검증
    dataset_stats = validate_dataset(data_path)
    print(f"Dataset stats: {dataset_stats}")

    # Step 2: Ray 분산 학습
    training_metrics = run_ray_training()
    print(f"Training metrics: {training_metrics}")

    # Step 3: 모델 평가
    eval_result = evaluate_model(training_metrics)
    print(f"Evaluation: {eval_result}")

    return {
        "dataset": dataset_stats,
        "training": training_metrics,
        "evaluation": eval_result,
    }


if __name__ == "__main__":
    fine_tuning_pipeline()
