"""
Prefect 기본 워크플로 예제 (Ray 없이 단일 노드 파인튜닝)

워크플로 구성:
  1. validate_dataset   - 데이터셋 검증
  2. run_fine_tuning    - 단일 노드 LoRA 파인튜닝
  3. evaluate_model     - 모델 추론 테스트

실행 방법:
  prefect server start                          # 별도 터미널
  python prefect_basic_workflow.py
"""

import json
import os
from pathlib import Path

import torch
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


# ============================================
# Configuration
# ============================================
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", str(Path(__file__).parent.parent / "training" / "train.jsonl"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/prefect-lora-output")


# ============================================
# Task 1: 데이터셋 검증
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
        "avg_instruction_len": round(sum(len(r["instruction"]) for r in records) / len(records)),
        "avg_output_len": round(sum(len(r["output"]) for r in records) / len(records)),
    }

    logger.info(f"Validation passed: {stats['valid_records']}/{stats['total_lines']} records valid")
    return stats


# ============================================
# Task 2: 단일 노드 LoRA 파인튜닝
# ============================================
@task(name="fine-tuning", timeout_seconds=3600)
def run_fine_tuning(data_path: str) -> dict:
    """단일 노드에서 LoRA 파인튜닝을 실행한다."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    logger = get_run_logger()
    logger.info(f"Loading model: {MODEL_ID}")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = load_dataset("json", data_files=data_path)

    def formatting_func(example: dict[str, str]) -> str:
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

    # Training
    training_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=20,
        max_seq_length=512,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_config,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    result = trainer.train()

    trainer.save_model(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")

    metrics = {
        "train_loss": result.training_loss,
        "train_runtime": result.metrics.get("train_runtime", 0),
        "train_samples_per_second": result.metrics.get("train_samples_per_second", 0),
    }
    logger.info(f"Training complete: {metrics}")
    return metrics


# ============================================
# Task 3: 모델 평가
# ============================================
@task(name="evaluate-model", retries=1)
def evaluate_model(training_metrics: dict) -> dict:
    """학습된 모델로 간단한 추론 테스트를 수행한다."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger = get_run_logger()
    logger.info("Evaluating model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

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
        logger.info(f"Q: {prompt.split(chr(10))[1][:50]}...")
        logger.info(f"A: {response[:80]}...")

    # Prefect Artifact 생성
    md = "## Fine-tuning Report\n\n"
    md += "| Metric | Value |\n|--------|-------|\n"
    md += f"| Train Loss | {training_metrics.get('train_loss', 'N/A'):.4f} |\n"
    md += f"| Runtime (s) | {training_metrics.get('train_runtime', 'N/A'):.1f} |\n"
    md += f"| Samples/sec | {training_metrics.get('train_samples_per_second', 'N/A'):.2f} |\n\n"
    md += "### Inference Samples\n\n"
    for r in results:
        md += f"**Q:** {r['prompt']}\n\n**A:** {r['response']}\n\n---\n\n"

    create_markdown_artifact(
        key="finetuning-report",
        markdown=md,
        description="LoRA fine-tuning evaluation report",
    )

    logger.info("Evaluation complete")
    return {"num_tests": len(results), "results": results}


# ============================================
# Flow: 전체 파이프라인
# ============================================
@flow(name="basic-fine-tuning-pipeline", log_prints=True)
def fine_tuning_pipeline(
    data_path: str = DATA_PATH,
):
    """
    Prefect 단독 LoRA Fine-tuning 파이프라인.

    Steps:
      1. validate_dataset  → 데이터셋 무결성 검증
      2. run_fine_tuning   → 단일 노드 LoRA 학습
      3. evaluate_model    → 추론 테스트 및 리포트
    """
    # Step 1
    dataset_stats = validate_dataset(data_path)
    print(f"Dataset: {dataset_stats}")

    # Step 2
    training_metrics = run_fine_tuning(data_path)
    print(f"Training: {training_metrics}")

    # Step 3
    eval_result = evaluate_model(training_metrics)
    print(f"Evaluation: {eval_result}")

    return {
        "dataset": dataset_stats,
        "training": training_metrics,
        "evaluation": eval_result,
    }


if __name__ == "__main__":
    fine_tuning_pipeline()
