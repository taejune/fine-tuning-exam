# Single-Node → KubeRay Ray Training 마이그레이션 가이드

## 개요

기존 단일 노드 LoRA fine-tuning 코드를 KubeRay 기반 분산 학습으로 변환한 내용을 정리합니다.

---

## 파일 구조 비교

```
변환 전:
├── train_model.py          # 단일 노드 트레이닝
├── train.jsonl             # 학습 데이터
└── pyproject.toml

변환 후:
├── train_model.py          # (유지) 로컬 개발/테스트용
├── train_model_ray.py      # (신규) Ray Train 분산 학습
├── train.jsonl
├── pyproject.toml          # ray[train] 의존성 추가
└── k8s/
    ├── kustomization.yaml  # Kustomize 설정
    ├── rayjob.yaml         # RayJob CRD
    ├── configmap.yaml      # 코드/데이터 ConfigMap
    └── pvc.yaml            # 출력 저장용 PVC
```

---

## 코드 비교: train_model.py vs train_model_ray.py

### 1. 초기화 및 설정

**변환 전 (단일 노드)**
```python
model_id = "/Users/ham/workspace/test_dir/Qwen2.5-0.5B-Instruct"
device = "cpu"
```

**변환 후 (Ray Train)**
```python
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
```

| 항목 | 변환 전 | 변환 후 |
|------|---------|---------|
| 모델 경로 | 하드코딩 로컬 경로 | 환경변수 (HuggingFace Hub 기본값) |
| 디바이스 | 수동 지정 | Ray Train 자동 관리 |
| 설정 | 코드 내 고정 | 환경변수로 주입 가능 |

---

### 2. 트레이닝 함수 구조

**변환 전**
```python
# 최상위 레벨에서 순차 실행
tokenizer = AutoTokenizer.from_pretrained(model_id, ...)
model = AutoModelForCausalLM.from_pretrained(model_id, ...).to(device)
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(...)
trainer.train()
```

**변환 후**
```python
def train_func():
    """각 Ray Worker에서 실행되는 함수"""
    context = train.get_context()
    world_size = context.get_world_size()
    rank = context.get_world_rank()
    
    # 각 worker가 독립적으로 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, ...)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)
    model = get_peft_model(model, lora_config)
    
    trainer = SFTTrainer(...)
    train_result = trainer.train()
    
    # rank 0에서만 저장
    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
    
    # Ray Train에 메트릭 리포트
    train.report({
        "train_loss": train_result.training_loss,
        ...
    })
```

| 항목 | 변환 전 | 변환 후 |
|------|---------|---------|
| 실행 방식 | 단일 프로세스 | 다중 Worker (DDP) |
| Worker 정보 | 없음 | `train.get_context()` |
| 모델 저장 | 직접 호출 | rank 0 조건부 |
| 메트릭 | 로컬 출력 | `train.report()` |

---

### 3. 트레이너 설정

**변환 전**
```python
training_args = TrainingArguments(
    output_dir="./intel-mac-lora",
    per_device_train_batch_size=1,
    ...
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    formatting_func=formatting_func
)

trainer.train()
```

**변환 후**
```python
# 1. HuggingFace TrainingArguments (각 worker 내부)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    ddp_find_unused_parameters=False,  # 분산 학습 최적화
    dataloader_pin_memory=False,
    ...
)

# 2. Ray Train 설정 (오케스트레이션)
scaling_config = ScalingConfig(
    num_workers=NUM_WORKERS,
    use_gpu=USE_GPU,
    resources_per_worker={"CPU": 2, "GPU": 1 if USE_GPU else 0},
)

run_config = RunConfig(
    name="lora-fine-tuning",
    storage_path=OUTPUT_DIR,
    checkpoint_config=CheckpointConfig(num_to_keep=2),
)

# 3. TorchTrainer로 분산 실행
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)

result = trainer.fit()
```

| 항목 | 변환 전 | 변환 후 |
|------|---------|---------|
| 스케일링 | 없음 (단일) | `ScalingConfig` |
| 체크포인트 | save_steps만 | `CheckpointConfig` (자동 관리) |
| 실행 | `trainer.train()` | `TorchTrainer.fit()` |
| 결과 | 없음 | `Result` 객체 (메트릭, 체크포인트 경로) |

---

### 4. 전체 아키텍처 비교

```
[변환 전: 단일 노드]
┌─────────────────────────┐
│     Local Machine       │
│  ┌───────────────────┐  │
│  │   train_model.py  │  │
│  │   - Model Load    │  │
│  │   - LoRA Setup    │  │
│  │   - SFTTrainer    │  │
│  └───────────────────┘  │
└─────────────────────────┘


[변환 후: KubeRay 분산 학습]
┌──────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                   RayJob (CRD)                      │ │
│  │  ┌─────────────┐                                    │ │
│  │  │  Ray Head   │ ← TorchTrainer orchestration       │ │
│  │  └──────┬──────┘                                    │ │
│  │         │                                           │ │
│  │    ┌────┴────┐                                      │ │
│  │    ▼         ▼                                      │ │
│  │ ┌──────┐ ┌──────┐                                   │ │
│  │ │Worker│ │Worker│  ← train_func() (DDP)             │ │
│  │ │ GPU  │ │ GPU  │                                   │ │
│  │ └──────┘ └──────┘                                   │ │
│  └─────────────────────────────────────────────────────┘ │
│                           │                              │
│                           ▼                              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              PVC (training-output-pvc)              │ │
│  │              - LoRA checkpoints                     │ │
│  │              - Training logs                        │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## Kubernetes 리소스 설명

### RayJob (`k8s/rayjob.yaml`)

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: lora-fine-tuning-job
spec:
  entrypoint: python /app/train_model_ray.py
  runtimeEnvYAML: |
    pip: [...]      # 의존성
    env_vars: {...} # 환경변수
  
  rayClusterSpec:
    headGroupSpec: ...    # Ray Head 노드
    workerGroupSpecs: ... # GPU Worker 노드들
```

### 주요 설정

| 설정 | 용도 |
|------|------|
| `runtimeEnvYAML.pip` | 런타임 의존성 설치 |
| `runtimeEnvYAML.env_vars` | 환경변수 주입 |
| `headGroupSpec` | Ray 클러스터 헤드 (오케스트레이션) |
| `workerGroupSpecs` | 실제 학습 수행 Worker |
| `shutdownAfterJobFinishes` | 완료 후 클러스터 자동 종료 |

---

## 실행 방법

### 로컬 테스트 (Ray 단독)

```bash
# Ray 의존성 설치
uv add "ray[train]>=2.40.0"

# 로컬 실행 (Ray가 로컬 클러스터 자동 생성)
uv run python train_model_ray.py
```

### KubeRay 클러스터 배포

```bash
# 1. KubeRay Operator 설치 (최초 1회)
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator

# 2. 네임스페이스 생성
kubectl create namespace ray-system

# 3. 리소스 배포
kubectl apply -k k8s/

# 4. Job 상태 확인
kubectl get rayjob -n ray-system
kubectl logs -f job/lora-fine-tuning-job -n ray-system

# 5. Ray Dashboard 접근
kubectl port-forward svc/lora-fine-tuning-job-head-svc 8265:8265 -n ray-system
# http://localhost:8265
```

---

## 환경변수 설정

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace 모델 ID |
| `DATA_PATH` | `train.jsonl` | 학습 데이터 경로 |
| `OUTPUT_DIR` | `/tmp/ray-lora-output` | 체크포인트 저장 경로 |
| `NUM_WORKERS` | `2` | 분산 학습 Worker 수 |
| `USE_GPU` | `false` | GPU 사용 여부 |

---

## 주요 차이점 요약

| 항목 | 단일 노드 | KubeRay Ray Training |
|------|-----------|----------------------|
| **스케일링** | 불가 | Worker 수 조절 |
| **리소스 관리** | 수동 | Kubernetes 자동 |
| **체크포인트** | 로컬 저장 | PVC 영구 저장 |
| **모니터링** | 없음 | Ray Dashboard |
| **내결함성** | 없음 | Worker 자동 복구 |
| **재현성** | 환경 의존 | 컨테이너 이미지 고정 |
| **비용** | 항상 실행 | 작업 완료 시 자동 종료 |

---

## 다음 단계

1. **GPU 클러스터 구성**: 실제 GPU 노드로 Worker 스케일 아웃
2. **모델 캐싱**: HuggingFace 모델을 PVC에 캐시하여 시작 시간 단축
3. **하이퍼파라미터 튜닝**: Ray Tune 연동
4. **모델 서빙**: Ray Serve로 학습된 LoRA 모델 배포
