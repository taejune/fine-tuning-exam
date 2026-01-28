# Ray Core Tutorial

Ray의 핵심 개념을 학습하는 예제 모음입니다.

## 실행 방법

```bash
cd tutorial/core
uv run python 01_tasks.py
```

## 예제 목록

| 파일 | 개념 | 설명 |
|------|------|------|
| `01_tasks.py` | Tasks | `@ray.remote` 함수, 병렬 실행, 태스크 의존성 |
| `02_actors.py` | Actors | 상태를 유지하는 원격 객체 |
| `03_object_store.py` | Object Store | 대용량 데이터 공유, `ray.put()` |
| `04_wait_pattern.py` | ray.wait() | 완료된 태스크부터 처리 |
| `05_resources.py` | Resources | CPU/GPU/커스텀 리소스 할당 |
| `06_error_handling.py` | Error Handling | 예외 처리, 재시도, 타임아웃 |
| `07_actor_pool.py` | ActorPool | Actor 풀로 작업 분배 |

## 핵심 개념

### Tasks vs Actors

```
Tasks (@ray.remote 함수)     Actors (@ray.remote 클래스)
─────────────────────────    ─────────────────────────
상태 없음 (Stateless)         상태 있음 (Stateful)
호출마다 새 worker            동일 인스턴스 유지
병렬 실행 용이                순차 처리 (메서드별)
```

### ObjectRef

```python
ref = task.remote(x)  # ObjectRef 반환 (미래 값)
result = ray.get(ref) # 실제 값 가져오기 (blocking)
```

### Object Store

```python
data_ref = ray.put(large_data)  # 한 번 저장
task1.remote(data_ref)          # 복사 없이 참조
task2.remote(data_ref)          # 복사 없이 참조
```

## 학습 순서

1. **01_tasks.py** - 가장 기본. `@ray.remote`와 `ray.get()` 이해
2. **02_actors.py** - 상태가 필요할 때 Actor 사용
3. **03_object_store.py** - 대용량 데이터 다룰 때 필수
4. **04_wait_pattern.py** - 실전에서 자주 쓰는 패턴
5. **05_resources.py** - GPU 등 리소스 관리
6. **06_error_handling.py** - 프로덕션에서 중요
7. **07_actor_pool.py** - Actor 스케일링
