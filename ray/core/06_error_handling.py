"""
에러 처리와 재시도
"""

import ray
import random

ray.init()


@ray.remote(max_retries=3, retry_exceptions=[ValueError])
def flaky_task(task_id: int) -> str:
    if random.random() < 0.7:
        raise ValueError(f"Task {task_id} failed randomly")
    return f"Task {task_id} succeeded"


print("=== Retry on failure ===")
refs = [flaky_task.remote(i) for i in range(5)]

for ref in refs:
    try:
        result = ray.get(ref)
        print(f"  {result}")
    except ray.exceptions.RayTaskError as e:
        print(f"  Failed after retries: {e.cause}")


@ray.remote
def always_fails() -> None:
    raise RuntimeError("This always fails")


print("\n=== Exception propagation ===")
try:
    ray.get(always_fails.remote())
except ray.exceptions.RayTaskError as e:
    print(f"Caught: {type(e.cause).__name__}: {e.cause}")


# ray.get with timeout
@ray.remote
def slow_task() -> str:
    import time
    time.sleep(10)
    return "done"


print("\n=== Timeout ===")
try:
    ray.get(slow_task.remote(), timeout=1)
except ray.exceptions.GetTimeoutError:
    print("Task timed out after 1 second")

ray.shutdown()
