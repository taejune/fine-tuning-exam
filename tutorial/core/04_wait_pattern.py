"""
ray.wait(): 완료된 태스크부터 처리하기
"""

import ray
import time
import random

ray.init()


@ray.remote
def unpredictable_task(task_id: int) -> dict:
    duration = random.uniform(0.5, 2.0)
    time.sleep(duration)
    return {"id": task_id, "duration": round(duration, 2)}


print("=== ray.get() - 모두 완료될 때까지 대기 ===")
refs = [unpredictable_task.remote(i) for i in range(5)]
start = time.time()
results = ray.get(refs)
print(f"All done in {time.time() - start:.2f}s")
for r in results:
    print(f"  Task {r['id']}: {r['duration']}s")


print("\n=== ray.wait() - 완료되는 순서대로 처리 ===")
refs = [unpredictable_task.remote(i) for i in range(5)]
start = time.time()

pending = refs
while pending:
    done, pending = ray.wait(pending, num_returns=1)
    result = ray.get(done[0])
    elapsed = time.time() - start
    print(f"  [{elapsed:.2f}s] Task {result['id']} completed (took {result['duration']}s)")

ray.shutdown()
