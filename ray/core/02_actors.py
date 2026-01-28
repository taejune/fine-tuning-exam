"""
Ray Actors: 상태를 유지하는 원격 객체
"""

import ray

ray.init()


@ray.remote
class Counter:
    def __init__(self, start: int = 0):
        self.value = start

    def increment(self) -> int:
        self.value += 1
        return self.value

    def get(self) -> int:
        return self.value


counter = Counter.remote(start=10)

# 메서드 호출도 .remote() 사용
refs = [counter.increment.remote() for _ in range(5)]
print(f"Increment results: {ray.get(refs)}")
print(f"Final value: {ray.get(counter.get.remote())}")


# 여러 Actor 인스턴스
@ray.remote
class Worker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.tasks_done = 0

    def process(self, data: str) -> str:
        self.tasks_done += 1
        return f"Worker-{self.worker_id} processed '{data}' (total: {self.tasks_done})"


print("\n=== Multiple Actors ===")
workers = [Worker.remote(i) for i in range(3)]

results = []
for i, item in enumerate(["a", "b", "c", "d", "e", "f"]):
    worker = workers[i % len(workers)]
    results.append(worker.process.remote(item))

for r in ray.get(results):
    print(r)

ray.shutdown()
