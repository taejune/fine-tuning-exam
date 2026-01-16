"""
ActorPool: Actor 풀로 작업 분배
"""

import ray
from ray.util import ActorPool

ray.init()


@ray.remote
class Processor:
    def __init__(self, processor_id: int):
        self.processor_id = processor_id

    def process(self, item: int) -> dict:
        return {
            "processor": self.processor_id,
            "input": item,
            "output": item * item
        }


processors = [Processor.remote(i) for i in range(3)]
pool = ActorPool(processors)

items = list(range(10))

print("=== map (ordered) ===")
results = pool.map(lambda actor, item: actor.process.remote(item), items)
for r in results:
    print(f"  {r}")

print("\n=== map_unordered (as completed) ===")
results = pool.map_unordered(lambda actor, item: actor.process.remote(item), items)
for r in results:
    print(f"  {r}")

ray.shutdown()
