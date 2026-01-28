"""
Ray Object Store: 대용량 데이터의 효율적 공유
"""

import ray
import numpy as np

ray.init()


@ray.remote
def process_chunk(data_ref, start: int, end: int) -> float:
    data = data_ref[start:end]
    return float(np.mean(data))


# 대용량 데이터를 Object Store에 저장
large_array = np.random.rand(10_000_000)
data_ref = ray.put(large_array)  # Object Store에 한 번만 저장

print(f"Data size: {large_array.nbytes / 1e6:.1f} MB")
print(f"ObjectRef: {data_ref}")

# 여러 태스크가 동일 데이터 참조 (복사 없음)
chunk_size = len(large_array) // 4
futures = [
    process_chunk.remote(data_ref, i * chunk_size, (i + 1) * chunk_size)
    for i in range(4)
]

results = ray.get(futures)
print(f"Chunk means: {results}")
print(f"Overall mean: {np.mean(results):.6f}")
print(f"Direct mean:  {np.mean(large_array):.6f}")

ray.shutdown()
