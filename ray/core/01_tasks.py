"""
Ray Tasks: 함수를 원격 실행하는 가장 기본적인 단위
"""

import ray
import time

ray.init()


@ray.remote
def slow_square(x: int) -> int:
    time.sleep(1)
    return x * x


# 순차 실행 vs 병렬 실행 비교
print("=== Sequential (Python) ===")
start = time.time()
results_seq = [slow_square.remote(i) for i in range(4)]
# .remote()는 즉시 반환 (ObjectRef)
# ray.get()으로 실제 결과 대기
results = ray.get(results_seq)
print(f"Results: {results}")
print(f"Time: {time.time() - start:.2f}s (4개 태스크가 병렬 실행)")


# 태스크 의존성
@ray.remote
def add(a: int, b: int) -> int:
    return a + b


x_ref = slow_square.remote(2)
y_ref = slow_square.remote(3)
z_ref = add.remote(x_ref, y_ref)  # x, y 완료 후 실행

print(f"\n=== Task Dependency ===")
print(f"2² + 3² = {ray.get(z_ref)}")

ray.shutdown()
