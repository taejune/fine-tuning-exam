"""
Resource 관리: CPU, GPU, 커스텀 리소스 할당
"""

import ray

ray.init()

print(f"Available resources: {ray.available_resources()}")


@ray.remote(num_cpus=2)
def cpu_intensive(x: int) -> int:
    return x * x


@ray.remote(num_cpus=1, num_gpus=0.5)
def gpu_task(x: int) -> int:
    return x * 2


@ray.remote(resources={"custom_accelerator": 1})
def custom_resource_task(x: int) -> int:
    return x + 100


# CPU 태스크
result = ray.get(cpu_intensive.remote(5))
print(f"CPU task result: {result}")

# GPU 태스크 (GPU 없으면 스킵)
if ray.available_resources().get("GPU", 0) > 0:
    result = ray.get(gpu_task.remote(5))
    print(f"GPU task result: {result}")
else:
    print("GPU not available, skipping gpu_task")


# Actor에도 리소스 지정 가능
@ray.remote(num_cpus=1)
class ResourceActor:
    def compute(self, x: int) -> int:
        return x * 3


actor = ResourceActor.remote()
print(f"Actor result: {ray.get(actor.compute.remote(7))}")

ray.shutdown()
