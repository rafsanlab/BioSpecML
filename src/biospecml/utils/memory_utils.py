import GPUtil
import psutil

def print_gpu_memory():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUsed} MB used, {gpu.memoryFree} MB free, {gpu.memoryTotal} MB total")

# Function to print CPU memory usage
def print_cpu_memory():
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / 1024 ** 2:.2f} MB")
    print(f"Used memory: {memory.used / 1024 ** 2:.2f} MB")
    print(f"Free memory: {memory.free / 1024 ** 2:.2f} MB")
    print(f"Memory usage: {memory.percent}%")
