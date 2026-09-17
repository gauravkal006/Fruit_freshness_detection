import os
import sys
import multiprocessing
import torch

# Ensure UTF-8 output encoding for terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def get_optimized_hardware_config():
    """
    Detect local CPU/GPU hardware acceleration resources and return optimized training parameters.
    """
    cuda_available = torch.cuda.is_available()
    device_name = "cpu"
    device_label = "CPU Mode"
    
    if cuda_available:
        device_name = "cuda:0"
        device_label = f"NVIDIA GPU Acceleration ({torch.cuda.get_device_name(0)})"
        torch.backends.cudnn.benchmark = True
    else:
        cpu_count = multiprocessing.cpu_count() or 4
        try:
            torch.set_num_threads(cpu_count)
        except Exception:
            pass
        device_label = f"Multi-threaded CPU ({cpu_count} logical cores)"
        
    cpu_cores = multiprocessing.cpu_count() or 4
    optimal_workers = max(2, min(cpu_cores, 8))
    
    config = {
        "device": device_name,
        "device_label": device_label,
        "cuda_available": cuda_available,
        "cpu_cores": cpu_cores,
        "workers": optimal_workers,
        "batch_size": 64 if cuda_available else 32,
        "imgsz": 224
    }
    
    print("=" * 65)
    print("⚡ LOCAL HARDWARE RESOURCE OPTIMIZER ACTIVATED")
    print(f"  * Execution Device: {config['device_label']}")
    print(f"  * CPU Core Threads: {config['cpu_cores']} cores allocated")
    print(f"  * Dataloader Workers: {config['workers']} parallel workers")
    print(f"  * Recommended Batch Size: {config['batch_size']}")
    print("=" * 65)
    
    return config

if __name__ == "__main__":
    get_optimized_hardware_config()
