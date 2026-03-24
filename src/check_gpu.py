"""
GPU Setup Verification Script
Checks if PyTorch can detect and use your GPU
"""

import torch
import sys

def check_gpu_setup():
    print("=" * 60)
    print("PyTorch GPU Setup Verification")
    print("=" * 60)
    
    # PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # GPU details
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Current device
        print(f"\nCurrent CUDA Device: {torch.cuda.current_device()}")
        print(f"Current Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Test tensor creation on GPU
        print("\n" + "=" * 60)
        print("Testing GPU Tensor Operations...")
        print("=" * 60)
        
        try:
            # Create a tensor on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✓ Successfully created tensors on GPU")
            print(f"✓ Matrix multiplication successful")
            print(f"  Result shape: {z.shape}")
            print(f"  Device: {z.device}")
            
            # Test memory allocation
            print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.4f} GB")
            
            print("\n" + "=" * 60)
            print("✓ GPU setup is working correctly!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ Error during GPU operations: {str(e)}")
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("⚠ WARNING: CUDA is not available!")
        print("=" * 60)
        print("\nPossible reasons:")
        print("1. PyTorch CPU version is installed instead of GPU version")
        print("2. NVIDIA GPU drivers are not installed")
        print("3. CUDA toolkit is not installed or incompatible")
        print("\nPlease check your installation.")
        sys.exit(1)

if __name__ == "__main__":
    check_gpu_setup()
