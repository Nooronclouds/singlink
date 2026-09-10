# gpu_detective.py
import os
import subprocess
import sys
import platform

def comprehensive_gpu_diagnostic():
    print("🕵️ COMPREHENSIVE GPU DETECTION DIAGNOSTIC")
    print("=" * 70)
    
    # System Info
    print(f"\n💻 SYSTEM INFORMATION:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Python: {sys.version}")
    
    # NVIDIA Driver Check
    print(f"\n🖥️ NVIDIA DRIVER CHECK:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA Driver: Working")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"   {line.strip()}")
                if 'CUDA Version' in line:
                    print(f"   {line.strip()}")
        else:
            print("❌ NVIDIA Driver: Not working properly")
    except Exception as e:
        print(f"❌ NVIDIA Driver: Error - {e}")
    
    # CUDA Installation Check
    print(f"\n🔧 CUDA TOOLKIT CHECK:")
    cuda_versions = ['13.0', '12.0', '11.8', '11.0', '10.0']
    cuda_found = False
    
    for version in cuda_versions:
        cuda_path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{version}"
        if os.path.exists(cuda_path):
            cuda_found = True
            print(f"✅ CUDA {version}: Found at {cuda_path}")
            
            # Check critical files
            critical_files = [
                os.path.join(cuda_path, "bin", "nvcc.exe"),
                os.path.join(cuda_path, "bin", "cudart64_*.dll"),
                os.path.join(cuda_path, "lib", "x64", "cudart.lib")
            ]
            
            for file_pattern in critical_files:
                import glob
                if glob.glob(file_pattern):
                    print(f"   ✅ {os.path.basename(file_pattern)}: Present")
                else:
                    print(f"   ❌ {os.path.basename(file_pattern)}: Missing")
            
            break
    
    if not cuda_found:
        print("❌ CUDA Toolkit: Not installed in standard locations")
        print("   💡 You have NVIDIA drivers but need CUDA Toolkit")
    
    # Environment Variables
    print(f"\n📁 ENVIRONMENT VARIABLES:")
    env_vars = ['CUDA_PATH', 'PATH', 'CUDA_PATH_V11_8', 'CUDA_PATH_V12_0']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH' and 'CUDA' in value:
            print(f"✅ {var}: Contains CUDA paths")
        elif var != 'PATH':
            print(f"{'✅' if value != 'Not set' else '❌'} {var}: {value}")
    
    # TensorFlow Check
    print(f"\n🧠 TENSORFLOW CHECK:")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow Version: {tf.__version__}")
        
        # Check build info
        print(f"✅ TensorFlow Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"✅ TensorFlow Built with GPU: {tf.test.is_built_with_gpu_support()}")
        
        # Check devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU Devices: {len(gpus)} found")
            for gpu in gpus:
                print(f"   {gpu}")
        else:
            print("❌ GPU Devices: None detected")
            print("\n🔍 REASON ANALYSIS:")
            if not cuda_found:
                print("   → CUDA Toolkit not installed")
            elif not tf.test.is_built_with_cuda():
                print("   → TensorFlow not built with CUDA support")
            else:
                print("   → CUDA/GPU libraries not accessible")
                
    except ImportError as e:
        print(f"❌ TensorFlow Import Error: {e}")
    except Exception as e:
        print(f"❌ TensorFlow Check Error: {e}")
    
    # DLL Check
    print(f"\n📚 DLL AVAILABILITY CHECK:")
    critical_dlls = [
        "cudart64_110.dll", "cudart64_11.dll", "cudart64_12.dll",
        "cudnn64_8.dll", "cublas64_11.dll", "cufft64_10.dll"
    ]
    
    path_dirs = os.environ.get('PATH', '').split(';')
    dll_found = False
    
    for dll in critical_dlls:
        for path_dir in path_dirs:
            if os.path.exists(os.path.join(path_dir, dll)):
                print(f"✅ {dll}: Found in {path_dir}")
                dll_found = True
                break
        else:
            print(f"❌ {dll}: Not found in PATH")
    
    if not dll_found:
        print("💡 No critical CUDA DLLs found in PATH")

    print("\n" + "=" * 70)
    print("🎯 RECOMMENDED SOLUTION:")
    if not cuda_found:
        print("Download and install CUDA Toolkit 11.8 from:")
        print("https://developer.nvidia.com/cuda-11-8-0-download-archive")
    else:
        print("Install cuDNN 8.6.0 and add to PATH")
        print("https://developer.nvidia.com/cudnn")

if __name__ == "__main__":
    comprehensive_gpu_diagnostic()