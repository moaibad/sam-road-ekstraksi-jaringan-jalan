import torch

# Periksa apakah CUDA tersedia
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA is not available.")
