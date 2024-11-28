import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should display "NVIDIA GeForce RTX 3050 Laptop GPU"
