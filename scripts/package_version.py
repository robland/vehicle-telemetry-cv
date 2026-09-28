import torch
import torchreid

print("PyTorch :", torch.__version__)
print("Torchreid :", torchreid.__version__)
print("CUDA :", torch.cuda.is_available())

torchreid.models.show_avai_models()