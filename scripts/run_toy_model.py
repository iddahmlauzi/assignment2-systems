import torch
import torch.nn as nn
import modal
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image, user_volume
from cs336_basics.loss import cross_entropy
from cs336_basics.optim import AdamW

wandb_secret = modal.Secret.from_name("wandb")


class ToyModel(nn.Module):
 def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 10, bias=False)
    self.ln = nn.LayerNorm(10)
    self.fc2 = nn.Linear(10, out_features, bias=False)
    self.relu = nn.ReLU()
    
 def forward(self, x):
    
    x = self.relu(self.fc1(x))
    print(f"Output of first feedforward layer: {x.dtype}")
    x = self.ln(x)
    print(f"Output of layer norm: {x.dtype}")
    x = self.fc2(x)
    print(f"Predicted logits: {x.dtype}")
    return x


@app.function(image=build_image(), 
              volumes=VOLUME_MOUNTS, 
              max_containers=1,
              gpu="B200", 
              secrets=[wandb_secret],
              timeout=2700)
def run_model():
    # Only hardcoding values cause this is a one time thing
    batch_size = 4
    in_features = 128
    out_features = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(batch_size, in_features, device=device)
    y = torch.randint(low=0, high=128, size=(batch_size,), device=device)
    
    model = ToyModel(in_features=in_features, out_features=out_features)
    model.to(device)
    optimizer = AdamW(params=model.parameters(), device=device)
    
    print("Model Parameter dtypes:")
    print(f" - fc1 dtype: {model.fc1.weight.dtype}")
    print(f" - ln dtype: {model.ln.weight.dtype}")
    print(f" - fc2 dtype: {model.fc2.weight.dtype}")
    
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    print(f"Loss dtype: {loss.dtype}")
    print("Gradient dtypes:")
    for name, param in model.named_parameters():
        print(f" - {name} dtype: {param.grad.dtype}")
    optimizer.step()
        
    

@app.local_entrypoint()
def main():
    print("Running the model")
    run_model.remote()

    