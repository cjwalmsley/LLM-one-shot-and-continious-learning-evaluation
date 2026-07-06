import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

class DummyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.ones_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = DummyLinear(10, 10)

    def forward(self, x):
        return self.q_proj(x)

model = DummyModel()
config = LoraConfig(target_modules=["q_proj.linear"])
peft_model = get_peft_model(model, config)

for name, param in peft_model.named_parameters():
    if "lora" in name:
        nn.init.ones_(param)

x = torch.ones(1, 10)

# Disable test
peft_model.eval()
with peft_model.disable_adapter():
    out1 = peft_model(x).sum().item()

# Check adapter mode
out2 = peft_model(x).sum().item()

print(f"Disabled adapter output: {out1}")
print(f"Enabled adapter output: {out2}")
