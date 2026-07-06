import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class Gemma4ClippableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.use_clipped_linears = False
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Gemma4ClippableLinear(10, 10)

model = DummyModel()
config = LoraConfig(target_modules=["q_proj"])

try:
    peft_model = get_peft_model(model, config)
    print("Success! Trainable parameters:", sum(p.numel() for p in peft_model.parameters() if p.requires_grad))
except Exception as e:
    print("Error:", e)
