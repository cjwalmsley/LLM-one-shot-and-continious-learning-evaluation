import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

class Gemma4ClippableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.use_clipped_linears = False
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # Initialize to ones for predictable testing
        nn.init.ones_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Gemma4ClippableLinear(10, 10)

    def forward(self, x):
        return self.q_proj(x)

model = DummyModel()
config = LoraConfig(target_modules=["q_proj.linear"])
peft_model = get_peft_model(model, config)

# Train the adapter slightly to see if output changes
x = torch.ones(1, 10)
print("Base output:", peft_model(x).sum().item())

# Manually alter LoRA weights to simulate fine-tuning
for name, param in peft_model.named_parameters():
    if "lora_B" in name:
        nn.init.ones_(param)
    elif "lora_A" in name:
        nn.init.ones_(param)

print("Output with adapter:", peft_model(x).sum().item())

with peft_model.disable_adapter():
    print("Output with adapter disabled:", peft_model(x).sum().item())
