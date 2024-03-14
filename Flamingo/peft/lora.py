""" 
see: DoRA: Weight-Decomposed Low-Rank Adaptation

reference: https://github.com/rasbt/dora-from-scratch/blob/main/Using-LinearDoRA.ipynb

DoRA official:
https://github.com/nbasyl/DoRA

huggingface:
https://huggingface.co/docs/peft/en/developer_guides/lora#weight-decomposed-low-rank-adaptation-dora

"""

import torch 
from torch import nn


def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.m = nn.Parameter(torch.ones(1, linear.out_features))

    def forward(self, x):
        linear_output = self.linear(x)
        lora_output = self.lora(x)
        lora_output_norm = lora_output / (lora_output.norm(p=2, dim=1, keepdim=True) + 1e-9)
        dora_modification = self.m * lora_output_norm
        return linear_output + dora_modification