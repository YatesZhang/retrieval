# Open Flamingo
environment
```
conda env create -f environment.yml
```

```python
lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",    #  attention layer in LLaMa
                   "to_q", "to_kv", "to_out",    # gate cross layer attention 
                    "ff.1", "ff.3"],    # 
tuning_config = dict(
    r=16,
    lora_alpha=16,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
     "to_q", "to_kv", "to_out",
      "ff.1", "ff.3"],
    lora_dropout=0.0,
    bias="none",
    modules_to_save=[],
    task_type="VL",
    )

```
LLaMa
```
root
├── model (LlamaModel)
│   ├── embed_tokens (Embedding) weight:[32003, 4096]
│   ├── layers (ModuleList)
│   │   └── 0-31(LlamaDecoderLayer)
│   │       ├── self_attn (LlamaAttention)
│   │       │   └── q_proj,k_proj,v_proj,o_proj(Linear) weight:[4096, 4096]
│   │       ├── mlp (LlamaMLP)
│   │       │   ├── gate_proj,up_proj(Linear) weight:[11008, 4096]
│   │       │   └── down_proj (Linear) weight:[4096, 11008]
│   │       └── input_layernorm,post_attention_layernorm(LlamaRMSNorm) weight:[4096]
│   └── norm (LlamaRMSNorm) weight:[4096]
└── lm_head (Linear) weight:[32003, 4096]
```
gated_cross_attn_layer
```
        │   │       ├── gated_cross_attn_layer (GatedCrossAttentionBlock) attn_gate:[1] ff_gate:[1]
        │   │       │   ├── attn (MaskedCrossAttention)
        │   │       │   │   ├── norm (LayerNorm) weight:[4096] bias:[4096]
        │   │       │   │   ├── to_q (Linear) weight:[512, 4096]
        │   │       │   │   │   ├── lora_dropout (ModuleDict)
        │   │       │   │   │   ├── lora_A (ModuleDict)
        │   │       │   │   │   │   └── default (Linear) weight:[16, 4096]
        │   │       │   │   │   └── lora_B (ModuleDict)
        │   │       │   │   │       └── default (Linear) weight:[512, 16]
        │   │       │   │   ├── to_kv (Linear) weight:[1024, 1024]
        │   │       │   │   │   ├── lora_dropout (ModuleDict)
        │   │       │   │   │   ├── lora_A (ModuleDict)
        │   │       │   │   │   │   └── default (Linear) weight:[16, 1024]
        │   │       │   │   │   └── lora_B (ModuleDict)
        │   │       │   │   │       └── default (Linear) weight:[1024, 16]
        │   │       │   │   └── to_out (Linear) weight:[4096, 512]
        │   │       │   │       ├── lora_dropout (ModuleDict)
        │   │       │   │       ├── lora_A (ModuleDict)
        │   │       │   │       │   └── default (Linear) weight:[16, 512]
        │   │       │   │       └── lora_B (ModuleDict)
        │   │       │   │           └── default (Linear) weight:[4096, 16]
        │   │       │   └── ff (Sequential)
        │   │       │       ├── 0 (LayerNorm) weight:[4096] bias:[4096]
        │   │       │       ├── 1 (Linear) weight:[16384, 4096]
        │   │       │       │   ├── lora_dropout (ModuleDict)
        │   │       │       │   ├── lora_A (ModuleDict)
        │   │       │       │   │   └── default (Linear) weight:[16, 4096]
        │   │       │       │   └── lora_B (ModuleDict)
        │   │       │       │       └── default (Linear) weight:[16384, 16]
        │   │       │       └── 3 (Linear) weight:[4096, 16384]
        │   │       │           ├── lora_dropout (ModuleDict)
        │   │       │           ├── lora_A (ModuleDict)
        │   │       │           │   └── default (Linear) weight:[16, 16384]
        │   │       │           └── lora_B (ModuleDict)
        │   │       │               └── default (Linear) weight:[4096, 16]
```