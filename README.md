# Open Flamingo
## environment
```
conda env create -f environment.yml
```
## fine tuning demo on flan-t5-small
- 使用int8和fp16混合精度训练flan-t5-small
- 使用bitsandbytes库进行混合精度训练时，对int8自动反量化有额外时间开销使训练时间变长
```bash
cd tuninglab
CUDA_VISIBLE_DEVICES=0 python flan_t5.py
```
|config|GPU cost| Time cost|
|-|-|-|
| enable_int8 = True|876MiB| 10min54s/epoch|
| enable_int8 = False|2104MiB| 3min36s/epoch|
## train on single GPU 
```bash
CUDA_VISIVLE_DEVICES=0 torchrun \
--nproc_per_node=1 \
--nnodes=1 \
fine_tuning.py \
  --run_name exp_1 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine \
  --workers 1 \
  --batch_size 1 \ 
```

```bash
cd tuninglab
python flamingo.py \
  --run_name exp_1 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine \
  --workers 1 \
  --batch_size 1 \ 
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