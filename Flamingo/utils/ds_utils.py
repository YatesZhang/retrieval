"""  
    a utils to get DeepSpeed Config

    BF16 vs FP16:
    BF16: 1 sign + 8 exponent + 7 mantissa
    FP16: 1 sign + 5 exponent + 10 fraction(mantissa)
    see this blog: https://medium.com/@furkangozukara/what-is-the-difference-between-fp16-and-bf16-here-a-good-explanation-for-you-d75ac7ec30fa

    LoRA tuning:
    huggingface: https://huggingface.co/docs/transformers/v4.15.0/performance

    If you have tried to finetune models pre-trained under bf16 mixed precision (e.g. T5) 
    it’s very likely that you have encountered overflow issues. 
    Now you should be able to finetune those models without any issues.

That said, also be aware that if you pre-trained a model in bf16, it’s likely to have overflow issues if someone tries to finetune it in fp16 down the road. So once started on the bf16-mode path it’s best to remain on it and not switch to fp16.
"""

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(args,
                        offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512):
    """ 
        requires:
            precision
            enable_tensorboard
    """
    if args.precision == 'fp16':
        enable_fp16 = True
        enable_bf16 = False
    elif args.precision == 'bf16':
        enable_fp16 = False
        enable_bf16 = True
    else:
        raise ValueError(f"Invalid precision {args.precision}")
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 0,
        "memory_efficient_linear": False,
    }
    output =  {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "fp16": {
            "enabled": enable_fp16,
            "loss_scale_window": 100
        },
        "bf16": {
            "enabled": enable_bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },

    }
    if args.enable_tensorboard:
        output.update({"tensorboard": {
            "enabled": True,
            "output_path": args.output_dir,
            "job_name": 'tb_logging'
        }}
        )
    return output

def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ],
                                     small_learning_rate_list=
                                     ["embed"], small_lr=1e-4):
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and (not any(nd in n
                            for nd in small_learning_rate_list)) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and (not any(nd in n
                            for nd in small_learning_rate_list)) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and (any(nd in n
                            for nd in small_learning_rate_list)) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
            "lr": small_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and (any(nd in n
                            for nd in small_learning_rate_list)) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
            "lr": small_lr
        },
    ]
    return optimizer_grouped_parameters