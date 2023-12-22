""" 
    utils
"""
import torch 
import os.path as osp 
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus 
from rich.console import Console


def path_finder(PATHS):
    """
        auto find path in PATHS
        代码检查无法通过FileNotFoundError
    """
    for path in PATHS:
        if osp.exists(path):
            return path
    raise OSError("path {} not found".format(path))

def maybe_zero_3(param):
    """ 
        zero stage == 3 will partition the model weight to different rank
    """
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """ 
        reference: DeepSpeed FastChat:
            https://github.com/lm-sys/FastChat/blob/4960ca702c66b9adaa65945746dba34f8d2c8ddc/fastchat/train/train_lora.py#L66
        only save huggingface Peft model's LoRA weight:
            1) lora_A
            2) lora_B
            3) lora_dropout
            4) lora_embedding_A
            5) lora_embedding_B
        usage:

    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def get_lora_weight_only(peft_model):
    """ 
        1) only get state dict of LoRA weight 
        2) support get state dict of DeepSpeed Zero stage 3 model

        return state_dict of LoRA weight 
    """
    bias = peft_model.peft_config['default'].bias
    state_dict = get_peft_state_maybe_zero_3(peft_model.named_parameters(), bias=bias)
    return state_dict 