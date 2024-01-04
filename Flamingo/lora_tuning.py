"""
    add LoRA weight for Flamingo
    #TODO:
    add visual prompt for CLIP visual encoder 
"""

from typing import Optional
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from open_flamingo import Flamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from bigmodelvis import Visualization
from huggingface_hub import hf_hub_download
from Flamingo.models.decoupled_flamingo import DecoupledFlamingo
import pdb 

def get_tokenizer(
    tokenizer_path,
    cache_dir="/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo",
    use_local_files=False,
):
    """
        get tokenizer
    """
    text_tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    local_files_only=use_local_files,
    trust_remote_code=True,
    cache_dir=cache_dir)
    return text_tokenizer


def create_model_and_transforms(
    clip_vision_encoder_path="",
    clip_vision_encoder_pretrained="",
    lang_encoder_path="",
    tokenizer_path="",
    cross_attn_every_n_layers=1,
    use_local_files=False,
    decoder_layers_attr_name=None,
    freeze_lm_embeddings=False,
    cache_dir=None,
    lora_tuning=False,
    add_eos_token=True,
    decoupled=False, 
    **flamingo_kwargs):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.
    
    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    from rich import print 
    global_rank = -1
    try: 
        global_rank = torch.distributed.get_rank()
    except RuntimeError:
        print("[yellow]Flamingo will use single GPU or CPU[/yellow]")
    # print("gloabl_rank:", global_rank)
    print("[[bold yellow]@rank{}[/bold yellow]|create Flamingo] create vision_encoder and image_processor from open_clip".format(global_rank))
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,    # "ViT-L-14"
        pretrained=clip_vision_encoder_pretrained,    # "openai"
        cache_dir=cache_dir,
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True
    print("[[bold yellow]@rank{}[/bold yellow]|create Flamingo] create text_tokenizer".format(global_rank))
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    print("[[bold yellow]@rank{}[/bold yellow]|create Flamingo] create LLM from ".format(global_rank), lang_encoder_path)
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:
        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte
            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    print("[[bold yellow]@rank{}[/bold yellow]|create Flamingo] create Flamingo with cross_attn_every_n_layers=".format(global_rank),
           cross_attn_every_n_layers)
    if not decoupled:
        model = Flamingo(
            vision_encoder,
            lang_encoder,
            text_tokenizer.encode("<|endofchunk|>")[-1],
            text_tokenizer.encode("<image>")[-1],
            vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
                "width"
            ],
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            **flamingo_kwargs)
    else:
        model = DecoupledFlamingo(
            lang_encoder,
            text_tokenizer.encode("<|endofchunk|>")[-1],
            text_tokenizer.encode("<image>")[-1],
            vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
                "width"
            ],
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            **flamingo_kwargs
        )
    # load checkpoint:
    print("[[bold yellow]@rank{global_rank}[/bold yellow]|create Flamingo] load checkpoint.pt from huggingface ".format(global_rank=global_rank))
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b",
     "checkpoint.pt",
      cache_dir=cache_dir)
    # checkpoint_path = "/home/yunzhi/yunzhi/yunzhi/checkpoints/flamingo/checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    print("[[bold yellow]@rank{global_rank}[/bold yellow]|create Flamingo] Freeze all parameters ".format(global_rank=global_rank))
    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # --------------------------------------------------------------------------
    from peft import LoraConfig, get_peft_model
    

    lora_target_modules=["Wqkv", "to_q", "to_kv", "to_out", "ff.1", "ff.3"]
    tuning_config = dict(
        r=16,
        lora_alpha=16,
        target_modules=lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        modules_to_save=[],
        task_type="VL",
    )
    tuning_config = LoraConfig(
        **tuning_config
    )
    print("[[bold yellow]@rank{global_rank}[/bold yellow]|create Flamingo] LoRa tuning mode: ".format(global_rank=global_rank),
     lora_tuning)
    if lora_tuning:
        print("[[bold yellow]@rank{global_rank}[/bold yellow]|LoRA tuning config] LoRa tuning adaptor injection: ".format(global_rank=global_rank),
         lora_target_modules)
        model = get_peft_model(model, peft_config=tuning_config)
        model.print_trainable_parameters()
    else:
        print("[[bold yellow]@rank{global_rank}[/bold yellow]|set requires_grad] No LoRA adaptor, unfrozen the gate cross attention layer".format(global_rank=global_rank))
        print("[[bold yellow]@rank{global_rank}[/bold yellow]|set requires_grad] unfrozen perceiver layer".format(global_rank=global_rank))
        model.perceiver.requires_grad_(True)
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if global_rank == 0:
        Visualization(lang_encoder).structure_graph()
    
    # --------------------------------------------------------------------------
    if not add_eos_token:
        text_tokenizer.eos_token = None

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    # model.perceiver.requires_grad_(True)
    # model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_trainable_params = num_trainable_params / (1024 * 1024)
    print(
        "Flamingo model initialized with {}MB trainable parameters".format(num_trainable_params)
    )

    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    """
        inject tools
    """
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        "We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}
