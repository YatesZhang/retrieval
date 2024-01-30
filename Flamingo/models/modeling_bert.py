import torch
from torch import nn
from open_flamingo import Flamingo
from open_flamingo.flamingo_lm import FlamingoLayer
from open_flamingo.src.utils import getattr_recursive, setattr_recursive


class FlamingoLayerForMaskedLM(FlamingoLayer):
    def __init__(
        self,
        gated_cross_attn_layer,
        decoder_layer, 
        gradient_checkpointing=False
    ):
        """ 
            Flamingo Model for Masked Language Modeling
        """
        super().__init__(
            gated_cross_attn_layer=gated_cross_attn_layer, 
            decoder_layer=decoder_layer, 
            gradient_checkpointing=gradient_checkpointing)
    
    def forward(
        self,
        lang_x,    # hidden_states
        attention_mask,
        *args,
        **decoder_layer_kwargs,
    ):  
        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError(
                    "media_locations must be conditioned before forward pass"
                )

            lang_x = self.gated_cross_attn_layer(
                lang_x,
                self.vis_x,
                media_locations=self.media_locations,
                use_cached_media=self.use_cached_media,
            )

        # Normal decoder layer
        """ 
        @transformers.model.bert.modeling_bert.BertEncoder.forward
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        """
        lang_x = self.decoder_layer(
            lang_x, attention_mask, *args, **decoder_layer_kwargs
        )
        return lang_x


# class FlamingoForMaskedLM(Flamingo):
class FlamingoForMaskedLM(object):
    def __init__(
        self,
        # vision_encoder: nn.Module,
        # lang_encoder: nn.Module,
        # eoc_token_id: int,
        # media_token_id: int,
        # vis_dim: int,
        # cross_attn_every_n_layers: int = 1,
        # gradient_checkpointing: bool = False,
    ):
        # super().__init__(vision_encoder=vision_encoder,
        #                 lang_encoder=lang_encoder,
        #                 eoc_token_id=eoc_token_id,
        #                 media_token_id=media_token_id,
        #                 vis_dim=vis_dim,
        #                 cross_attn_every_n_layers=cross_attn_every_n_layers,
        #                 gradient_checkpointing=gradient_checkpointing)
        pass 
    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        use_cache: bool = False,
        **kwargs
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)
            self._condition_media_locations(input_ids=lang_x)
        # import pdb 
        # pdb.set_trace()
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            # use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output
    def init_layers_for_masked_lm(self):
        pass 
    # def generate(*args, **kwargs):
    #     raise NotImplementedError