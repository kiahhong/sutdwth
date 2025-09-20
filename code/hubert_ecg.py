import torch
import torch.nn as nn
from transformers import HubertConfig, HubertModel
from transformers.models.hubert.modeling_hubert import _compute_mask_indices
from typing import List, Optional, Union, Tuple
from transformers.modeling_outputs import BaseModelOutput

class HuBERTECGConfig(HubertConfig):
    def __init__(self,
                 # --- FIX 1: Changed vocab_size to vocab_sizes and made it a list ---
                 vocab_sizes=[100],
                 # --- FIX 2: Added the missing ensemble_length parameter ---
                 ensemble_length=1,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout=0.1,
                 activation_dropout=0.1,
                 attention_dropout=0.1,
                 feat_proj_dropout=0.0,
                 final_dropout=0.1,
                 layerdrop=0.1,
                 initializer_range=0.02,
                 layer_norm_eps=1e-5,
                 feat_extract_norm="group",
                 feat_extract_activation="gelu",
                 conv_dim=(512, 512, 512, 512, 512, 512, 512),
                 conv_stride=(5, 2, 2, 2, 2, 2, 2),
                 conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                 conv_bias=False,
                 num_conv_pos_embeddings=128,
                 num_conv_pos_embedding_groups=16,
                 do_stable_layer_norm=False,
                 classifier_proj_size=256,
                 apply_spec_augment=True,
                 mask_time_prob=0.05,
                 mask_time_length=10,
                 mask_time_min_masks=2,
                 mask_feature_prob=0.0,
                 mask_feature_length=10,
                 mask_feature_min_masks=0,
                 ctc_loss_reduction="sum",
                 ctc_zero_infinity=False,
                 conv_pos_batch_norm=False,
                 **kwargs):
        
        # --- FIX 3: Pass the first vocab size to the parent class for compatibility ---
        super().__init__(vocab_size=vocab_sizes[0], **kwargs)
        
        # --- FIX 4: Assign all the new and corrected attributes ---
        self.vocab_sizes = vocab_sizes
        self.ensemble_length = ensemble_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.layerdrop = layerdrop
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.do_stable_layer_norm = do_stable_layer_norm
        self.classifier_proj_size = classifier_proj_size
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.conv_pos_batch_norm = conv_pos_batch_norm
        
class HuBERTECG(HubertModel):
    def __init__(self, config: HuBERTECGConfig):
        super().__init__(config)
        self.config = config

        # --- FIX 5: Use the correct attribute names from the fixed config ---
        self.pretraining_vocab_size = config.vocab_sizes[0]
            
        assert config.ensemble_length > 0 and config.ensemble_length == len(config.vocab_sizes), f"ensemble_length {config.ensemble_length} must be equal to len(vocab_sizes) {len(config.vocab_sizes)}"

        self.final_proj = nn.ModuleList([nn.Linear(config.hidden_size, config.classifier_proj_size) for _ in range(config.ensemble_length)])

        self.label_embedding = nn.ModuleList([nn.Embedding(vocab_size, config.classifier_proj_size) for vocab_size in config.vocab_sizes])
        # --- END OF FIXES IN THIS CLASS ---
        
        assert len(self.final_proj) == len(self.label_embedding), f"final_proj and label_embedding must have the same length"

    # ... (The rest of the file remains exactly the same) ...

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states
        batch_size, sequence_length, hidden_size = hidden_states.size()
        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        if self.config.mask_feature_prob > 0 and self.training:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0
        return hidden_states, mask_time_indices
        
    def logits(self, transformer_output: torch.Tensor) -> torch.Tensor:
        projected_outputs = [final_projection(transformer_output) for final_projection in self.final_proj]
        ensemble_logits = [torch.cosine_similarity(
            projected_output.unsqueeze(2),
            label_emb.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        ) / 0.1 for projected_output, label_emb in zip(projected_outputs, self.label_embedding)]
        return ensemble_logits
    
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
        hidden_states = self.feature_projection(extract_features)
        hidden_states, mask_time_indices = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if not return_dict:
            output = (hidden_states,) + encoder_outputs[1:]
            if mask_time_indices is not None:
                output = output + mask_time_indices
            return output
        final_dict = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        final_dict["mask_time_indices"] = mask_time_indices
        return final_dict