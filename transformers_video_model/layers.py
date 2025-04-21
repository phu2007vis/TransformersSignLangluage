from catalog import VideoMAEConfig
from transformers.models.videomae.modeling_videomae import VideoMAEAttention
from transformers.activations import ACT2FN
from typing import Optional,Tuple,Union
import torch
from torch import nn

# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->VideoMAE
class VideoMAESelfOutput(nn.Module):
	"""
	The residual connection is defined in VideoMAELayer instead of here (as is the case with other models), due to the
	layernorm applied before each block.
	"""

	def __init__(self, config: VideoMAEConfig) -> None:
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)

		return hidden_states
# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->VideoMAE
class VideoMAEOutput(nn.Module):
	def __init__(self, config: VideoMAEConfig) -> None:
		super().__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)

		hidden_states = hidden_states + input_tensor

		return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->VideoMAE
class VideoMAEIntermediate(nn.Module):
	def __init__(self, config: VideoMAEConfig) -> None:
		super().__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		if isinstance(config.hidden_act, str):
			self.intermediate_act_fn = ACT2FN[config.hidden_act]
		else:
			self.intermediate_act_fn = config.hidden_act

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)

		return hidden_states



# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->VideoMAE,VIT->VIDEOMAE
class VideoMAELayer(nn.Module):
	"""This corresponds to the Block class in the timm implementation."""

	def __init__(self, config: VideoMAEConfig) -> None:
		super().__init__()
		self.chunk_size_feed_forward = config.chunk_size_feed_forward
		self.seq_len_dim = 1
		self.attention = VideoMAEAttention(config)
		self.intermediate = VideoMAEIntermediate(config)
		self.output = VideoMAEOutput(config)
		self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

	def forward(
		self,
		hidden_states: torch.Tensor,
		head_mask: Optional[torch.Tensor] = None,
		output_attentions: bool = False,
	) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
		self_attention_outputs = self.attention(
			self.layernorm_before(hidden_states),  # in VideoMAE, layernorm is applied before self-attention
			head_mask,
			output_attentions=output_attentions,
		)
		attention_output = self_attention_outputs[0]
		outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

		# first residual connection
		hidden_states = attention_output + hidden_states

		# in VideoMAE, layernorm is also applied after self-attention
		layer_output = self.layernorm_after(hidden_states)
		layer_output = self.intermediate(layer_output)

		# second residual connection is done here
		layer_output = self.output(layer_output, hidden_states)

		outputs = (layer_output,) + outputs

		return outputs