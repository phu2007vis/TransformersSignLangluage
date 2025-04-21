

from typing import  Optional, Tuple
from dataclasses import dataclass

import torch.nn as nn
import torch

from transformers.utils import ModelOutput
from transformers.models.videomae.configuration_videomae import VideoMAEConfig
from transformers.modeling_utils import  PreTrainedModel

@dataclass
class VideoMAEDecoderOutput(ModelOutput):

	logits: Optional[torch.FloatTensor] = None
	hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VideoMAEForPreTrainingOutput(ModelOutput):
	loss: Optional[torch.FloatTensor] = None
	logits: Optional[torch.FloatTensor] = None
	hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	attentions: Optional[Tuple[torch.FloatTensor]] = None
 
 

class VideoMAEPreTrainedModel(PreTrainedModel):
	"""
	An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
	models.
	"""

	config_class = VideoMAEConfig
	base_model_prefix = "videomae"
	main_input_name = "pixel_values"
	supports_gradient_checkpointing = True
	_supports_sdpa = True
	_supports_flash_attn_2 = True

	def _init_weights(self, module):
		"""Initialize the weights"""
		if isinstance(module, (nn.Linear, nn.Conv3d)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)