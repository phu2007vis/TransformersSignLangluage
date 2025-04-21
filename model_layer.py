from typing import  Optional, Tuple, Union
import torch

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