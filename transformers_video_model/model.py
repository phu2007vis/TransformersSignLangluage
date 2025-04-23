

"""PyTorch VideoMAE (masked autoencoder) model."""
#Basic import
from copy import deepcopy
from typing import  Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


# Transformers utils
from transformers.utils import logging
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput

#add the path for easy import
import sys
import os
#add parrent dir to path
sys.path.append(os.path.dirname(__file__))

#abtract class to keep the data
from catalog import VideoMAEConfig,VideoMAEPreTrainedModel,VideoMAEDecoderOutput,VideoMAEForPreTrainingOutput
#model layer
from layers import VideoMAELayer
from patch_embedding import VideoMAEEmbeddings
#log info
logger = logging.get_logger(__name__)



# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->VideoMAE
class VideoMAEEncoder(nn.Module):
	def __init__(self, config: VideoMAEConfig) -> None:
		super().__init__()
		self.config = config
		self.layer = nn.ModuleList([VideoMAELayer(config) for _ in range(config.num_hidden_layers)])
		self.gradient_checkpointing = False

	def forward(
		self,
		hidden_states: torch.Tensor,
		head_mask: Optional[torch.Tensor] = None,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	) -> Union[tuple, BaseModelOutput]:
		all_hidden_states = () if output_hidden_states else None
		all_self_attentions = () if output_attentions else None
		
		for i, layer_module in enumerate(self.layer):
			if output_hidden_states:
				all_hidden_states = all_hidden_states + (hidden_states,)

			layer_head_mask = head_mask[i] if head_mask is not None else None

			if self.gradient_checkpointing and self.training:
				layer_outputs = self._gradient_checkpointing_func(
					layer_module.__call__,
					hidden_states,
					layer_head_mask,
					output_attentions,
				)
			else:
				layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

			hidden_states = layer_outputs[0]

			if output_attentions:
				all_self_attentions = all_self_attentions + (layer_outputs[1],)

		if output_hidden_states:
			all_hidden_states = all_hidden_states + (hidden_states,)

		if not return_dict:
			return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
		return BaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=all_hidden_states,
			attentions=all_self_attentions,
		)





class VideoMAEModel(VideoMAEPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.config = config

		self.embeddings = VideoMAEEmbeddings(config)
		self.encoder = VideoMAEEncoder(config)

		if config.use_mean_pooling:
			self.layernorm = None
		else:
			self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.embeddings.patch_embeddings

	def _prune_heads(self, heads_to_prune,landmarks  ):
		"""
		Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
		class PreTrainedModel
		"""
		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)


	def forward(
		self,
		pixel_values: torch.FloatTensor,
		landmarks = None,
		bool_masked_pos: Optional[torch.BoolTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, BaseModelOutput]:
		r"""
		bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Each video in the
			batch must have the same number of masked patches. If `None`, then all patches are considered. Sequence
			length is `(num_frames // tubelet_size) * (image_size // patch_size) ** 2`.

		Returns:

		Examples:

		```python
		>>> import av
		>>> import numpy as np

		>>> from transformers import AutoImageProcessor, VideoMAEModel
		>>> from huggingface_hub import hf_hub_download

		>>> np.random.seed(0)


		>>> def read_video_pyav(container, indices):
		...     '''
		...     Decode the video with PyAV decoder.
		...     Args:
		...         container (`av.container.input.InputContainer`): PyAV container.
		...         indices (`List[int]`): List of frame indices to decode.
		...     Returns:
		...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
		...     '''
		...     frames = []
		...     container.seek(0)
		...     start_index = indices[0]
		...     end_index = indices[-1]
		...     for i, frame in enumerate(container.decode(video=0)):
		...         if i > end_index:
		...             break
		...         if i >= start_index and i in indices:
		...             frames.append(frame)
		...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


		>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
		...     '''
		...     Sample a given number of frame indices from the video.
		...     Args:
		...         clip_len (`int`): Total number of frames to sample.
		...         frame_sample_rate (`int`): Sample every n-th frame.
		...         seg_len (`int`): Maximum allowed index of sample's last frame.
		...     Returns:
		...         indices (`List[int]`): List of sampled frame indices
		...     '''
		...     converted_len = int(clip_len * frame_sample_rate)
		...     end_idx = np.random.randint(converted_len, seg_len)
		...     start_idx = end_idx - converted_len
		...     indices = np.linspace(start_idx, end_idx, num=clip_len)
		...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
		...     return indices


		>>> # video clip consists of 300 frames (10 seconds at 30 FPS)
		>>> file_path = hf_hub_download(
		...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
		... )
		>>> container = av.open(file_path)

		>>> # sample 16 frames
		>>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
		>>> video = read_video_pyav(container, indices)

		>>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
		>>> model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

		>>> # prepare video for the model
		>>> inputs = image_processor(list(video), return_tensors="pt")

		>>> # forward pass
		>>> outputs = model(**inputs)
		>>> last_hidden_states = outputs.last_hidden_state
		>>> list(last_hidden_states.shape)
		[1, 1568, 768]
		```"""
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

		embedding_output = self.embeddings(pixel_values, bool_masked_pos,landmarks)

		encoder_outputs = self.encoder(
			embedding_output,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
	
		sequence_output = encoder_outputs[0]
		if self.layernorm is not None:
			sequence_output = self.layernorm(sequence_output)

		if not return_dict:
			return (sequence_output,) + encoder_outputs[1:]

		return BaseModelOutput(
			last_hidden_state=sequence_output,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
		)



class VideoMAEForVideoClassification(VideoMAEPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.num_labels = config.num_labels
		self.videomae = VideoMAEModel(config)

		# Classifier head
		self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
		self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

		# Initialize weights and apply final processing
		self.post_init()

	def forward(
		self,
		landmarks : Optional[torch.Tensor] = None,
		pixel_values: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		
	) -> Union[Tuple, ImageClassifierOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

		Returns:

		Examples:

		```python
		>>> import av
		>>> import torch
		>>> import numpy as np

		>>> from transformers import AutoImageProcessor, VideoMAEForVideoClassification
		>>> from huggingface_hub import hf_hub_download

		>>> np.random.seed(0)


		>>> def read_video_pyav(container, indices):
		...     '''
		...     Decode the video with PyAV decoder.
		...     Args:
		...         container (`av.container.input.InputContainer`): PyAV container.
		...         indices (`List[int]`): List of frame indices to decode.
		...     Returns:
		...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
		...     '''
		...     frames = []
		...     container.seek(0)
		...     start_index = indices[0]
		...     end_index = indices[-1]
		...     for i, frame in enumerate(container.decode(video=0)):
		...         if i > end_index:
		...             break
		...         if i >= start_index and i in indices:
		...             frames.append(frame)
		...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


		>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
		...     '''
		...     Sample a given number of frame indices from the video.
		...     Args:
		...         clip_len (`int`): Total number of frames to sample.
		...         frame_sample_rate (`int`): Sample every n-th frame.
		...         seg_len (`int`): Maximum allowed index of sample's last frame.
		...     Returns:
		...         indices (`List[int]`): List of sampled frame indices
		...     '''
		...     converted_len = int(clip_len * frame_sample_rate)
		...     end_idx = np.random.randint(converted_len, seg_len)
		...     start_idx = end_idx - converted_len
		...     indices = np.linspace(start_idx, end_idx, num=clip_len)
		...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
		...     return indices


		>>> # video clip consists of 300 frames (10 seconds at 30 FPS)
		>>> file_path = hf_hub_download(
		...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
		... )
		>>> container = av.open(file_path)

		>>> # sample 16 frames
		>>> indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
		>>> video = read_video_pyav(container, indices)

		>>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
		>>> model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

		>>> inputs = image_processor(list(video), return_tensors="pt")

		>>> with torch.no_grad():
		...     outputs = model(**inputs)
		...     logits = outputs.logits

		>>> # model predicts one of the 400 Kinetics-400 classes
		>>> predicted_label = logits.argmax(-1).item()
		>>> print(model.config.id2label[predicted_label])
		eating spaghetti
		```"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.videomae(
			pixel_values,
			landmarks = landmarks,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
	
		)

		sequence_output = outputs[0]

		if self.fc_norm is not None:
			sequence_output = self.fc_norm(sequence_output.mean(1))
		else:
			sequence_output = sequence_output[:, 0]
		
		logits = self.classifier(sequence_output)
		
		loss = None
		if labels is not None:
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(logits.squeeze(), labels.squeeze())
				else:
					loss = loss_fct(logits, labels)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(logits, labels)

		if not return_dict:
			output = (logits,) + outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return ImageClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)



__all__ = ["VideoMAEForPreTraining", "VideoMAEModel", "VideoMAEPreTrainedModel", "VideoMAEForVideoClassification"]
