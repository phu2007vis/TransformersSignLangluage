import torch.nn as nn
import collections
from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table
import torch

class VideoMAEEmbeddings(nn.Module):
	"""
	Construct the patch and position embeddings.

	"""

	def __init__(self, config):
		super().__init__()

		self.patch_embeddings = VideoMAEPatchEmbeddings(config)
		self.num_patches = self.patch_embeddings.num_patches
		# fixed sin-cos embedding 
     	#100 for additionall info like landmarks
		self.position_embeddings = get_sinusoid_encoding_table(self.num_patches+64, config.hidden_size)
		self.config = config

		
	def forward(self, pixel_values, bool_masked_pos,landmarks):
		
		# create patch embeddings
		embeddings = self.patch_embeddings(pixel_values,landmarks)
		t = embeddings.shape[1]
		# add position embeddings
		embeddings = embeddings + self.position_embeddings[:,:t,:].detach().type_as(embeddings).to(
			device=embeddings.device, copy=True
		)
		# only keep visible patches
		# ~bool_masked_pos means visible
		if bool_masked_pos is not None:
			batch_size, _, num_channels = embeddings.shape
			embeddings = embeddings[~bool_masked_pos]
			embeddings = embeddings.reshape(batch_size, -1, num_channels)

		return embeddings


class VideoMAEPatchEmbeddings(nn.Module):
	"""
	Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
	height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

	The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
	patch_size).

	"""

	def __init__(self, config):
		super().__init__()

		image_size = config.image_size
		patch_size = config.patch_size
		num_channels = config.num_channels
		hidden_size = config.hidden_size
		num_frames = config.num_frames
		tubelet_size = config.tubelet_size

		image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
		self.image_size = image_size
		self.patch_size = patch_size
		self.tubelet_size = int(tubelet_size)
		num_patches = (
			(image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
		)
		self.num_channels = num_channels
		self.num_patches = num_patches
		self.projection = nn.Conv3d(
			in_channels=num_channels,
			out_channels=hidden_size,
			kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
			stride=(self.tubelet_size, patch_size[0], patch_size[1]),
		)
		self.landmark_proj = nn.Linear(133*3,hidden_size)

	def forward(self, pixel_values,landmarks):
		batch_size, num_frames, num_channels, height, width = pixel_values.shape
		if num_channels != self.num_channels:
			raise ValueError(
				"Make sure that the channel dimension of the pixel values match with the one set in the configuration."
			)
		if height != self.image_size[0] or width != self.image_size[1]:
			raise ValueError(
				f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
			)
		# permute to (batch_size, num_channels, num_frames, height, width)
		pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
		embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
  
		if landmarks is not None:
			# B,T,133,3 -> B,T,133*3
			landmarks = self.landmark_proj(landmarks.flatten(-2))
			embeddings = torch.cat([embeddings,landmarks],dim = 1)
   
		return embeddings