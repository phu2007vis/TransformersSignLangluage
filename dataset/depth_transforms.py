import torch
import torch.nn.functional as F




class DepthResize(torch.nn.Module):
	"""Horizontally flip landmarks randomly with a given probability.
	For landmarks, this means negating the x coordinates.
	
	Args:
		p (float): probability of the landmarks being flipped. Default value is 0.5
	"""

	def __init__(self,size = (224,224)):
		super().__init__()
		self.size = size

	def forward(self, depth):
		# Resize to (240, 320)
		depth = F.interpolate(depth, size=self.size, mode='nearest')  # or mode='bilinear', depending on need
		return depth
 