import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn import functional as F


# Model configuration
class SelfAttentionConfig(PretrainedConfig):
	model_type = "self_attention"
	def __init__(self, hidden_size,config, **kwargs):
		super().__init__()
		self.config = config

		self.hidden_size = hidden_size
		self.num_heads = self.config['num_heads']
		self.num_layers = self.config['num_layers']
		self.max_len = self.config['max_len']
		self.dropout = self.config['dropout']
		self.num_keypoints = self.config['num_keypoints']

# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.num_heads = config.num_heads
		self.head_size = config.hidden_size // config.num_heads
		self.query = nn.Linear(config.hidden_size, config.hidden_size)
		self.key = nn.Linear(config.hidden_size, config.hidden_size)
		self.value = nn.Linear(config.hidden_size, config.hidden_size)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		batch, seq, _ = x.size()
		q = self.query(x).view(batch, seq, self.num_heads, self.head_size).transpose(1, 2)
		k = self.key(x).view(batch, seq, self.num_heads, self.head_size).transpose(1, 2)
		v = self.value(x).view(batch, seq, self.num_heads, self.head_size).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_size ** 0.5)
		
		attn = F.softmax(scores, dim=-1)
		attn = self.dropout(attn)
		out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, -1)
		return out

# Encoder Layer
class EncoderLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.attn = MultiHeadAttention(config)
		self.norm1 = nn.LayerNorm(config.hidden_size)
		self.ffn = nn.Sequential(
			nn.Linear(config.hidden_size, config.hidden_size * 4),
			nn.GELU(),
			nn.Linear(config.hidden_size * 4, config.hidden_size),
		)
		self.norm2 = nn.LayerNorm(config.hidden_size)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		x = self.norm1(x + self.dropout(self.attn(x)))
		x = self.norm2(x + self.dropout(self.ffn(x)))
		return x

# Self-Attention Encoder
class LanmarkEncoder(PreTrainedModel):
	
	def __init__(self,hidden_size, config ):
		config = SelfAttentionConfig(hidden_size,config)
		super().__init__(config)
		self.projection = nn.Linear(config.num_keypoints,config.hidden_size)
		self.pos_embed = nn.Embedding(config.max_len, config.hidden_size)
		self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
		self.norm = nn.LayerNorm(config.hidden_size)
		self.dropout = nn.Dropout(config.dropout)
		self.init_weights()

	def forward(self, seq):
		B,T,V,C= seq.size()
		seq  = seq.flatten(2)
		seq = self.projection(seq) 
  
		positions = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, T)
		pos_emb = self.pos_embed(positions)
		
		# Add positional embeddings and apply dropout
		x = self.dropout(self.norm(seq + pos_emb))

		for layer in self.layers:
			x = layer(x)
		return x
