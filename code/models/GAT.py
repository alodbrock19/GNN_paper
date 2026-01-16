import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
	"""
	Single layer GAT model optimized for T-GCN.
	Uses explicit heads=1 and concat=False to ensure stable output dimensions.
	"""
	def __init__(self, in_channels: int, layer_sizes: list[int] = None, bias: bool = True):
		super(GAT, self).__init__()
		layer_sizes = layer_sizes or [32, 32]
		# Single layer to avoid vanishing gradients (T-GCN best practice)
		# heads=1 and concat=False ensure output shape = layer_sizes[0] (not heads*layer_sizes[0])
		self.convs = nn.ModuleList([
		   GATv2Conv(in_channels, layer_sizes[0], heads=1, concat=False, bias=bias),
		])

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
		"""
		Performs a forward pass on the GAT model.
		:param x: The feature matrix of the graph X_t (Nodes_nb, Features_nb)
		:param edge_index: The edge index of the graph A (2, Edges_nb)
		:param edge_weight: The edge weight of the graph (Edges_nb,)
		:return: The hidden state of the GAT h_t (Nodes_nb, Hidden_size)
		"""
		# Reshape edge_weight from 1D (Edges_nb,) to 2D (Edges_nb, 1) for GATv2Conv
		edge_attr = edge_weight.unsqueeze(-1)
		# Single layer with activation
		return F.leaky_relu(self.convs[0](x, edge_index, edge_attr))