from datetime import datetime
from typing import Tuple, List

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import trange

from .evaluate import measure_accuracy


class FocalLoss(nn.Module):
	"""
	Focal Loss for addressing class imbalance in multi-class classification.
	Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
	
	Used to reduce loss from easy examples (well-classified samples) and focus 
	on hard examples (misclassified samples). Particularly effective for imbalanced 
	class distributions where the model tends to default to the majority class.
	
	Args:
		alpha: Tensor of shape (num_classes,) with weights for each class.
		       Typically inverse frequency weights. Higher alpha = higher loss weight.
		gamma: Focusing parameter (typically 2.0). Controls how much to down-weight 
		       easy examples. Higher gamma = more focus on hard examples.
	"""
	
	def __init__(self, alpha=None, gamma=2.0):
		super().__init__()
		if alpha is not None:
			self.register_buffer('alpha', alpha)
		else:
			self.alpha = None
		self.gamma = gamma
	
	def forward(self, inputs, targets):
		"""
		Args:
			inputs: Model logits of shape (batch_size, num_classes)
			targets: Ground truth labels of shape (batch_size,)
		
		Returns:
			Scalar loss value
		"""
		# Standard cross-entropy loss
		ce_loss = F.cross_entropy(inputs, targets, reduction='none')
		
		# Calculate probability of true class
		p_t = torch.exp(-ce_loss)
		
		# Apply focal term: (1 - p_t)^gamma
		focal_weight = (1 - p_t) ** self.gamma
		
		# Apply alpha weighting if provided
		if self.alpha is not None:
			alpha_t = self.alpha[targets]
			focal_loss = alpha_t * focal_weight * ce_loss
		else:
			focal_loss = focal_weight * ce_loss
		
		return focal_loss.mean()


class MultiTaskLoss(nn.Module):
	"""
	Multi-Task Loss for Hierarchical TGCN with Signal Detection and Direction Prediction.
	
	Combines two classification tasks with weighted balance:
	- Task 1 (Signal Detection): Noise vs Signal (applied to all samples)
	- Task 2 (Direction Prediction): Down vs Up (applied only to signal samples)
	
	The combined loss allows the shared backbone to learn features beneficial for both tasks.
	Task weighting enables fine-tuning emphasis between tasks.
	
	Args:
		lambda_direction (float): Weight for direction task loss (default: 1.0).
			- λ = 1.0: Equal importance for both tasks
			- λ < 1.0: Emphasize signal detection
			- λ > 1.0: Emphasize direction prediction
	
	Reference: Caruana (1997), "Multitask Learning" - IEEE Transactions on Pattern Analysis
	"""
	
	def __init__(self, lambda_direction: float = 1.0):
		super().__init__()
		self.lambda_direction = lambda_direction
		self.ce_loss = nn.CrossEntropyLoss(reduction='none')
	
	def forward(
		self,
		signal_logits: torch.Tensor,
		direction_logits: torch.Tensor,
		signal_targets: torch.Tensor,
		direction_targets: torch.Tensor,
		signal_mask: torch.Tensor = None
	) -> Tuple[torch.Tensor, dict]:
		"""
		Compute multi-task loss.
		
		Args:
			signal_logits: [N, 2] logits from signal detection head
			direction_logits: [N, 2] logits from direction prediction head
			signal_targets: [N] ground truth signal labels (0=Noise, 1=Signal)
			direction_targets: [N] ground truth direction labels (0=Down, 1=Up)
			signal_mask: [N] boolean mask indicating valid direction targets
				- True where signal_targets==1 (real signal samples)
				- False where signal_targets==0 (noise samples)
				If None, computes direction loss on all samples
		
		Returns:
			Tuple of (total_loss, loss_dict):
				- total_loss: Scalar loss value for backpropagation
				- loss_dict: Dict with breakdown {signal_loss, direction_loss, total_loss}
		
		Example:
			>>> criterion = MultiTaskLoss(lambda_direction=1.0)
			>>> signal_logits = torch.randn(100, 2)
			>>> direction_logits = torch.randn(100, 2)
			>>> signal_targets = torch.randint(0, 2, (100,))
			>>> direction_targets = torch.randint(0, 2, (100,))
			>>> signal_mask = (signal_targets == 1)
			>>> total_loss, loss_dict = criterion(
			...     signal_logits, direction_logits,
			...     signal_targets, direction_targets,
			...     signal_mask
			... )
		"""
		
		# ============================================
		# Task 1: Signal Detection Loss
		# ============================================
		# Applied to ALL samples (entire batch)
		# Goal: Distinguish noise (within ±threshold) from signals
		signal_loss = F.cross_entropy(signal_logits, signal_targets)
		
		# ============================================
		# Task 2: Direction Prediction Loss
		# ============================================
		# Applied ONLY to samples where signal_targets==1 (real signals)
		# Goal: Predict Down vs Up for detected signals only
		if signal_mask is not None:
			# Only compute loss for valid samples (where signal_targets == 1)
			num_signals = signal_mask.sum().item()
			if num_signals > 0:
				direction_loss = F.cross_entropy(
					direction_logits[signal_mask],
					direction_targets[signal_mask]
				)
			else:
				# No signal samples in batch - use zero loss with gradients
				direction_loss = torch.tensor(0.0, device=signal_logits.device, requires_grad=True)
		else:
			# If no mask provided, apply to all samples
			direction_loss = F.cross_entropy(direction_logits, direction_targets)
		
		# ============================================
		# Combined Loss
		# ============================================
		# Total loss = L_signal + λ * L_direction
		# where λ (lambda_direction) balances task importance
		total_loss = signal_loss + self.lambda_direction * direction_loss
		
		# ============================================
		# Loss Dictionary for Monitoring
		# ============================================
		loss_dict = {
			'signal_loss': signal_loss.item(),
			'direction_loss': direction_loss.item() if isinstance(direction_loss, torch.Tensor) else direction_loss,
			'total_loss': total_loss.item()
		}
		
		return total_loss, loss_dict


def train(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, num_epochs: int, task_title: str = "", measure_acc: bool = False) -> Tuple[List[float], List[float]]:
	"""
	Train function for a regression / classification model
	:param model: Model to train
	:param optimizer: Optimizer to use (Adam, ...)
	:param criterion: Loss function to use (MSE, CrossEntropy, ...)
	:param train_dataloader: Train data loader
	:param test_dataloader: Test data loader
	:param num_epochs: Number of epochs to train on the train dataset
	:param task_title: Title of the tensorboard run
	:param measure_acc: Whether to measure accuracy or not (for classification tasks)
	:return: Tuple of (train_losses_epoch, test_losses_epoch)
	"""
	writer = SummaryWriter(f'runs/{task_title}_{datetime.now().strftime("%d_%m_%Hh%M")}_{model.__class__.__name__}')
	train_losses_epoch = []
	test_losses_epoch = []
	
	for epoch in (pbar := trange(num_epochs, desc="Epochs")):
		train_loss = train_iteration(model, optimizer, pbar, criterion, train_dataloader, epoch, writer, measure_acc)
		test_loss = test_iteration(model, criterion, test_dataloader, epoch, writer, measure_acc)
		train_losses_epoch.append(train_loss)
		test_losses_epoch.append(test_loss)
	
	return train_losses_epoch, test_losses_epoch


def test_iteration(model: nn.Module, criterion: nn.Module, test_dataloader: DataLoader, epoch: int, writer: SummaryWriter, measure_acc: bool = False) -> float:
	"""
	Test iteration
	:param model: Model to test
	:param criterion: Loss function to use (MSE, CrossEntropy, ...)
	:param test_dataloader: Test data loader
	:param epoch: Current epoch
	:param writer: Tensorboard writer
	:param measure_acc: Whether to measure accuracy or not (for classification tasks)
	:return: Average loss for the epoch
	"""
	model.eval()
	device = next(model.parameters()).device # get model device: cpu or gpu
	total_loss = 0
	num_batches = 0
	
	for idx, data in enumerate(test_dataloader):
		data = data.to(device)
		out = model(data.x, data.edge_index, data.edge_weight)
		labels = data.y.long()  # 3-class labels: 0=Downward, 1=Neutral, 2=Upward
		loss = criterion(out, labels)
		writer.add_scalar("Loss/Test Loss", loss.item(), epoch * len(test_dataloader) + idx)
		total_loss += loss.item()
		num_batches += 1
		if measure_acc:
			acc = measure_accuracy(model, data)
			writer.add_scalar("Accuracy/Test Accuracy", acc, epoch * len(test_dataloader) + idx)
	
	avg_loss = total_loss / num_batches if num_batches > 0 else 0
	return avg_loss


def train_iteration(model: nn.Module, optimizer: optim.Optimizer, pbar: trange, criterion: nn.Module, train_dataloader: DataLoader, epoch: int, writer: SummaryWriter, measure_acc: bool = False) -> float:
	"""
	Train iteration
	:param model: Model to train
	:param optimizer: Optimizer to use (Adam, ...)
	:param pbar: tqdm progress bar
	:param criterion: Loss function to use (MSE, CrossEntropy, ...)
	:param train_dataloader: Train data loader
	:param epoch: Current epoch
	:param writer: Tensorboard writer
	:param measure_acc: Whether to measure accuracy or not (for classification tasks)
	:return: Average loss for the epoch
	"""
	model.train()
	device = next(model.parameters()).device # get model device: cpu or gpu
	total_loss = 0
	num_batches = 0
	
	for idx, data in enumerate(train_dataloader):
		data = data.to(device)
		optimizer.zero_grad()
		out = model(data.x, data.edge_index, data.edge_weight)
		labels = data.y.long()  # 3-class labels: 0=Downward, 1=Neutral, 2=Upward
		loss = criterion(out, labels)
		loss.backward()
		optimizer.step()
		pbar.set_postfix({"Batch": f"{(idx + 1) / len(train_dataloader) * 100:.1f}%"})
		writer.add_scalar("Loss/Train Loss", loss.item(), epoch * len(train_dataloader) + idx)
		total_loss += loss.item()
		num_batches += 1
		if measure_acc:
			acc = measure_accuracy(model, data)
			writer.add_scalar("Accuracy/Train Accuracy", acc, epoch * len(train_dataloader) + idx)
	
	avg_loss = total_loss / num_batches if num_batches > 0 else 0
	return avg_loss
