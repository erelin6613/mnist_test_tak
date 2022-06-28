import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as t_models

from settings import IN_CHANNELS, NUM_CLASSES


loss_fn = nn.BCELoss() if NUM_CLASSES == 1 else nn.CrossEntropyLoss()


class BaseModel(nn.Module):
	def train_step(self, batch, device="cpu"):
		images, targets = batch 
		out = self(images.to(device))
		loss = loss_fn(out.to(device), targets.to(device).view(-1, 1).float())
		return out, loss

	def validation_step(self):
		images, targets = batch 
		out = self(images)                      
		loss = loss_fn(out.long(), targets.view(-1, 1).float())
		return out, loss


class ResNet(BaseModel):
	def __init__(
		self,
		in_channels=IN_CHANNELS,
		classes=NUM_CLASSES,
		pretrained_backone=True):

		super().__init__()
		backbone = t_models.resnet34(pretrained=pretrained_backone)
		self.in_channels = in_channels
		self.classes = classes
		self.model = self._modify_resnet(backbone)

	def _modify_resnet(self, model):
		model.conv1 = nn.Conv2d(
			in_channels=self.in_channels,
			out_channels=64,
			kernel_size=(7, 7),
			stride=(2, 2),
			padding=(3, 3),
			bias=False)

		model.fc = nn.Linear(
			in_features=512,
			out_features=self.classes,
			bias=True)

		return model

	def forward(self, x):
		x = self.model(x)
		return nn.Sigmoid()(x)

	def freeze(self):
		for param in self.model.parameters():
			param.require_grad = False
		for param in self.model.fc.parameters():
			param.require_grad = True
		for param in self.model.conv1.parameters():
			param.require_grad = True
	
	def unfreeze(self):
		for param in self.model.parameters():
			param.require_grad = True


if __name__ == '__main__':
	model = ResNet()