import random
import torch
import torchvision
from typing import Any, Callable, Dict, List, Optional, Tuple

from settings import DATASET_DIR


class CustomMNIST(torchvision.datasets.MNIST):
	def __init__(
		self, root: str = DATASET_DIR,
		train: bool = True,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = False) -> None:
		super().__init__(root, transform=transform, target_transform=target_transform, download=download)

		if self._check_legacy_exist():
			self.data, self.targets = self._load_legacy_data()
			return

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError("Dataset not found. You can use download=True to download it")

		self.data, self.targets = self._load_data()
		
	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		
		num_images = random.randint(4, 30)
		targets = []
		img = torch.zeros(30, 28, 28)
		
		for i in range(num_images):
			index = random.randint(0, len(self)-1)
			# images.append(self.data[index])
			img[i] = self.data[index]
			targets.append(int(self.targets[index]))

		# multichannel_image = torch
		label = 1 if 4 in targets else 0
			
		return img, torch.tensor(label)


if __name__ == '__main__':
	dataset = CustomMNIST()
	print(dataset)
	print(dataset[1])