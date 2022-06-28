from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from dataset import CustomMNIST
from models import ResNet
import settings

import warnings
warnings.filterwarnings("ignore")


def add_logs(df, record):
	df = df.append(record, ignore_index=True)
	return df


def f_score(pred_batch, gt_batch, threshold=0.5):
	prob = pred_batch > threshold

	tp = (prob & gt_batch).sum().float()
	tn = ((~prob) & (~gt_batch)).sum().float()
	fp = (prob & (~gt_batch)).sum().float()
	fn = ((~prob) & gt_batch).sum().float()

	precision = torch.mean(tp / (tp + fp + 1e-12))
	recall = torch.mean(tp / (tp + fn + 1e-12))
	score = 2 * precision * recall / (precision + recall + 1e-12)
	return score.mean(0)


def run_batch(model, epoch, batch, optimizer, df, stage, p_bar, device=torch.device("cpu")):
	out, loss = model.train_step(batch, device)
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	score = f_score(out, batch[1])
	results = {
		"loss": loss.detach().cpu().item(),
		"f_score": score.cpu().item()}
	p_bar.set_postfix(results)
	results["epoch"] = epoch
	results["stage"] = stage
	df = add_logs(df, results)
	return model, df


def split_dataset(dataset, size=[0.7, 0.15, 0.15], return_loaders=True):
	sizes = [int(x * len(dataset)) for x in size]
	train_set, val_set, test_set = random_split(dataset, sizes)

	if return_loaders:
		train_loader = DataLoader(
			train_set, batch_size=settings.BATCH_SIZE, num_workers=2, shuffle=True)
		val_loader = DataLoader(
			val_set, batch_size=settings.BATCH_SIZE, num_workers=2, shuffle=False)
		test_loader = DataLoader(
			test_set, batch_size=settings.BATCH_SIZE, num_workers=2, shuffle=False)
		return train_loader, val_loader, test_loader

	return train_set, val_set, test_set


def trainining_pipeline(experiment_name="base"):

	device = torch.device(
		"cuda") if torch.cuda.is_available() else torch.device("cpu")

	log_dir = Path(settings.LOG_DIR).joinpath(experiment_name)
	log_dir.mkdir(parents=True, exist_ok=True)
	log_df = pd.DataFrame()

	dataset = CustomMNIST()
	train_loader, val_loader, test_loader = split_dataset(dataset)
	model = ResNet().to(device)
	optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

	if not log_dir.joinpath("model_last.pth").exists():
		for epoch in range(settings.EPOCHS):
			
			model.train()
			p_bar = tqdm(train_loader, desc=f"Train epoch {epoch+1}")
			for batch in p_bar:
				model, log_df = run_batch(model, epoch, batch, optimizer, log_df, "train", p_bar, device)

			model.eval()
			p_bar = tqdm(val_loader, desc="Validation")
			for batch in tqdm(val_loader):
				model, log_df = run_batch(model, epoch, batch, optimizer, log_df, "valid", p_bar, device)

			torch.save(model, log_dir.joinpath(f"model_{epoch}.pth"))

		model.eval()
		p_bar = tqdm(test_loader, desc="Testing")
		for batch in test_loader:
			model, log_df = run_batch(
				model, epoch, batch, optimizer, log_df, "test", p_bar)

		torch.save(model, log_dir.joinpath("model_last.pth"))
		log_df.to_csv(log_dir.joinpath("logs.csv"))
	else:
		log_df = pd.read_csv(log_dir.joinpath("logs.csv"))

	test_score = log_df.groupby("stage").mean()["f_score"][0]
	print(f"Test F1 score: {test_score}")



if __name__ == '__main__':
	trainining_pipeline("resnet34")