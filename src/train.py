import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm
from statistics import mean

from utils import evaluate
from datasets import GANTripletDataset
from pipelines.relation_network import RelationNetwork
from blocks import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_setup_relation_net():

	# TODO: Parameterize the function.

	train_loader = DataLoader(
		GANTripletDataset(mode='train'),
		batch_size=8,
		pin_memory=True,
		shuffle=True,
	)

	val_loader = DataLoader(
		GANTripletDataset(mode='validate'),
		batch_size=2,
		shuffle=False
	)

	loss_function = nn.BCELoss()

	backbone_net_one = resnet101(
		big_kernel=True,
		use_fc=False,
		use_pooling=False,
		zero_init_residual=True
	)
	backbone_net_two = resnet101(
		big_kernel=True,
		use_fc=False,
		use_pooling=False,
		zero_init_residual=True
	)
	clf_model = RelationNetwork(
		backbone_one=backbone_net_one, 
		backbone_two=backbone_net_two,
		use_softmax=True
	).to(DEVICE)

	# default params adopted from the base paper.
	scheduler_milestones = [120, 160]
	scheduler_gamma = 0.1
	learning_rate = 1e-5

	train_optimizer = SGD(
		clf_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
	)
	train_scheduler = MultiStepLR(
		train_optimizer,
		milestones=scheduler_milestones,
		gamma=scheduler_gamma,
	)
	
	return clf_model, train_loader, val_loader, loss_function, train_optimizer, train_scheduler 


def train_loop(
	model, 
	train_loader, 
	val_loader, 
	loss_function, 
	optimizer,
	scheduler,
	n_epochs=100,
	checkpoint_file=None):

	if checkpoint_file is not None:
		checkpoint = torch.load(f=checkpoint_file)
		model.load_state_dict(state_dict=checkpoint["model_state_dict"])
		# optimizer.load_state_dict(state_dict=checkpoint["optimizer_state_dict"])
		# scheduler.load_state_dict(state_dict=checkpoint["scheduler_state_dict"])
		print(f"Checkpoint loaded from {checkpoint_file}")

	def train_epoch(dataloader):  

		all_loss = []
		model.train()
		
		with tqdm(
			enumerate(dataloader), total=len(dataloader), desc="Training"
		) as tqdm_train:
			for step_idx, (
				imgs_real,
				imgs_generated,
				similarity_scores 
			) in tqdm_train:
				
				optimizer.zero_grad()
				computed_scores = model(imgs_real.to(DEVICE), imgs_generated.to(DEVICE))
				# print(computed_scores)
				# print(similarity_scores)
				loss = loss_function(computed_scores, similarity_scores.float().to(DEVICE))
				loss.backward()
				optimizer.step()

				all_loss.append(loss.item())
				tqdm_train.set_postfix(loss=mean(all_loss))

		return mean(all_loss)


	weights_dir = Path(os.path.join(os.path.split(__file__)[0], '../logs/relation-net-1/weights'))
	tb_logs_dir = Path(os.path.join(os.path.split(__file__)[0], '../logs/relation-net-1'))
	tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

	best_state = model.state_dict()
	best_validation_accuracy = 0.0
	for epoch in range(n_epochs):
		
		print(f"Epoch {epoch}")
		epoch_output_path = os.path.join(weights_dir, f"weights-1_{epoch}.pth")
		
		average_loss = train_epoch(train_loader)
		validation_auc, validation_accuracy = evaluate(
			model, val_loader, device=DEVICE, tqdm_prefix="Validation"
		)

		if validation_auc > best_validation_accuracy:
			best_validation_accuracy = validation_accuracy
			best_state = model.state_dict()
			
			torch.save(
                obj={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1
                },
                f=os.path.join(weights_dir, f"weights-1_{epoch}.pth")
			)
			print("New 'best' model saved!")
		
		else:
			torch.save(
				obj={
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"epoch": epoch + 1
				},
				f=os.path.join(weights_dir, f"last_model.pth")
			)

		tb_writer.add_scalar("Train/loss", average_loss, epoch)
		tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
		tb_writer.add_scalar("Val/auc", validation_auc, epoch)
		print(f"Acc: {validation_accuracy}\tAUC: {validation_auc}")

		scheduler.step()


def run_train():
	
	# Relation Net
	model, train_loader, val_loader, loss_fn, optimizer, scheduler = train_setup_relation_net()
	train_loop(model, train_loader, val_loader, loss_fn, optimizer, scheduler, checkpoint_file=os.path.join(os.path.split(__file__)[0], '../logs/relation-net-1/weights/weights-1_0.pth'))

run_train()
