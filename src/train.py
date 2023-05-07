from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils import evaluate
from pipelines.relation_network import RelationNetwork
from blocks import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_setup_relation_net():

	# TODO: Parameterize the function.

	loss_function = nn.CrossEntropyLoss()

	backbone_net = resnet101()
	clf_model = RelationNetwork(backbone_net).to(DEVICE)

	# default params adopted from the base paper.
	n_epochs = 100
	scheduler_milestones = [120, 160]
	scheduler_gamma = 0.1
	learning_rate = 1e-2
	tb_logs_dir = Path(".")

	train_optimizer = SGD(
		clf_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
	)
	train_scheduler = MultiStepLR(
		train_optimizer,
		milestones=scheduler_milestones,
		gamma=scheduler_gamma,
	)
	tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))
	
	return model, dataloader, loss_function, train_optimizer 


def train_loop(model, train_loader, val_loader, loss_function, optimizer):

	def train_epoch(dataloader):    
		all_loss = []
		model.train()
		
		with tqdm(
			enumerate(dataloader), total=len(dataloader), desc="Training"
		) as tqdm_train:
			for step_idx, (
				img_real,
				img_generate,
				sim_score 
			) in tqdm_train:
				
				optimizer.zero_grad()
				classification_scores = model(query_images.to(DEVICE))

				loss = loss_function(classification_scores, query_labels.to(DEVICE))
				loss.backward()
				optimizer.step()

				all_loss.append(loss.item())
				tqdm_train.set_postfix(loss=mean(all_loss))

		return mean(all_loss)

	best_state = model.state_dict()
	best_validation_accuracy = 0.0
	for epoch in range(n_epochs):
		print(f"Epoch {epoch}")
		average_loss = train_epoch(train_loader)
		validation_accuracy = evaluate(
			model, val_loader, device=DEVICE, tqdm_prefix="Validation"
		)

		if validation_accuracy > best_validation_accuracy:
			best_validation_accuracy = validation_accuracy
			best_state = model.state_dict()
			print("New 'best' model!")

		tb_writer.add_scalar("Train/loss", average_loss, epoch)
		tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

		train_scheduler.step()


def run_train():
	
	# Relation Net
	model, train_loader, val_loader, loss_fn, optimizer = train_setup_relation_net()
	train_loop(model, train_loader, val_loader, loss_fn, optimizer)