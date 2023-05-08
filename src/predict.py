import torch
from torch import nn
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


def predict_setup_relation_net():

	# TODO: Parameterize the function.

	test_loader = DataLoader(
		GANTripletDataset(mode='test'),
		batch_size=8,
		pin_memory=True,
		shuffle=True,
	)

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
	
	return clf_model, test_loader


def test_loop(
	model, 
	test_loader, 
	checkpoint_file):

	checkpoint = torch.load(f=checkpoint_file)
	model.load_state_dict(state_dict=checkpoint["model_state_dict"])

	model.eval()
	with torch.no_grad():
		with tqdm(
			enumerate(test_loader),
			total=len(test_loader),
			desc="Prediction"
		) as tqdm_eval:
			for _, (
				imgs_real,
				imgs_generated
			) in tqdm_eval:
				
				predictions = model.to(DEVICE)(imgs_real.to(DEVICE), imgs_generated.to(DEVICE)).detach().data
				print(predictions)


def run_predict():
	
	# Relation Net
	model, test_loader = predict_setup_relation_net()
	test_loop(model, test_loader, checkpoint_file='/home/miruna/.dumps/nag-implementation/repository/logs/relation-net-1/weights/last_model.pth')

run_predict()