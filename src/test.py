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


def test_setup_relation_net():

	# TODO: Parameterize the function.

	test_dataset = GANTripletDataset(mode='validate')

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
	
	return clf_model, test_dataset


def test_loop(
	model, 
	test_dataset, 
	checkpoint_file):

	checkpoint = torch.load(f=checkpoint_file)
	model.load_state_dict(state_dict=checkpoint["model_state_dict"])

	model.eval()
	with torch.no_grad():
		
		with tqdm(
			range(test_dataset.get_test_size())
			total=test_dataset.get_test_size(),
			desc="Testing"
		) as tqdm_eval:
			
			for _, (
				idx
			) in tqdm_eval:				
				for real_img, gen_img, sim_score in test_dataset.get_test_samples(idx):

					predictions = model.to(DEVICE)(real_img.to(DEVICE), gen_img.to(DEVICE)).detach().data
					print(predictions)


def run_test():
	
	# Relation Net
	model, test_dataset = test_setup_relation_net()
	test_loop(model, test_dataset, checkpoint_file='/home/miruna/.dumps/nag-implementation/repository/logs/relation-net-1/weights/last_model.pth')

run_test()