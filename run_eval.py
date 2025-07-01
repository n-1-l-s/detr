import torchvision
import torch
import argparse
import dataset

from dataset.doclaynet import build
from models import build_model
from main import get_args_parser


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
args.dataset_file = "doclaynet"
args.device = "cpu"

ds = build("val", args)
model, criterion, postprocessors = build_model(args)

print(len(ds))