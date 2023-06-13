from __future__ import print_function

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.dirichlet import Dirichlet
from torch.utils.data import RandomSampler
from PIL import Image
from IPython.display import HTML
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--seed', default=20200608, type=int, help='random seed')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[100, 150], 
	help="decay learning rate by decay_rate at these epochs")
parser.add_argument("--ablation", nargs="+", type=float, default=[1, 1, 1])
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epsilon', default=0, type=float, help='epsilon')
parser.add_argument('--model', default="ResNet18", type=str, help='model type (default: ResNet18)')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--num-classes', default=10, type=int, help='the number of classes (default: 10)')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--precision', default=100000, type=float)
parser.add_argument('--sigma', default=0.5, type=float, help='sigma')
parser.add_argument('--alpha', default=0.1, type=float, help='alpha')

args = parser.parse_args()
args.num_classes = 100 if args.dataset == 'CIFAR100' else 10

def tpr95(soft_IN, soft_OOD, precision):
	#calculate the falsepositive error when tpr is 95%

	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision # precision:200000

	total = 0.0
	fpr = 0.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		if tpr <= 0.9505 and tpr >= 0.9495:
			fpr += error2
			total += 1
	if total == 0:
		print('corner case')
		fprBase = 1
	else:
		fprBase = fpr/total
	return fprBase

def auroc(soft_IN, soft_OOD, precision):
	#calculate the AUROC
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		aurocBase += (-fpr+fprTemp)*tpr
		fprTemp = fpr
	aurocBase += fpr * tpr
	#improve
	return aurocBase

def auprIn(soft_IN, soft_OOD, precision):
	#calculate the AUPR

	precisionVec = []
	recallVec = []
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(start, end, gap):
		tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
		if tp + fp == 0: continue
		precision = tp / (tp + fp)
		recall = tp
		precisionVec.append(precision)
		recallVec.append(recall)
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def auprOut(soft_IN, soft_OOD, precision):
	#calculate the AUPR
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(end, start, -gap):
		fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
		if tp + fp == 0: break
		precision = tp / (tp + fp)
		recall = tp
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def detection(soft_IN, soft_OOD, precision):
	#calculate the minimum detection error
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	errorBase = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

def get_softmax(net, dataloader):
	net.eval()
	res = np.array([])
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs = net(inputs)
			softmax_vals, predicted = torch.max(F.softmax(outputs.data, dim=1), dim=1)
			res = np.append(res, softmax_vals.cpu().numpy())
	return res

class SALoss(nn.Module):
	"""docstring for SALoss"""
	def __init__(self, alpha = 0.1, sigma = 0.1):
		super(SALoss, self).__init__()
		self.CELoss = nn.CrossEntropyLoss()
		self.NLLLoss = nn.NLLLoss()
		self.Softmax = nn.Softmax(dim=1)
		self.LogSoftmax = nn.LogSoftmax(dim=1)
		self.alpha = alpha
		self.sigma = sigma
	def forward(self, outputs_ID, labels, outputs):
		loss_CE = self.CELoss(outputs_ID, labels)
		# loss_NLL = -self.alpha*self.NLLLoss(torch.log(self.Softmax(outputs_ID) + 1), labels)
		loss_LM = -self.alpha * self.LogSoftmax(outputs).mean()
		loss_CP = self.sigma * (self.Softmax(outputs) * self.LogSoftmax(outputs)).sum(dim=1).mean()
		loss = loss_CE + loss_LM + loss_CP
		return loss

def build_model(model, num_classes=10):
	if model == 'ResNet18':
		net = ResNet18(num_classes=num_classes)
	elif model == 'VGG19':
		net = VGG('VGG19',num_classes=num_classes)
	elif model == 'MobileNetV2':
		net = MobileNetV2(num_classes=num_classes)
	elif model == 'EfficientNet':
		net = EfficientNetB0(num_classes=num_classes)
	net.cuda()
	cudnn.benchmark = True
	return net

def adjust_learning_rate(optimizer, epoch):
	if args.adlr:
		if epoch in args.decay_epochs:
			for param_group in optimizer.param_groups:
				new_lr = param_group['lr'] * 0.1
				param_group['lr'] = new_lr

def train(trainloader_ID, trainloader_OOD, net, criterion, optimizer):
	net.train()

	for idx, (data_ID, data_OOD) in enumerate(zip(trainloader_ID, trainloader_OOD)):
		inputs_ID, targets_ID = data_ID
		inputs_OOD == data_OOD

		num_ID = inputs_ID.size(0)
		num_OOD = inputs_OOD.size(0)

		inputs = torch.cat((inputs_ID, inputs_OOD), dim=0)
		inputs = inputs.cuda()
		targets_ID = targets_ID.cuda()
		outputs = net(inputs)
		outputs_ID = outputs[0:num_ID]
		outputs_OOD = outputs[num_ID:num_ID + num_OOD]

		loss = criterion(outputs_ID, targets_ID, outputs)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def test_ACC(testloader, net, criterion):
	net.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for idx, (inputs_ID, targets) in enumerate(testloader):
			inputs_ID, targets = inputs_ID.cuda(), targets.cuda()
			outputs_ID = net(inputs_ID)
			loss = criterion(outputs_ID, targets)
			_, predicted = torch.max(outputs_ID.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	test_acc = 100.*correct/total
	return test_acc

def test_OOD(testloader_ID, testloader_OOD, net, criterion):
	soft_ID = get_softmax(net, testloader_ID)
	soft_OOD = get_softmax(net, testloader_OOD)
	precision=args.precision
	OOD_detection = np.array([0.0,0.0,0.0,0.0,0.0])
	OOD_detection[0] = auroc(soft_ID, soft_OOD, precision)*100
	OOD_detection[1] = auprIn(soft_ID, soft_OOD, precision)*100
	OOD_detection[2] = auprOut(soft_ID, soft_OOD, precision)*100
	OOD_detection[3] = tpr95(soft_ID, soft_OOD, precision)*100
	OOD_detection[4] = detection(soft_ID, soft_OOD, precision)*100
	return OOD_detection

def test(testloader_ID, testloader_OOD, net):
	OOD_detection = test_OOD(testloader_ID, testloader_OOD, net, criterion)
	ACC = test_ACC(testloader, net, criterion)
	return ACC, OOD_detection

def get_transform(dataset):
	if dataset == 'CIFAR10':
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2470, 0.2435, 0.2616)
		transform_train = transforms.Compose([
			transforms.Resize(32),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
		transform_test = transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	elif dataset == 'CIFAR100':
		mean = (0.5071, 0.4865, 0.4409)
		std = (0.2673, 0.2564, 0.2762)
		transform_train = transforms.Compose([
			transforms.Resize(32),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
		transform_test = transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	return transform_train, transform_test

def load_CIFAR10(batch_size, transform_train, transform_test):
	trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
	testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
	return trainloader, testloader
def load_CIFAR100(batch_size, transform_train, transform_test):
	trainset = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
	testset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
	return trainloader, testloader

def load_OOD(dataset, batch_size, transform):
	if dataset == 'TinyImageNet(r)':
		dataset = datasets.ImageFolder('./data/Imagenet',transform=transform)
	elif dataset == 'LSUN':
		dataset = datasets.ImageFolder('./data/LSUN',transform=transform)
	elif dataset == 'iSUN':
		dataset = datasets.ImageFolder('./data/iSUN',transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
	return dataloader

def load_data():
	transform_train, transform_test = get_transform(args.dataset)
	if args.dataset == "CIFAR10":
		trainloader_ID, testloader_ID = load_CIFAR10(args.batch_size, transform_train, transform_test)
	elif args.dataset == "CIFAR100"::
		trainloader_ID, testloader_ID = load_CIFAR100(args.batch_size, transform_train, transform_test)
	trainloader_OOD = load_OOD(args.dataset, args.batch_size, transform_train)
	testloader_OOD = load_OOD(args.dataset, args.batch_size, transform_train)
	trainloader_ID, testloader_ID, trainloader_OOD, testloader_OOD
	return trainloader_ID, testloader_ID, trainloader_OOD, testloader_OOD

def main():

	trainloader_ID, testloader_ID, trainloader_OOD, testloader_OOD = load_data()

	criterion = SALoss(alpha = args.alpha, sigma = args.sigma)

	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

	for epoch in tqdm(range(0, args.epoch)):
		train(trainloader_ID, trainloader_OOD, net, criterion, optimizer)
		adjust_learning_rate(optimizer, epoch)

	ACC, OOD_detection = test(testloader_ID, testloader_OOD, net)