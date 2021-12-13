from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
import random
import math
import time
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import torch.nn.functional as F

from models.model_func import *
from utils import rand_bbox

def imagenet_denorm(tensor):
    inv_normalize = transforms.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
    )


    return inv_normalize(tensor)

def train_one_epoch(epoch, train_loader, model, optimizer, criterion, save_dir, log_path, cuda_device, beta = 0, cutmix_prob = 0):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(cuda_device), target.cuda(cuda_device)
            data_ori = data.detach()

        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(data.size()[0]).cuda(cuda_device)
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

            #visualize
            if batch_idx < 10:
                save_image(imagenet_denorm(data), save_dir + str(batch_idx) + '_cutmix_ori.png')

            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            if batch_idx < 10:
                save_image(imagenet_denorm(data), save_dir + str(batch_idx) + '_cutmix.png')

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            # compute output
            outputs, resize_data = model(data)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)

        else:
            outputs, resize_data = model(data)
            loss = criterion(outputs, target)

        #visualize
        if batch_idx < 10:
            save_image(imagenet_denorm(data_ori), save_dir + str(batch_idx) + '_ori.png')
            save_image(F.interpolate(imagenet_denorm(data_ori), (224,224)), save_dir + str(batch_idx) + '_bilinear.png')
            save_image(imagenet_denorm(resize_data), save_dir + str(batch_idx) + '_resize.png')

            if resize_data.size(-1) == 224:
                save_image(F.interpolate(imagenet_denorm(data_ori), (224,224)) - imagenet_denorm(resize_data), save_dir + str(batch_idx) + '_diff.png')
            else:
                save_image(F.interpolate(imagenet_denorm(data_ori), (224,224)) - F.interpolate(imagenet_denorm(resize_data), (224,224)), save_dir + str(batch_idx) + '_diff.png')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        if batch_idx % 10 == 0:
            train_log = 'Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'.format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total)
            
            with open(log_path, "a") as f:
                f.write(str(train_log) + "\n")
            
            print(train_log)


def test(epoch, test_loader, model, optimizer, criterion, save_dir, log_path, cuda_device):
     
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(cuda_device), target.cuda(cuda_device)

        outputs, resize_data = model(data)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

        #visualization  
        if batch_idx < 10:
            save_image(imagenet_denorm(data), save_dir + str(batch_idx) + '_ori.png')
            save_image(imagenet_denorm(resize_data), save_dir + str(batch_idx) + '_resize.png')
            save_image(F.interpolate(imagenet_denorm(data), (224,224)), save_dir + str(batch_idx) + '_bilinear.png')

            if resize_data.size(-1) == 224:
                save_image(F.interpolate(imagenet_denorm(data), (224,224)) - imagenet_denorm(resize_data), save_dir + str(batch_idx) + '_diff.png')
            else:
                save_image(F.interpolate(imagenet_denorm(data), (224,224)) - F.interpolate(imagenet_denorm(resize_data), (224,224)), save_dir + str(batch_idx) + '_diff.png')



    test_log = '# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'.format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total)

    with open(log_path, "a") as f:
        f.write(str(epoch) + '/' +str(test_log) + "\n")

    return test_log, 100. * correct / total


def save_checkpoint(directory, state, filename='latest.tar.gz'):

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_filename = os.path.join(directory, filename)
    torch.save(state, model_filename)
    print("=> saving checkpoint")

def load_checkpoint(directory, filename='latest.tar.gz'):

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        return None


def run_training(data_dir,
                model_path,
                model,
                epochs,
                img_resize,
                learning_rate,
                batch_size,
                beta,
                cutmix_prob,
                cuda_device,
                workers):

    # Data Augmentation
    transform_train = transforms.Compose([
        #transforms.Resize((img_resize, img_resize)),
        transforms.RandomCrop(32, padding=4),               # Random Position Crop
        transforms.RandomHorizontalFlip(),                  # right and left flip
        transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                             std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),                              # change [0,255] Int value to [0,1] Float value
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467), # RGB Normalize MEAN
                             std=(0.2471, 0.2436, 0.2616))  # RGB Normalize Standard Deviation
    ])

    # automatically download
    train_dataset = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=data_dir,
                                    train=False,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,            # at Training Procedure, Data Shuffle = True
                                               num_workers=4)           # CPU loader number

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False,            # at Test Procedure, Data Shuffle = False
                                              num_workers=4,
                                              drop_last = False)            # CPU loader number

    optimizer = optim.SGD(model.parameters(), learning_rate,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if torch.cuda.device_count() > 0:
        print("USE", torch.cuda.device_count(), "GPUs!")
        #model = nn.DataParallel(model).cuda()
        model = model.cuda(cuda_device)
        cudnn.benchmark = True
    else:
        print("USE ONLY CPU!")
        
    start_epoch = 0
    
    checkpoint = load_checkpoint(model_path)

    if not checkpoint:
        pass
    else:
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_time = time.time()

    train_log_path = model_path + '/train_log.txt'
    test_log_path = model_path + '/test_log.txt'
    test_log_best_path = model_path + '/test_best_log.txt'
    with open(train_log_path, "w"):
        pass
    with open(test_log_path, "w"):
        pass
    with open(test_log_best_path, "w"):
        pass



    test_acc_best = -1
    for epoch in range(start_epoch, epochs):
        train_save_dir = model_path + '/visual/' + str(epoch) + '/' + 'train/'
        test_save_dir = model_path + '/visual/' + str(epoch) + '/' + 'test/'
        Path(train_save_dir).mkdir(exist_ok=True, parents=True)
        Path(test_save_dir).mkdir(exist_ok=True, parents=True)
    
    
        train_one_epoch(epoch, train_loader, model, optimizer, criterion, train_save_dir, train_log_path, cuda_device, beta = beta, cutmix_prob = cutmix_prob)
        test_log, test_acc = test(epoch, test_loader, model, optimizer, criterion, test_save_dir, test_log_path, cuda_device)

        save_checkpoint(model_path, {
            'epoch': epoch,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        filename = 'latest.tar.gz')

        if test_acc_best < test_acc:
            test_acc_best = test_acc
            save_checkpoint(model_path, {
                'epoch': epoch,
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            filename = 'best.tar.gz')

            with open(test_log_best_path, "a") as f:
                f.write(str(epoch) + '/' + str(test_log) + "\n")


        """decrease the learning rate at 100 and 150 epoch"""
        lr = learning_rate
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    
    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training cifar10 classification')
    parser.add_argument('--data_dir', default="./cifar10/")
    parser.add_argument('--save_dir', default="./experiments")
    parser.add_argument('--net_type', default="vgg16", type=str)
    parser.add_argument('--version', default="cutmix", type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default="128", type=int)
    parser.add_argument('--learning_rate', default="0.01", type=float)
    parser.add_argument('--img_resize', default="32", type=int)
    parser.add_argument('--resizer_img_resize', default="224", type=int)
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=2021, type=int)

    #cutmix 
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--cutmix_prob', default=0.5, type=float)

    args = parser.parse_args()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    all_model_base_lists = {'vgg16': VGG16_cifar(pretrained = True, num_classes = len(classes))}
    all_model_resizer_lists = {'vgg16_resizer': VGG16_resizer_cifar(pretrained = True, num_classes = len(classes), out_resize = args.resizer_img_resize)}
    all_model_resizer_att_lists = {'vgg16_resizer_att': VGG16_resizer_att_cifar(pretrained = True, num_classes = len(classes), out_resize = args.resizer_img_resize)}

    all_model_lists = {**all_model_base_lists, **all_model_resizer_lists, **all_model_resizer_att_lists}
    print(all_model_lists.keys())

    all_model_lists_selected = [all_model_lists[args.net_type]]
    print(all_model_lists_selected)
    del all_model_lists

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cuda_device = args.cuda_device
    torch.cuda.set_device(cuda_device)
    print(f"using device: {cuda_device}")


    for model_selected in all_model_lists_selected:
        # create save model path
        model_path = 'experiments' + '/' + args.net_type  + '/' + args.version
        Path(model_path).mkdir(exist_ok=True, parents=True)
        print(f"training {args.net_type}")
        run_training(data_dir = args.data_dir,
                    model_path=model_path,
                    model=model_selected,
                    epochs=args.epochs,
                    img_resize = args.img_resize,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    beta=args.beta,
                    cutmix_prob = args.cutmix_prob,
                    cuda_device=cuda_device,
                    workers=args.workers)







