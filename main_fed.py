#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
import socket
import time

from Functions import *
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

# matplotlib.use('Agg')  # XY: Using Agg(for no GPU condition) mode, just save image, cannot plot it
if __name__ == '__main__':
    # socket info
    PORT = 5050
    # SERVER = "10.17.198.243"
    SERVER = "192.168.0.6"
    ADDR = (SERVER, PORT)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect_ex(ADDR)
    client.setblocking(0)

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # print(dataset_train[0])
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            # print('iid: ', dict_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
            # print('non_iid: ', dict_users, len(dict_users), len(dict_users[0]))
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    # XY: Specific Training and Test dataset and shuffled Training sets idx
    training_list = list(dataset_train.targets)
    test_list = list(dataset_test.targets)

    if args.target_set != [0, 1, 2, 3, 4]:
        targets = []
        for t in args.target_set:
            targets.append(int(t))
    else:
        targets = args.target_set

    new_dataset_train = []
    new_dataset_test = []
    # targets = np.random.choice(np.arange(0, 10, dtype=int), int(10 / args.N), replace=False)  # random choose N target
    for target in training_list:
        if target in targets:
            new_dataset_train.append(dataset_train[training_list.index(target)])
    for item in test_list:
        if item in targets:
            new_dataset_test.append(dataset_test[test_list.index(item)])
    # print('the length of new training data set: ', len(new_dataset_train), len(new_dataset_test), '\n')
    idx_len = len(new_dataset_train) - len(new_dataset_train) % 10
    idx = list(np.arange(len(new_dataset_train)))
    random.shuffle(idx)
    # print(len(new_data), idx_len)
    # i_d = np.random.choice(idx, idx_len, replace=False)
    i_d = np.arange(0, idx_len, dtype=int)
    # print(i_d, '\n', len(idx), len(i_d))

    img_size = dataset_train[0][0].shape
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':  # Multi-Layer Perceptron 多层感应器
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # print(net_glob, '\n')
    # XY: Turn to Training mode, activate dropout layer and the same layers
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    # print('w_glob keys: ', w_glob.keys(), '\n')

    # training  XY: Set all parameters to 0 and None for next training process
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # if args.all_clients:
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        print(f'[START] Start {iter} iteration local update ...')
        loss_locals = []
        # if not args.all_clients:  # XY: at this time, choose 10 users at one time, so it's not all_clients
        #     w_locals = []
        local = LocalUpdate(args=args, dataset=new_dataset_train, idxs=i_d)  # XY: Each Local update has 10 local iterations
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        # print('w: ', w, '\n')
        # print(f'{iter} loss: ', loss)
        print('Round {}, Average loss {:.8f}'.format(iter, loss), '\n')
        # loss_train.append(loss_avg)
        loss_train.append(loss)  # Print the loss trend changing along with iteration times

        # XY: Test the model  Original test function cannot be running because Mac don't have GPU mode for cuda (torch.cuda.is_available() = False)
        # net_glob.load_state_dict(w)
        # net_glob.eval()
        # acc_test, loss_test = test_img(net_glob, new_dataset_test, args)
        # print(iter, acc_test, loss_test, '\n')

        # XY: Socket process
        send_msg(client, w)
        print(f'[WAITING] number {iter} iterations finished and waiting for server {SERVER} response ...', '\n')
        time.sleep(10)
        if True:
            back_msg = recv_msg(client)
            back_msg = pickle.loads(back_msg)
            print(f'[RECEIVED] Received new weights from server {SERVER} and load weights then start next iteration ... ', '\n')
            net_glob.load_state_dict(back_msg)
        # if args.all_clients:
        #     w_locals[idx] = copy.deepcopy(w)
        # else:
        #     w_locals.append(copy.deepcopy(w))
        # loss_locals.append(copy.deepcopy(loss))
        # print('w_locals: ', w_locals, len(w_locals), '\n')
        # print('loss_locals: ', loss_locals)
        # update global weights
        # w_glob = FedAvg(w_locals) # XY: Server doing this process: Get weights from several clients and average them

        # copy weight to net_glob  XY: unpickle the message from server and load it to local model
        # net_glob.load_state_dict(w_glob)
        # net_glob.load_state_dict(w)

        # print loss
        # loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.5f}'.format(iter, loss_avg))
        # print('Round {:3d}, Average loss {:.5f}'.format(iter, loss))
        # loss_train.append(loss_avg)
        # loss_train.append(loss)

    # Plot loss curve  XY: just save the raster image without plot it
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, round(time.time()), 5))
    plt.show()
    print(f'[END] the total loss_train is loss_train {loss_train}', '\n')

    # testing  XY: Not work because torch.cuda()
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))
