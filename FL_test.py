from torchvision import datasets, transforms
import numpy as np
from sympy import Matrix
import torch
from utils.options import args_parser
import PIL
import random


args = args_parser()
# print(parser.defaults)

# args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
# print(args.device)
if args.target_set != [0, 1, 2, 3, 4]:
    targets = []
    for t in args.target_set:
        targets.append(int(t))
else:
    targets = args.target_set
print(targets)
# m = torch.nn.Dropout(p=0.2)
# input = torch.randn(20, 16)
# output = m(input)
# print(output)
# trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
# dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
# # # print(dataset_train._load_data())
# # # print(dataset_train, type(dataset_train), len(dataset_train))
# # # print(dataset_train[0], type(dataset_train[0]))
# t_list = list(dataset_test.targets)
# new_data = []
# # new_target = [1, 3]
# for target in t_list:
#     if target in targets:
#         new_data.append(dataset_test[t_list.index(target)])
# print(len(new_data))
# idx_len = len(new_data) - len(new_data) % 10
# idx = list(np.arange(len(new_data)))
# random.shuffle(idx)
# print(len(new_data), idx_len)
# i_d = np.random.choice(idx, idx_len, replace=False)
# print(i_d, '\n', len(idx), len(i_d))
# print(dataset_train[0][0], dataset_train[0][1])
# new_data = []
# for item in dataset_train:
#     if item[1] == 1:
#         new_data.append(item)
# print(new_data, len(new_data))
# print(dataset_train.data[0])
# idxs = np.arange(60000)
# labels = dataset_train.train_labels.numpy()
# idxs_labels = np.vstack((idxs, labels))
# print(idxs_labels)
# idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
# idxs = idxs_labels[0, :]
# print(idxs_labels)
# idx = (dataset_train.targets == 2) | (dataset_train.targets == 4) | (dataset_train.targets == 6)
# idx = dataset_train.targets == 5
# new_data, new_label = dataset_train.data[idx], dataset_train.targets[idx]
# print(new_data, new_label, len(new_data), len(new_label))
# a = PIL.Image.fromarray(new_data)
# print(trans_mnist(a))
# dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
# for i in range(2):
    # print(dataset_train[i])
    # for j in range(len(dataset_train[i])):
        # print(dataset_train[i][j], '\n', type(dataset_train[i][j]))
# print(len(dataset_train), len(dataset_test))

# idxs = np.arange(0, 10)
# labels = np.arange(10, 20)
# num_shards = 20
# idx_shard = [i for i in range(num_shards)]
# # a = np.concatenate((idxs, labels), axis=0)
# rand_set = set(np.random.choice(idx_shard, 2, replace=False))
# print(rand_set)
# idx_shard = list(set(idx_shard) - rand_set)
# print(idx_shard)
# idxs_labels = np.vstack((idxs, labels))
# print(idxs_labels[0, :])
# idxs_labels = idxs_labels[1, :]
# print(idxs_labels)
# idxs_labels = idxs_labels.argsort()
# print(idxs_labels)
# idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
# idxs = idxs_labels[0, :]
# print(idxs_labels)
# print(idxs)


