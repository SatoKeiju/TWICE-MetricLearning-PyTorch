import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import matplotlib.pyplot as plt

from datasets import *
from losses import *
from models import *
from train import *
from test import *
from parameters import args


def main():
    train_dic = make_datapath_dic('train')
    test_dic = make_datapath_dic('test')

    transform = ImageTransform(64)

    train_dataset = TripletDataset(train_dic, transform=transform, phase='train')
    test_dataset = TripletDataset(test_dic, transform=transform, phase='test')

    batch_size = 12
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TripletNet().to(device)
    criterion = TripletLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    summary(model, (3, 64, 64))
    torch.autograd.set_detect_anomaly(True)

    x_epoch_data = []
    y_train_loss_data = []
    y_test_loss_data = []
    y_test_accuracy_data = []

    for epoch in range(1, args.epochs+1):
        train_loss_per_epoch = train(
            args, model, train_dataloader, criterion, optimizer, epoch
        )
        # test_loss_per_epoch, test_accuracy_per_epoch = test(
        #     args, model, test_dataloader, criterion
        # )

        x_epoch_data.append(epoch)
        # y_train_loss_data.append(train_loss_per_epoch)
        # y_test_loss_data.append(test_loss_per_epoch)
        # y_test_accuracy_data.append(test_accuracy_per_epoch)

    # plt.plot(x_epoch_data, y_train_loss_data, color='blue', label='train_loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(loc='upper right')
    # plt.title('loss')
    # plt.show()
    #
    # plt.plot(x_epoch_data, y_test_loss_data, color='red', label='test_loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(loc='upper right')
    # plt.title('loss')
    # plt.show()
    #
    # plt.plot(x_epoch_data, y_test_accuracy_data, label='test_accuracy')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend(loc='lower right')
    # plt.show()

    if args.save_model:
        torch.save(model.state_dict(), 'TripletNetwork.pt')
        print('Saved model as TripletNetwork.pt')


if __name__ == '__main__':
    main()
