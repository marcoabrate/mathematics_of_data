import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from util import Autoencoder, obtain_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_num', type=int, default=50,
        help='the number of epochs, default = 50')
    parser.add_argument('--hidden_dim', type=int, default=100,
        help='the dimension of hidden neurons, default = 100')
    parser.add_argument('--activation', type=str, default='relu',
        help='the activation function to use, default = "relu", allowed = ["relu", "tanh", "identity", "sigmoid", "negative"]')
    parser.add_argument('--step_size', type=float, default = 0.01,
        help='the step size, default = 0.01')
    parser.add_argument('--save_file', type=str, default="model_relu.ckpt",
        help='the file to restore the model, default = "model_relu.ckpt"')

    args = parser.parse_args()

    input_dim = 28 * 28
    output_dim = 28 * 28
    batch_size = 100
    epoch_num = args.epoch_num
    hidden_dim = args.hidden_dim
    activation = args.activation
    step_size = args.step_size
    save_file = args.save_file
    
    plotname = 'eps'+str(int(epoch_num))+'step'+(str(step_size-int(step_size)).split('.')[1])[0:5]+activation
    print(plotname)

    train_loader, test_loader = obtain_dataloader(batch_size)

    model = Autoencoder(batch_size, input_dim, hidden_dim, output_dim, activation)

    train_loss_list = []
    test_loss_list = []
    for epoch_idx in range(epoch_num):

        loss_list = []
        for idx, (data_batch, label_batch) in enumerate(train_loader, 0):
            data_batch = data_batch.view(batch_size, -1)
            loss = model.train(data_batch, step_size)
            #if (idx % 100 == 0):
            #    print('iteration %d: LOSS %.5f'%(idx, loss))
            loss_list.append(loss)
        loss_mean = np.mean(loss_list)
        print('Train loss after epoch %d: %1.3e'%(epoch_idx + 1, loss_mean))
        train_loss_list.append(loss_mean)

        loss_list = []
        for idx, (data_batch, label_batch) in enumerate(test_loader, 0):
            data_batch = data_batch.view(batch_size, -1)
            loss = model.test(data_batch)
            loss_list.append(loss)
        loss_mean = np.mean(loss_list)
        print('Test loss after epoch %d: %1.3e'%(epoch_idx + 1, loss_mean))
        test_loss_list.append(loss_mean)

    model.save_model(file2dump = save_file)
    print('Model saved in %s'%save_file)

    plt.plot(np.arange(1, epoch_num + 1), train_loss_list, label = 'train loss', color = 'r')
    plt.plot(np.arange(1, epoch_num + 1), test_loss_list, label = 'test loss', color = 'b')
    plt.legend()
    plt.savefig(plotname+'.pdf')
    plt.show()