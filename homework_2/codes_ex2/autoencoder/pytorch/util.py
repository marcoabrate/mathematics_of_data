import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pickle
import numpy as np

from torch.autograd import Variable

def obtain_dataloader(batch_size):
    """
    obtain the data loader
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
    test_set = dset.MNIST(root='./data', train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

class Autoencoder(nn.Module):
    """
    Autoencoder
    """

    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, activation):
        """
        specify an autoencoder
        """
        assert input_dim == output_dim, 'The input and output dimension should be the same'
        
        self.encoder_weight = torch.randn([input_dim, hidden_dim]) * 0.02
        self.decoder_weight = torch.randn([hidden_dim, output_dim]) * 0.02
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if activation.lower() in ['relu']:
            self.activation=lambda x: torch.clamp(x, min = 0.)
            self.dactivation=lambda x: (torch.sign(x) + 1) / 2
        elif activation.lower() in ['tanh']:
            self.activation=lambda x: (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
            self.dactivation=lambda x: 4 / (torch.exp(x) + torch.exp(-x)) ** 2
        elif activation.lower() in ['identity']:
            self.activation=lambda x: x
            self.dactivation=lambda x: torch.ones_like(x)
        elif activation.lower() in ['sigmoid', 'sigd']:
            self.activation=lambda x: 1. / (1. + torch.exp(-x))
            self.dactivation=lambda x: torch.exp(x) / (1 + torch.exp(x)) ** 2
        elif activation.lower() in ['negative']:
            self.activation=lambda x: -x
            self.dactivation=lambda x: -torch.ones_like(x)
        else:
            raise ValueError('unrecognized activation function')

    def train(self, data_batch, step_size):
        """
        training a model

        :param data_batch: of shape [batch_size, input_dim]
        :param step_size: float, step size
        """

        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)                # of shape [batch_size, hidden_dim]
        decode = torch.matmul(encode, self.decoder_weight)  # of shape [batch_size, output_dim]

        error = data_batch - decode
        loss = torch.mean(torch.sum(error ** 2, dim = 1)) / 2
        
        gradW2 = torch.zeros_like(self.decoder_weight)
        gradW1 = torch.zeros_like(self.encoder_weight)
        for i in range(self.batch_size):
            # picking the i-th sample from the batch (xi)
            xi = data_batch[i, :]
            projectioni = torch.matmul(xi, self.encoder_weight)
            encodei = self.activation(projectioni)
            decodei = torch.matmul(encodei, self.decoder_weight)
            errori = xi - decodei
            # computing the gradient with respect to W2
            gradW2i = torch.matmul(encodei.unsqueeze(1), errori.unsqueeze(1).t())
            # adding the gradient of the i-th sample to the final gradient
            # the minus comes from the gradient's formula
            # the division by the dimension of the batch is done here so to avoid overflows
            gradW2 -= (gradW2i / self.batch_size)
            
            # computing the gradient with respect to W1
            # right and left are just the two terms of the gradient's formula
            sgW1right = \
                torch.matmul(xi.unsqueeze(1), self.dactivation(projectioni).unsqueeze(1).t())
            row = torch.matmul(errori, self.decoder_weight.t())
            sgW1left = row.repeat(self.input_dim, 1)
            sgW1i = sgW1left * sgW1right
            gradW1 -= (sgW1i / self.batch_size)
            
        if (torch.isnan(gradW2).sum() != 0 or (gradW2 == float('inf')).sum() != 0):
            raise ValueError('encountered inf or nan in gradientW2')
        
        if (torch.isnan(gradW1).sum() != 0 or (gradW1 == float('inf')).sum() != 0):
            raise ValueError('encountered inf or nan in gradientW1')
        
        # updating the weights
        self.encoder_weight = self.encoder_weight - step_size * gradW1
        self.decoder_weight = self.decoder_weight - step_size * gradW2

        return float(loss)

    def test(self, data_batch):
        """
        test and calculate the reconstruction loss
        """
        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)
        decode = torch.matmul(encode, self.decoder_weight)

        error = decode - data_batch
        loss = torch.mean(torch.sum(error ** 2, dim = 1)) / 2

        return loss

    def compress(self, data_batch):
        """
        compress a data batch
        """

        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)

        return np.array(encode)

    def reconstruct(self, data_batch):
        """
        reconstruct the image
        """
        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)
        decode = torch.matmul(encode, self.decoder_weight)

        return np.array(decode)

    def save_model(self, file2dump):
        """
        save the model
        """
        pickle.dump(
                [np.array(self.encoder_weight), np.array(self.decoder_weight)],
                open(file2dump, 'wb'))

    def load_model(self, file2load):
        """
        load the model
        """
        encoder_weight, decoder_weight = pickle.load(open(file2load, 'rb'))
        self.encoder_weight = torch.FloatTensor(encoder_weight)
        self.decoder_weight = torch.FloatTensor(decoder_weight)
