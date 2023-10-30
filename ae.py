#!/usr/bin/env python
import torch.nn as nn


def init_weights(m):
    """ initialize weights of fully connected layer
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# autoencoder with hidden units 200, 20, 200
# Hidden1_1
class Hidden1_1(nn.Module):
    def __init__(self, num_inputs):
        super(Hidden1_1, self).__init__()
        self.hidden1_1 = nn.Linear(num_inputs, 200)
        self.hidden1_1.apply(init_weights)

    def forward(self, x):
        x = self.hidden1_1(x)
        return x


# Encoder1
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(200, 20)
        )
        self.encoder1.apply(init_weights)

    def forward(self, x):
        x = self.encoder1(x)
        return x


# Decoder1
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.decoder2 = nn.Linear(20, 200)
        self.decoder2.apply(init_weights)

    def forward(self, x):
        x = self.decoder2(x)
        return x


# Hidden2_1
class Hidden2_1(nn.Module):
    def __init__(self, num_inputs):
        super(Hidden2_1, self).__init__()
        self.hidden2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(200, num_inputs),
            nn.ReLU()
        )
        self.hidden2_1.apply(init_weights)

    def forward(self, x):
        x = self.hidden2_1(x)
        return x


# Hidden1_2
class Hidden1_2(nn.Module):
    def __init__(self, num_inputs):
        super(Hidden1_2, self).__init__()
        self.hidden1_2 = nn.Linear(num_inputs, 200)
        self.hidden1_2.apply(init_weights)

    def forward(self, x):
        x = self.hidden1_2(x)
        return x


# Encoder2
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.encoder2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(200, 20)
        )
        self.encoder2.apply(init_weights)

    def forward(self, x):
        x = self.encoder2(x)
        return x


# Decoder2
class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.decoder2 = nn.Linear(20, 200)
        self.decoder2.apply(init_weights)

    def forward(self, x):
        x = self.decoder2(x)
        return x


# Hidden2_2
class Hidden2_2(nn.Module):
    def __init__(self, num_inputs):
        super(Hidden2_2, self).__init__()
        self.hidden2_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(200, num_inputs),
            nn.ReLU()
        )
        self.hidden2_2.apply(init_weights)

    def forward(self, x):
        x = self.hidden2_2(x)
        return x


# autoencoder1
class autoencoder1(nn.Module):
    def __init__(self, num_inputs):
        super(autoencoder1, self).__init__()
        self.hidden1 = Hidden1_1(num_inputs)
        self.encoder = Encoder1()
        self.decoder = Decoder1()
        self.hidden2 = Hidden2_1(num_inputs)

    def forward(self, x):
        hidden1 = self.hidden1(x)
        code = self.encoder(hidden1)
        hidden2 = self.decoder(code)
        x = self.hidden2(hidden2)
        return code, x, hidden1, hidden2


# autoencoder2
class autoencoder2(nn.Module):
    def __init__(self, num_inputs):
        super(autoencoder2, self).__init__()
        self.hidden1 = Hidden1_2(num_inputs)
        self.encoder = Encoder1()
        self.decoder = Decoder1()
        self.hidden2 = Hidden2_2(num_inputs)

    def forward(self, x):
        hidden1 = self.hidden1(x)
        code = self.encoder(hidden1)
        hidden2 = self.decoder(code)
        x = self.hidden2(hidden2)
        return code, x, hidden1, hidden2
