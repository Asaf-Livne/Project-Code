## Create an audio discriminator model with a convolutional neural network
## The model is used to classify audio samples as real or fake
## 7 layers of 1D convolutional layers with batch normalization and leaky ReLU activation functions
## 1 fully connected layer with sigmoid activation function
## The model is trained using the binary cross-entropy loss function
## Use nn.sequential to define the model

from imports import *

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            Layer(128, 32, 10, 1, 1),
            Layer(32, 128, 21, 1, 1),
            Layer(128, 512, 21, 1, 1),
            Layer(512, 1024, 21, 1, 1),
            Layer(1024, 1024, 21, 1, 1),
            Layer(1024, 1024, 5, 1, 1),
            Layer(1024, 1, 3, 1, 1),
            nn.Tanh()
        )
    
    

    def forward(self, x):
        return self.seq(x)
    

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.layer(x)

