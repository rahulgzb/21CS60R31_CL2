from multiprocessing.dummy import Pool
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M

class ModelCNN(nn.Module):
    def __init__(self, n_input_channels=3, n_conv_output_channels=16, k=3, s=1, pad=1, p = 0.5):
        """Init method for initializaing the equiCNN model"""
        super(ModelCNN, self).__init__()
        # 1. Convolutional layers
        # Single image is in shape: 3x96x96 (CxHxW, H==W), RGB images
        self.conv0 = P4MConvZ2(in_channels = 3, out_channels = 8, kernel_size = k, stride = s, padding = pad, bias=False)
        self.bn0 = nn.BatchNorm3d(8)
        self.conv1 =  P4MConvP4M(in_channels = 8, out_channels = 16, kernel_size = k, stride = s, padding = pad)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 =  P4MConvP4M(in_channels =16, out_channels = 32, kernel_size = k, stride = s, padding = pad)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 =  P4MConvP4M(in_channels = 32, out_channels = 64, kernel_size = k, stride = 2, padding = pad)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 =  P4MConvP4M(in_channels = 64, out_channels = 128, kernel_size = k, stride = 2, padding = pad)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(kernel_size = k - 1)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p = p)
        
        # 2. FC layers to final output
        self.fc1 = nn.Linear(in_features = 4096, out_features =32*n_conv_output_channels)
        self.fc_bn1 = nn.BatchNorm1d(32*n_conv_output_channels)
        self.fc2 = nn.Linear(in_features = 32*n_conv_output_channels, out_features = 16*n_conv_output_channels)
        self.fc_bn2 = nn.BatchNorm1d(16*n_conv_output_channels)
        self.fc3 = nn.Linear(in_features = 16*n_conv_output_channels, out_features = 8*n_conv_output_channels)
        self.fc_bn3 = nn.BatchNorm1d(8*n_conv_output_channels)
        self.fc4 = nn.Linear(in_features = 8*n_conv_output_channels, out_features = 1)

    def forward(self, x):
        # Convolution Layers, followed by Batch Normalizations, Maxpool, and ReLU
        x = self.conv0(x)                      # batch_size x 96 x 96 x 16
        #print(x.size())
        x=self.bn0(x)
        x = self.conv1(x)                      # batch_size x 96 x 96 x 16
        #print(x.size())
        x=self.bn1(x)
        x=self.relu(x)                        # batch_size x 48 x 48 x 16
        #print(x.size())
        x=self.conv2(x)
        #print(x.size())
        x = self.bn2(x)                      # batch_size x 48 x 48 x 16
        #print(x.size())
        x = self.relu(x)                        # batch_size x 24 x 24 x 32
        #print(x.size())
        x = self.bn3(self.conv3(x))                      # batch_size x 24 x 24 x 64
        x = self.relu(x)                        # batch_size x 12 x 12 x 64
        x = self.bn4(self.conv4(x))                      # batch_size x 12 x 12 x 128
        x = self.relu(x)
        #print(x.size())                        # batch_size x  6 x  6 x 128
        outs = x.size()
        #print(outs)
        x = x.view(outs[0], outs[1]*outs[2], outs[3], outs[4])
        #print(x.size())
        out = F.avg_pool2d(x, 12)
        #print(x.size())
        x = out.view(out.size(0), -1)
        #print(x.size())
        # Flatten the output for each image
        x = x.reshape(-1, self.num_flat_features(x))        # batch_size x 6*6*128
        #print(x.size())
        # Apply 4 FC Layers
        x = self.fc1(x)
        #print(x)
        x = self.fc_bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.fc_bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def eqcnn():
    return ModelCNN ()

def test():
    net = eqcnn()
    print(net)
    y = net(Variable(torch.randn(21,3,96,96)))
    print(y.size())

#test()