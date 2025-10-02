import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# model
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes, superfamily):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=5, stride=1,padding=2)
        #self.conv_layer2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=5, stride=1,padding='same')
        self.max_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        if superfamily == 'Enolase':
            self.fc1 = nn.Linear(64 * 14 * 17 * 15, 128)
            self.fc2 = nn.Linear(128, num_classes)
        elif superfamily == 'Serprot':
            self.fc1 = nn.Linear(64 * 17 * 19 * 21, 128)
            self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.7)


    # Progresses data across layers
    def forward(self, x):
        out = F.relu(self.conv_layer1(x))
        #out = F.relu(self.conv_layer2(out))
        out = self.max_pool1(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out