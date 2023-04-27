import torch.nn as nn

'''4 layers MLP'''
class ConditionalTransform(nn.Module):
    '''
    A 4 layer NN with ReLU as activation function
    '''

    def __init__(self, Ni, No, Nh=64):
        super(ConditionalTransform, self).__init__()

        layers = []
        self.fc_first = nn.Linear(Ni, Nh)
        layers.append(nn.ReLU())
        layers.append(nn.Linear(Nh, Nh))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(Nh, Nh))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(Nh, Nh))
        self.relu_last = nn.ReLU()
        self.fc_last = nn.Linear(Nh, No)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.fc_first(x)
        x0 = x.clone()
        for layer in self.layers:
            x = layer(x)
        x = self.relu_last(x0+x)
        return self.fc_last(x)

