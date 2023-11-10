import torch
from torch import nn
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet101
from torchvision.models.resnet import ResNet101_Weights

'''backbone network to extract feature from image'''

'''address for pretrained matrix fisher checkpoint'''
fisher_network = dict(
    modelnet='Matrixfisher/storage/modelnet/saved_weights/state_dict_49.pkl',
    pascal3d='Matrixfisher/storage/pascal/saved_weights/state_dict_119.pkl',
)


def get_network(config):
    assert config.condition

    if config.pretrain_fisher:
        base = resnet101(pretrained=True, progress=True)
        net = ResnetHead(base, config.category_num + (1 if config.dataset == 'pascal3d' else 0),
                         config.embedding_dim, 512, 9)
        path = fisher_network[config.dataset]
        with open(path, 'rb') as f:
            state_dict = torch.load(f)
        net.load_state_dict(state_dict)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
    elif 'resnet' in config.network:
        net = get_ResNet(config)
    else:
        raise NotImplementedError()
    net.cuda()
    return net


def get_ResNet(config, pretrain=True):
    model = models.__dict__[config.network]()
    
    # if use pretrained matrix fisher model
    if pretrain:
        print(f"Loading pretrained model...")
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)

    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(512*4, 128),
        nn.BatchNorm1d(128),
        nn.ReLU6(inplace=True),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU6(inplace=True),
        nn.Linear(64, config.feature_dim),
    )

    for m in model.fc:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    if not pretrain:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    return model


class TestConfig:
    def __init__(self):
        self.num_classes = 9
        self.network = 'resnet18'


if __name__ == '__main__':
    config = TestConfig()
    net = get_network(config)
    img = torch.randn(2, 3, 227, 227).cuda()
    output = net(img)
    print(output.shape)
