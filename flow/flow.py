import torch
from torch import nn
import numpy as np

from flow.mobiusflow import get_mobius, MobiusFlow
from flow.affineflow import get_affine

# create flow from config
def get_flow(config): 
    return Flow(config)


_permute_prop = torch.Tensor(
    [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 0, 1]]
).type(torch.long)


class Flow(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(Flow, self).__init__()
        self.config = config
        self.layers = config.layers
        self.condition = config.condition
        self._permute = _permute_prop # permute strategy

        if self.condition:
            self.feature_dim = 32 if config.feature_dim == None else config.feature_dim
            if config.embedding:
                self.feature_dim += config.embedding_dim
        else:
            self.feature_dim = 0

        layers = []
        if config.last_affine:
            layers.append(get_affine(config, self.feature_dim,
                                     first_layer_condition=True))

        for i in range(self.layers):
            tmp = get_mobius(config, self.feature_dim)
            if tmp != None:
                layers.append(tmp)
        
            tmp = get_affine(config, self.feature_dim)
            if tmp != None and (i != self.layers-1 or config.first_affine):
                layers.append(tmp)
            
        print("total layers of flow: ", len(layers))
        self.layers = nn.ModuleList(layers)

    def forward(self, rotation, feature=None, inverse=False, draw=False):
        if inverse:
            return self.inverse(rotation, feature, draw)
        permute = self._permute.to(rotation.device)

        ldjs = 0
        exchange_count = 0

        if not self.condition:
            feature = None

        for i in range(len(self.layers)):
            rotation, ldj = self.layers[i](
                rotation, permute[exchange_count % 6], feature
            )
            ldjs += ldj
            if (isinstance(self.layers[i], MobiusFlow)) or self.config.frequent_permute:
                exchange_count += 1

        return rotation, ldjs

    def inverse(self, rotation, feature=None, draw=False):
        permute = self._permute.to(rotation.device)

        ldjs = 0
        exchange_count = len(
            self.layers) if self.config.frequent_permute else self.config.layers

        if not self.condition:
            feature = None

        for i in range(len(self.layers))[::-1]:
            if (isinstance(self.layers[i], MobiusFlow)) or self.config.frequent_permute:
                exchange_count -= 1
            rotation, ldj = self.layers[i].inverse(
                rotation, permute[exchange_count % 6], feature
            )
            ldjs += ldj

        return rotation, ldjs
