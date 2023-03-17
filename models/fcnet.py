import torch.nn as nn
from .normalize_layer import Normalize


class FCNet(nn.Module):
    def __init__(self, in_dim, mid_dim=256, low_dim=128, classify_num=2):
        super(FCNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(True),
            nn.Dropout()
        )

        # output
        self.l2norm = Normalize(2)
        self.self_supervised_layers = nn.Sequential(
            # nn.Linear(mid_dim, mid_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(mid_dim, low_dim),
        )
        self.classification = nn.Linear(mid_dim, classify_num)

    def forward(self, x, truth_features=False, self_supervised_features=False, classify=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # output
        if truth_features:
            x = self.l2norm(x)
            return x
        elif self_supervised_features:
            x = self.self_supervised_layers(x)
            x = self.l2norm(x)
            return x
        elif classify:
            x = self.classification(x)
            return x