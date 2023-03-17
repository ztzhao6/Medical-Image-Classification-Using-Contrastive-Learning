import torch
import torch.nn as nn
from .resnet import resnet18, resnet50, resnet101
from .fcnet import FCNet


class TwoResNet(nn.Module):
    def __init__(self, model_name, in_channel, classify_num):
        super(TwoResNet, self).__init__()
        if model_name == 'resnet50':
            self.l_to_ab = resnet50(in_channel=in_channel, classify_num=classify_num)
            self.ab_to_l = resnet50(in_channel=in_channel, classify_num=classify_num)
        elif model_name == 'resnet18':
            self.l_to_ab = resnet18(in_channel=in_channel, classify_num=classify_num)
            self.ab_to_l = resnet18(in_channel=in_channel, classify_num=classify_num)
        else:
            raise NotImplementedError('model {} is not implemented'.format(model_name))

    def forward(self, x, truth_features=False, self_supervised_features=False, classify=False):
        l, ab = torch.split(x, [x.shape[-3] // 2, x.shape[-3] // 2], dim=1)
        feat_l = self.l_to_ab(l, truth_features, self_supervised_features, classify)
        feat_ab = self.ab_to_l(ab, truth_features, self_supervised_features, classify)
        return feat_l, feat_ab


class HandAddResNet(nn.Module):
    def __init__(self, fc_in_dim, model_name, in_channel, classify_num, low_dim=128):
        super(HandAddResNet, self).__init__()
        if model_name == 'resnet50':
            self.radio_net = FCNet(in_dim=fc_in_dim, low_dim=low_dim, classify_num=classify_num)
            self.deep_net = resnet50(in_channel=in_channel, low_dim=low_dim, classify_num=classify_num)
        elif model_name == 'resnet18':
            self.radio_net = FCNet(in_dim=fc_in_dim, low_dim=low_dim, classify_num=classify_num)
            self.deep_net = resnet18(in_channel=in_channel, low_dim=low_dim, classify_num=classify_num)
        elif model_name == 'resnet101':
            self.radio_net = FCNet(in_dim=fc_in_dim, low_dim=low_dim, classify_num=classify_num)
            self.deep_net = resnet101(in_channel=in_channel, low_dim=low_dim, classify_num=classify_num)
        else:
            raise NotImplementedError('model {} is not implemented'.format(model_name))

    def forward(self, image_data, radio_data, truth_features=False, self_supervised_features=False,
                classify=False):
        radio_feature = self.radio_net(radio_data,
                                       truth_features=truth_features,
                                       self_supervised_features=self_supervised_features,
                                       classify=classify)
        image_feature = self.deep_net(image_data,
                                      truth_features=truth_features,
                                      self_supervised_features=self_supervised_features,
                                      classify=classify)
        return image_feature, radio_feature