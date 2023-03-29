import torch
from torch import nn


class Conv1D_ReLU_BN(nn.Module):
    def __init__(self, c_in, c_out, ks, stride, padding, dilation):
        super(Conv1D_ReLU_BN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv1d(c_in, c_out, ks, stride, padding, dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(c_out),
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Res2_Conv1D(nn.Module):
    def __init__(self, c, scale, ks, stride, padding, dilation):
        super(Res2_Conv1D, self).__init__()
        assert c % scale == 0
        self.c = c
        self.scale = scale
        self.width = c // scale

        self.convs = []
        self.bns = []

        for i in range(scale - 1):
            self.convs.append(nn.Conv1d(self.width, self.width, ks, stride, padding, dilation))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        """
        param x: (B x c x d)
        """

        xs = torch.split(x, self.width, dim=1)  # channel-wise split
        ys = []

        for i in range(self.scale):
            if i == 0:
                x_ = xs[i]
                y_ = x_
            elif i == 1:
                x_ = xs[i]
                y_ = self.bns[i - 1](self.convs[i - 1](x_))
            else:
                x_ = xs[i] + ys[i - 1]
                y_ = self.bns[i - 1](self.convs[i - 1](x_))
            ys.append(y_)

        y = torch.cat(ys, dim=1)  # channel-wise concat
        return y


class Res2_Conv1D_ReLU_BN(nn.Module):
    def __init__(self, channel, scale, ks, stride, padding, dilation):
        super(Res2_Conv1D_ReLU_BN, self).__init__()

        self.network = nn.Sequential(
            Res2_Conv1D(channel, scale, ks, stride, padding, dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channel),
        )

    def forward(self, x):
        y = self.network(x)
        return y


class SE_Block(nn.Module):
    def __init__(self, c_in, c_mid):
        super(SE_Block, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(c_in, c_mid),
            nn.ReLU(inplace=True),
            nn.Linear(c_mid, c_in),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.network(x.mean(dim=-1))
        y = x * s.unsqueeze(-1)
        return y


class SE_Res2_Block(nn.Module):
    def __init__(self, channel, scale, ks, stride, padding, dilation):
        super(SE_Res2_Block, self).__init__()
        self.network = nn.Sequential(
            Conv1D_ReLU_BN(channel, channel, 1, 1, 0, 1),
            Res2_Conv1D_ReLU_BN(channel, scale, ks, stride, padding, dilation),
            Conv1D_ReLU_BN(channel, channel, 1, 1, 0, 1),
            SE_Block(channel, channel)
        )

    def forward(self, x):
        y = self.network(x) + x
        return y


class AttentiveStatisticPool(nn.Module):
    def __init__(self, c_in, c_mid):
        super(AttentiveStatisticPool, self).__init__()

        self.network = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=1),
            nn.Tanh(),  # seems like most implementations uses tanh?
            nn.Conv1d(c_mid, c_in, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x.shape: B x C x t
        alpha = self.network(x)
        mu_hat = torch.sum(alpha * x, dim=-1)
        var = torch.sum(alpha * x ** 2, dim=-1) - mu_hat ** 2
        std_hat = torch.sqrt(var.clamp(min=1e-9))
        y = torch.cat([mu_hat, std_hat], dim=-1)
        # y.shape: B x (c_in*2)
        return y


class ECAPA_TDNN(nn.Module):
    def __init__(self, c_in=80, c_mid=512, c_out=192):
        super(ECAPA_TDNN, self).__init__()

        self.layer1 = Conv1D_ReLU_BN(c_in, c_mid, 5, 1, 2, 1)
        self.layer2 = SE_Res2_Block(c_mid, 8, 3, 1, 2, 2)
        self.layer3 = SE_Res2_Block(c_mid, 8, 3, 1, 3, 3)
        self.layer4 = SE_Res2_Block(c_mid, 8, 3, 1, 4, 4)

        self.network = nn.Sequential(
            # Figure 2 in https://arxiv.org/pdf/2005.07143.pdf seems like groupconv?
            nn.Conv1d(c_mid * 3, 1536, kernel_size=1, groups=3),
            AttentiveStatisticPool(1536, 128),
        )

        self.bn1 = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, c_out)
        self.bn2 = nn.BatchNorm1d(c_out)

    def forward(self, x):
        # x.shape: B x C x t
        y1 = self.layer1(x)
        y2 = self.layer2(y1) + y1
        y3 = self.layer3(y1 + y2) + y1 + y2
        y4 = self.layer4(y1 + y2 + y3) + y1 + y2 + y3

        y = torch.cat([y2, y3, y4], dim=1)  # channel-wise concat
        y = self.network(y)

        y = self.linear(self.bn1(y.unsqueeze(-1)).squeeze(-1))
        y = self.bn2(y.unsqueeze(-1)).squeeze(-1)

        return y


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    x = torch.zeros(2, 80, 200)
    model = ECAPA_TDNN(80, 512, 192)
    out = model(x)
    print(model)
    print(out.shape)  # should be [2, 192]
