import torch
import torch.nn.functional as F
import torch.nn as nn
from gode_ATFFNET_PLI import CGP
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
class STODE(nn.Module):
    def __init__(self, in_channels, node_num=62, **kwargs):
        super().__init__()

        # build networks
        spatial_kernel_size = 62
        temporal_kernel_size = 9     # 时间纬度感受野
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * node_num)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_ode_networks = nn.ModuleList((

            st_ode(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_ode(64, 64, kernel_size, 1, **kwargs),
            st_ode(64, 64, kernel_size, 1, **kwargs),
            st_ode(64, 64, kernel_size, 1, **kwargs),
            st_ode(64, 128, kernel_size, 2, **kwargs),
            st_ode(128, 128, kernel_size, 1, **kwargs),
            st_ode(128, 128, kernel_size, 1, **kwargs),
            st_ode(128, 256, kernel_size, 2, **kwargs),
            st_ode(256, 256, kernel_size, 1, **kwargs),
            st_ode(256, 256, kernel_size, 1, **kwargs),
        ))
        # fcn for prediction
        # self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # self.SoftMax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        N, C, T, V = x.size()
        x = x.transpose(2,3).contiguous()    # N, C, V, T
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).transpose(2,3).contiguous()    # N, C, T, V

        # forward
        for ode in self.st_ode_networks:
            x = ode(x, adj)

        # 池化
        #x = F.avg_pool2d(x, x.size()[2:])
        # x = self.fcn(x)
        # x = x.view(x.size(0), -1)
        # x = self.SoftMax(x)

        return x

class st_ode(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,       # 时间维度上的感知野,邻接矩阵的分区数
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gode = CGP(cin=in_channels, cout=out_channels, nodenum=62, alpha=2.0, method='euler', time=1.0, step_size=1.0,
                 rtol=1e-4, atol=1e-3, perturb=False)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:#残差
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):

        res = self.residual(x)
        x = x.permute(0, 1, 3, 2)
        x= self.gode(x, adj)
        x = x.permute(0, 1, 3, 2)
        x = self.tcn(x) + res
        return self.relu(x)

if __name__ == '__main__':
    x1 = torch.rand(4, 64, 200, 62).to(device)
    mo = STODE(64).to(device)
    y = mo(x1)
    print(y.shape)