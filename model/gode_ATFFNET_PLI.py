import torch.nn as nn
import torch
import numpy as np
import torchdiffeq
from PLI import Graph
import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # x.shape = (batch, dim, nodes, seq_len)
        # A.shape = (node, node)
        x = x.double()
        A = A.double()
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()
# 生成邻接矩阵
class GATENet(nn.Module):
    def __init__(self, inc, reduction_ratio=128):
        super(GATENet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias=False),
                                nn.ELU(inplace=False),
                                nn.Linear(inc // reduction_ratio, inc, bias=False),
                                nn.Tanh(),
                                nn.ReLU(inplace=False))

    def forward(self, x):
        y = self.fc(x)
        return y
class CGPFunc(nn.Module):

    def __init__(self, c_in, c_out, init_alpha):
        super(CGPFunc, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.x0 = None
        self.adj = None
        self.nfe = 0
        self.alpha = init_alpha
        self.nconv = nconv()
        self.out = []

    def forward(self, t, x):
        row,_ = self.adj.shape
        adj = self.adj.to(x.device) + torch.eye(row).to(x.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(_d, adj), _d)
        self.out.append(x)
        self.nfe += 1
        ax = self.nconv(x, adj_norm)
        f = 0.5 * self.alpha * (ax - x)
        return f
class CGPODEBlock(nn.Module):
    def __init__(self, cgpfunc, method, step_size, rtol, atol, perturb, estimated_nfe):
        super(CGPODEBlock, self).__init__()
        self.odefunc = cgpfunc
        self.method = method
        self.step_size = step_size
        self.perturb = perturb
        self.atol = atol
        self.rtol = rtol
        self.mlp = linear((estimated_nfe + 1) * self.odefunc.c_in, self.odefunc.c_out)

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def set_adj(self, adj):
        self.odefunc.adj = adj

    def forward(self, x, t):
        self.integration_time = torch.tensor([0, t]).float().type_as(x)
        out = torchdiffeq.odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                     method=self.method, options=dict(step_size=self.step_size,
                                                                      perturb=self.perturb))
        #print("CGPODEBlock",out.shape)
        outs = self.odefunc.out
        self.odefunc.out = []
        outs.append(out[-1])
        h_out = torch.cat(outs, dim=1)
        #print("CGPODEBlock", h_out.shape)
        # h_out = h_out.permute(0, 2, 1, 3)
        h_out = h_out.float()
        h_out = self.mlp(h_out)
        #print("CGPODEBlock", h_out.shape)
        return h_out


class CGP(nn.Module):
    def __init__(self, cin, cout, nodenum=62, alpha=2.0, method='euler', time=1.0, step_size=0.25,
                 rtol=1e-4, atol=1e-3, perturb=False):

        super(CGP, self).__init__()
        self.c_in = cin
        self.c_out = cout
        self.alpha = alpha
        self.adj = torch.rand((1, nodenum * nodenum), dtype=torch.float32, requires_grad=False).cuda()
        self.GATENet = GATENet(nodenum * nodenum, reduction_ratio=128)

        if method == 'euler':
            self.integration_time = time
            self.estimated_nfe = round(self.integration_time / step_size)
        elif method == 'rk4':
            self.integration_time = time
            self.estimated_nfe = round(self.integration_time / (step_size / 4.0))
        else:
            raise ValueError("Oops! The CGP solver is invaild.")

        self.CGPODE = CGPODEBlock(CGPFunc(self.c_in, self.c_out, self.alpha), method,
                                  step_size, rtol, atol, perturb,
                                  self.estimated_nfe)

    def forward(self, x, adj_PLI):
        self.CGPODE.set_x0(x)
        adj_ds = self.GATENet(self.adj)
        adj_ds = adj_ds.reshape(62, 62)
        adj_PLI = torch.as_tensor(adj_PLI)
        adj_PLI = adj_PLI.to(device).float()
        adj_xs = torch.matmul(adj_PLI, adj_ds)
        self.CGPODE.set_adj(adj_xs)
        h = self.CGPODE(x, self.integration_time)
        #h = h.squeeze(dim=2)
        return h
if __name__ == '__main__':
    x1 = torch.rand(32, 200, 62)
    model = CGP(cin=62, cout=62, alpha=2.0, method='euler', time=1.2, step_size=0.4,
                 rtol=1e-4, atol=1e-3, perturb=False)
    ma = model(x1)
    print(ma.shape)
