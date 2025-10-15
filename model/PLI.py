import mne_connectivity
import numpy as np
import mne
import torch
import torch.nn as nn
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
mne.set_log_level('ERROR')
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # x.shape = (batch, dim, nodes, seq_len)
        # A.shape = (node, node)
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()
class Graph():
    def __init__(self, freqrange=(8, 13), srate=200, TH=0.3, init_alpha=2.0):
        self.freqrange = freqrange
        self.srate = srate
        self.TH = TH

    def __call__(self, data):
        return self.get_adjacency(data)

    def get_adjacency(self, data):
        data = data.cpu().detach().numpy()

        data = data.transpose(0, 2, 1)
        adj = get_wPLI(data, self.freqrange, self.srate, threshold = self.TH)
        adj = torch.from_numpy(adj)
        return adj

def get_wPLI(EEG_data, freqrange, srate , threshold):
    labels = np.arange(62)
    labels_list = [str(label) for label in labels]
    adj = np.empty([EEG_data.shape[0], 62, 62])
    for i in range(EEG_data.shape[0]):
        EEG_data1 = EEG_data[i, :, :]
        EEG_data1 = EEG_data1.reshape(-1, EEG_data1.shape[0], EEG_data1.shape[1])
        data = mne.EpochsArray(EEG_data1, info=mne.create_info(labels_list, srate), tmin=0)
        # Calculate the power spectral density
        #spectrum = data.compute_psd(method="multitaper", fmin=freqrange[0], fmax=freqrange[1], bandwidth=2.0, picks=labels_list)
        # Calculate wPLI connectivity
        wpli = mne_connectivity.spectral_connectivity_epochs(data, method='wpli', sfreq=srate,fmin=freqrange[0], fmax=freqrange[1],
                                                             mode='multitaper', faverage=True)#faverage=True,对所有epochs进行了求平均
        wpli = wpli.get_data().reshape(-1, 62)
        wpli = wpli + wpli.T
        for w in range(62):
            for r in range(62):
                if wpli[w][r] < threshold:
                    wpli[w][r] = 0
                else:
                    wpli[w][r] = 1
        adj[i] = wpli
    return adj
if __name__ == '__main__':
    x1 = torch.rand(32, 200, 62).to(device)
    model = Graph()
    ma = model(x1)
    print(ma.shape)

