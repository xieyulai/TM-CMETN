import torch.nn as nn
import torch.nn.functional as F


class GeneratorFCOS(nn.Module):

    def __init__(self, d_model, voc_size):
        super(GeneratorFCOS, self).__init__()
        self.linear = nn.Linear(d_model, voc_size)      # 300-->10172/1024-->10172
        print('Using vanilla Generator')

    def forward(self, x):
        '''
        Inputs:
            x: (B, Sc, Dc)
        Outputs:
            (B, seq_len, voc_size)
        '''
        x = self.linear(x)
        # print('x.shape:\n', x.shape)
        return F.log_softmax(x, dim=-1)
