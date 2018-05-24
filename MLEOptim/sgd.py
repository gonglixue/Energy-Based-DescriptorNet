import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class SGD(optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # update
                p.data.add_(-group['lr'], d_p)

        return

class MLE_SGD(optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MLE_SGD, self).__init__(params, defaults)

    def step(self, data_f, sample_f):
        batch_size = len(data_f)
        sample_size = len(sample_f)

        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue



                # first term
                avg_grad1 = torch.zeros(p.data.size()).cuda()
                for ind in range(batch_size):
                    p.grad.data.zero_()
                    scalar_term1 = data_f[ind][0]
                    scalar_term1.backward(retain_graph=True)

                    avg_grad1 += p.grad.data
                avg_grad1 /= batch_size

                # second term
                avg_grad2 = torch.zeros(p.data.size()).cuda()
                for ind in range(sample_size):
                    p.grad.data.zero_()
                    scalar_term2 = sample_f[ind][0]
                    scalar_term2.backward(retain_graph=True)

                    avg_grad2 += p.grad.data
                avg_grad1 /= sample_size

                delta = avg_grad1 - avg_grad2
                p.data.add_(group['lr'], delta)






