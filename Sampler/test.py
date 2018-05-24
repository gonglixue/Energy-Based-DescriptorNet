import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def test():
    matrix = [[1, 2],[2,3],[5, 5]]
    matrix = torch.FloatTensor(matrix)

    batch = torch.zeros(2, 3, 2)
    batch[0, :, :] = matrix
    batch[1, :, :] = matrix
    sum = torch.sum(batch, dim=1)
    sum = torch.sum(sum, dim=1)
    print(sum)

    one = torch.ones(1) * 2
    print(torch.min(matrix, one))

if __name__ == '__main__':
    test()