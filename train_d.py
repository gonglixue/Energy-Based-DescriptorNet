import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import descriptor
from PIL import Image
from torch import optim
from Sampler import HamiltonianDynamics


def train_descriptor():
    batch_size = 8
    img_size = 32
    learning_rate = 0.01
    data_folder = ''
    n_samples = 25
    step_size = 0.1
    n_steps = 20


    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    train_set = dsets.ImageFolder(data_folder, transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)


    # f()
    energy_func = descriptor.E(img_size)
    energy_func.cuda()
    energy_func = nn.DataParallel(energy_func)

    # Energy function = -f()
    for images, labels in train_loader:
        images = Variable(images[:, 0, :, :]).cuda()

        # data-dependent term
        data_f = -1 * energy_func(images)   # size of batch_size

        # sample-term
        init_rnd_sample = torch.randn(1, 1, img_size, img_size)
        samples = HamiltonianDynamics.hmc_sampling(init_pos=init_rnd_sample,
                                                   energy_fn=energy_func,
                                                   n_samples=n_samples,
                                                   step_size=step_size,
                                                   n_steps=n_steps)



