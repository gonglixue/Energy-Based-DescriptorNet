import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import optim

from PIL import Image

from model import descriptor
from Sampler import HamiltonianDynamics
from MLEOptim import MLE_SGD


def train_descriptor():
    batch_size = 8
    img_size = 32
    learning_rate = 0.01
    data_folder = '/media/gonglixue/540f1624-ffab-4cc5-909a-214dda2aa568/dataset/weizmann_horse_db/origin_size_crop_horses'
    n_samples = 25
    step_size = 0.1
    n_steps = 20


    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_set = dsets.ImageFolder(data_folder, transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=True)


    # f()
    energy_func = descriptor.E(img_size)
    energy_func.cuda()
    energy_func = nn.DataParallel(energy_func)

    mle_optimizer = MLE_SGD(energy_func.parameters(), lr=0.001)

    # Energy function = -f()
    for batch_id, (data, labels) in enumerate(train_loader):

        images = torch.zeros(batch_size, 1, img_size, img_size)
        images[:, 0, :, :] = data[:, 0, :, :]
        images = Variable(images).cuda()

        # mle_optimizer.zero_grad()

        # data-dependent term
        data_f = -1 * energy_func(images)   # size of batch_size * 1

        # sample-term
        init_rnd_sample = torch.randn(1, 1, img_size, img_size).cuda()
        samples_batch = HamiltonianDynamics.hmc_sampling(init_pos=init_rnd_sample,
                                                   energy_fn=energy_func,
                                                   n_samples=n_samples,
                                                   step_size=step_size,
                                                   n_steps=n_steps)

        samples_batch = Variable(samples_batch).cuda()
        samples_f = -1 * energy_func(samples_batch)


        mle_optimizer.step(data_f, samples_f)

        print("iter")

if __name__ == '__main__':
    train_descriptor()





