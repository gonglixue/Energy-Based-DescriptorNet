import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class F(nn.Module):
    def __init__(self, img_size):
        super(F, self).__init__()
        self.img_size = img_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(16, 16), stride=3),   #
            nn.BatchNorm2d(num_features=200),
            nn.ReLU(),
            nn.Conv2d(in_channels=200, out_channels=100, kernel_size=(6, 6), stride=2),
            # nn.BatchNorm2d(num_features=100),
            nn.ReLU(),
        )
        self.fc = nn.Linear(in_features=100, out_features=1)

        self._init_weights()

    def forward(self, input_data):
        conv_feature = self.conv(input_data)
        conv_feature = conv_feature.view(len(conv_feature), -1)
        energy = self.fc(conv_feature)
        return energy

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class E(nn.Module):
    def __init__(self, img_size):
        super(F, self).__init__()
        self.img_size = img_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(16, 16), stride=3),   #
            nn.BatchNorm2d(num_features=200),
            nn.ReLU(),
            nn.Conv2d(in_channels=200, out_channels=100, kernel_size=(6, 6), stride=2),
            # nn.BatchNorm2d(num_features=100),
            nn.ReLU(),
        )
        self.fc = nn.Linear(in_features=100, out_features=1)

        self._init_weights()

    def forward(self, input_data):
        conv_feature = self.conv(input_data)
        conv_feature = conv_feature.view(len(conv_feature), -1)
        energy = self.fc(conv_feature)
        return -1 * energy

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def test_f():
    BATCH_SIZE = 1
    IMG_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCH = 5

    transform = transforms.Compose([
        transforms.RandomSizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    trainData = dsets.ImageFolder('/media/gonglixue/540f1624-ffab-4cc5-909a-214dda2aa568/dataset/DAVIS-2016/crop_val',
                                  transform)
    testData = dsets.ImageFolder('/media/gonglixue/540f1624-ffab-4cc5-909a-214dda2aa568/dataset/DAVIS-2016/crop_val',
                                 transform)

    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

    descriptor = F(IMG_SIZE)
    descriptor.cuda()
    descriptor = nn.DataParallel(descriptor)
    for images, labels in trainLoader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        energy = descriptor(images)

if __name__ == '__main__':
    test_f()



