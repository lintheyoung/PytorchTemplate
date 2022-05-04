import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet5(nn.Module):

    def __init__(self, args, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = args.classes_num

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(

            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.Tanh(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.Tanh(),
            nn.Linear(84 * in_channels, args.classes_num),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits