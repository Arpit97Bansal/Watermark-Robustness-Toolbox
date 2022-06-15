import torch

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn import functional as F

from torch.utils.data import Subset
from PIL import Image, ImageFont, ImageDraw


data_split = '1'
wmtype = "text"
wmcount = 100
lam = 0


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

