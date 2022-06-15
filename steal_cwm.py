""" This script trains null models given a configuration file (see configs) """

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from shutil import copyfile

import mlconfig
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import wrt.training
from wrt.attacks import RemovalAttack

from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import PyTorchClassifier
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets.cifar10 import cifar_classes
from wrt.utils import reserve_gpu, get_max_index

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn import functional as F

from torch.utils.data import Subset
from PIL import Image, ImageFont, ImageDraw

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack_config', type=str,
                        default='configs/imagenet/attack_configs/label_smoothing.yaml',
                        help="Path to config file for the attack.")
    parser.add_argument('-w', "--wm_dir", type=str,
                        default="outputs/imagenet/wm/jia/00000_jia",
                        help="Path to the directory with the watermarking files. "
                             "This scripts expects a 'best.pth' and one '*.yaml' file "
                             "to exist in this dir.")
    parser.add_argument('-r', "--resume", type=str,
                        default=None,
                        help="Path to checkpoint to continue the attack. ")
    parser.set_defaults(true_labels=False, help="Whether to use ground-truth labels.")
    parser.add_argument('--true_labels', action='store_true')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")

    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes, defense_filename: str = None):
    """ Loads a source model from a directory and wraps it into a pytorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    # Load defense model from a saved state, if available.
    # We allow loading the optimizer, as it only loads states that the attacker could tune themselves (E.g. learning rate)
    if defense_filename is not None:
        pretrained_data = torch.load(defense_filename)
        model.load_state_dict(pretrained_data["model_state_dict"])
        try:
            optimizer.load_state_dict(pretrained_data["optimizer"])
        except:
            print("Optimizer could not be loaded. ")
            pass

        print(f"Loaded model and optimizer from '{defense_filename}'.")

    model = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model


def file_with_suffix_exists(dirname, suffix, not_contains="", raise_error=False):
    for file in os.listdir(dirname):
        if file.endswith(suffix) and (not not_contains in file or len(not_contains) == 0):
            return os.path.abspath(os.path.join(dirname, file))
    if raise_error:
        raise FileNotFoundError(f"No file found with suffix '{suffix}' in '{dirname}")
    return False

def sanity(net):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)
    test_accuracy = 0.0
    net.eval()
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)

        max_vals, max_indices = torch.max(outputs, 1)

        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        test_accuracy += 100 * correct
    test_accuracy /= len(testloader)
    print("Sanity Check Test Acc ", test_accuracy)

def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    reserve_gpu(args.gpu)
    device = "cuda"

    # Discover the '*.yaml' config file and the 'best.pth' file.
    defense_yaml = "/cmlscratch/bansal01/spring_2022/Watermark-Robustness-Toolbox/resnet.yaml"
    defense_config = mlconfig.load(defense_yaml)
    # print(defense_config)

    attack_config = mlconfig.load(args.attack_config)
    print(attack_config)
    pth_file = args.wm_dir

    # Create output folder.
    if not os.path.exists(attack_config.output_dir):
        os.makedirs(attack_config.output_dir)
    output_dir = os.path.join(attack_config.output_dir,
                              f"{get_max_index(attack_config.output_dir, suffix=attack_config.create.name).zfill(5)}_"
                              f"{attack_config.create.name}")
    os.makedirs(output_dir)
    print(f"======> Logging outputs to '{os.path.abspath(output_dir)}'")
    print(f"Saving outputs? {args.save}")

    # Save the cmd line arguments.
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    source_model = ResNet18()
    source_model = source_model.to(device)
    source_model = torch.nn.DataParallel(source_model, device_ids=range(torch.cuda.device_count()))

    checkpoint = torch.load(pth_file)
    source_model.load_state_dict(checkpoint['model_state_dict'])
    source_model = source_model.module

    sanity(source_model)

    optimizer = torch.optim.Adam(source_model.parameters(), lr=0.0001)

    criterion = torch.nn.CrossEntropyLoss()
    source_model = PyTorchClassifier(
        model=source_model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10
    )


    # defense = defense_config.wm_scheme(classifier=source_model, optimizer=optimizer, config=defense_config)
    # x_wm, y_wm = defense.load(filename=model_filename, path=model_basedir)
    # just load the x_wm and y_wm in this
    watermarkset = torch.load("/cmlscratch/bansal01/spring_2022/Watermark-Robustness-Toolbox/watermarkset.pth")
    x_wm = []
    y_wm = []

    for data in watermarkset:
        x, y = data
        x_wm.append(x)
        y_wm.append(torch.tensor(y))

    x_wm = torch.stack(x_wm).cuda()
    y_wm = torch.stack(y_wm).cuda()


    use_gt = args.true_labels or ("true_labels" in attack_config.keys() and attack_config.true_labels)
    print(f"Using ground truth labels? {use_gt}")
    if use_gt:
        print("Using ground-truth labels ..")
        train_loader = attack_config.dataset(train=True)
        valid_loader = attack_config.dataset(train=False)
    else:
        print("Using predicted labels ..")
        train_loader = attack_config.dataset(source_model=source_model, train=True)
        valid_loader = attack_config.dataset(source_model=source_model, train=False)

    defense = defense_config.wm_scheme(classifier=source_model, wm_set=watermarkset)

    source_test_acc_before_attack = evaluate_test_accuracy(source_model, valid_loader)
    print(f"Source model test acc: {source_test_acc_before_attack:.4f}")
    source_wm_acc = defense.verify(x_wm, y_wm, classifier=source_model)[0]
    print(f"Source model wm acc: {source_wm_acc:.4f}")



    if "surrogate_model" in attack_config.keys():
        surrogate_model = attack_config.surrogate_model()
        optimizer = attack_config.optimizer(surrogate_model.parameters())
        surrogate_model = __load_model(surrogate_model, optimizer,
                                       image_size=attack_config.surrogate_model.image_size,
                                       num_classes=attack_config.surrogate_model.num_classes)
    else:
        surrogate_model = deepcopy(source_model)


    if args.resume is not None:
        print(f"Resuming from checkpoint '{args.resume}' ... ")
        pretrained_data = torch.load(args.resume)
        surrogate_model.model.load_state_dict(pretrained_data["model"])
        try:
            surrogate_model.optimizer.load_state_dict(pretrained_data["optimizer"])
        except:
            pass

    surrogate_test_acc_before_attack, surrogate_wm_acc_before_attack = -1, -1
    try:
        surrogate_test_acc_before_attack = evaluate_test_accuracy(surrogate_model, valid_loader)
        print(f"Surrogate model test acc: {surrogate_test_acc_before_attack:.4f}")
        surrogate_wm_acc_before_attack = defense.verify(x_wm, y_wm, classifier=surrogate_model)[0]
        print(f"Surrogate model wm acc: {surrogate_wm_acc_before_attack:.4f}")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Could not extract watermark accuracy from the surrogate model ... Continuing ..")

    attack: RemovalAttack = attack_config.create(classifier=surrogate_model, config=attack_config)


    # Run the removal. We still need wrappers to conform to the old interface.
    start = time.time()
    attack, train_metric = attack_config.remove(attack=attack,
                                                source_model=source_model,
                                                train_loader=train_loader,
                                                valid_loader=valid_loader,
                                                config=attack_config,
                                                output_dir=output_dir,
                                                wm_data=(defense, x_wm, y_wm))
    end = time.time()
    execution_time = end - start


    surrogate_model = attack.get_classifier()
    surrogate_test_acc_after_attack = evaluate_test_accuracy(surrogate_model, valid_loader)
    print(f"Surrogate model test acc: {surrogate_test_acc_after_attack:.4f}")
    surrogate_wm_acc_after_attack = defense.verify(x_wm, y_wm, classifier=surrogate_model)[0]
    print(f"Surrogate model wm acc: {surrogate_wm_acc_after_attack:.4f}")

    print("Execution Time ", execution_time)

    net = surrogate_model.model

    ############################################################
    # EVAL
    ############################################################
    wmloader = torch.utils.data.DataLoader(
        watermarkset, batch_size=100, shuffle=True, num_workers=1, drop_last=True)

    # A new classifier g
    Array = []
    times = 100
    net.eval()
    wm_train_accuracy_avg = 0.0
    for j in range(times):
        Noise = {}
        # Add noise
        for name, param in net.named_parameters():
            gaussian = torch.randn_like(param.data)
            Noise[name] = 1.0 * gaussian
            param.data = param.data + Noise[name]

        wm_running_loss = 0.0
        wm_train_accuracy = 0.0
        for i, data in enumerate(wmloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)

            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            wm_train_accuracy += 100 * correct

        wm_train_accuracy /= len(wmloader)
        wm_train_accuracy_avg += wm_train_accuracy
        Array.append(wm_train_accuracy)

        # remove the noise
        for name, param in net.named_parameters():
            param.data = param.data - Noise[name]

    wm_train_accuracy_avg /= times
    Array.sort()
    wm_median = Array[int(len(Array) / 2)]

    print("Avg White Box ", wm_train_accuracy_avg)
    print("Median White Box ", wm_median)

    exit()
    if args.save:
        with open(os.path.join(output_dir, 'result.json'), "w") as f:
            json.dump({
                "test_acc_before": surrogate_test_acc_before_attack,
                "wm_acc_before": surrogate_wm_acc_before_attack,
                "test_acc_after": surrogate_test_acc_after_attack,
                "wm_acc_after": surrogate_wm_acc_after_attack,
                "time": execution_time
                       }, f)

        if train_metric is None:
            train_metric = {}

        # Save the model and the watermarking key.
        checkpoint = {
            "model": surrogate_model.model.state_dict(),
            "optimizer": surrogate_model.optimizer.state_dict(),
            "x_wm": x_wm,
            "y_wm": y_wm
        }
        torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))

        # Save the training metrics.
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            all_metrics = {
                "source_test_acc": source_test_acc_before_attack,
                "source_wm_acc": source_wm_acc,
                "surr_test_acc_before": surrogate_test_acc_before_attack,
                "surr_wm_acc_before": surrogate_wm_acc_before_attack,
                **train_metric
            }
            json.dump(all_metrics, f)
        print(f"Successfully saved data to '{os.path.abspath(output_dir)}'")


if __name__ == "__main__":
    main()
