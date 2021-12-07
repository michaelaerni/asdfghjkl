import argparse

import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torchvision import transforms

import asdfghjkl as asdl


def parse_args():
    parser = argparse.ArgumentParser(description="unit-wise")
    parser.add_argument("--model",
                        type=str,
                        default="resnet18",
                        choices=["resnet18"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data", type=str, default="./.data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--log_intvl", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log", type=str, default="./.log")

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--stat_intvl", type=int, default=10)
    parser.add_argument("--inv_intvl", type=int, default=100)

    return parser.parse_args()


def train(model, loader, loss_fn, opt, metrics, device, epoch, args):
    model.train()
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if i % args.stat_intvl == 0:
            #update_stat(model, inputs, targets)
            opt.kfac.update_curvature(inputs, targets)
            opt.kfac.accumulate_curvature(to_pre_inv=True)

        opt.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()

        if i % args.inv_intvl == 0:
            #calc_inverse_of_model(model, args)
            opt.kfac.update_inv()

        #precondition_model(model)
        opt.kfac.precondition()
        opt.step()
        metrics += (output, targets)
        if i % args.log_intvl == 0 or i == len(loader) - 1:
            print("Epoch {} Train: {}".format(epoch, metrics))
    kmetrics = {"train_loss": metrics[1](), "train_acc": metrics[2]()}
    #wandb.log(kmetrics, commit=False)


def test(model, loader, metrics, epoch, device):
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            metrics += (output, targets)
    #wandb.log({"test_loss": metrics[1](), "test_acc": metrics[2]()})
    cprint("red")("Epoch {} Test: {}".format(epoch, metrics))


def get_model(n_input, n_output):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_input, 128),
        nn.ReLU(),
        nn.Linear(128, n_output),
    )

    for p in model.parameters():
        if len(p.data.shape) > 1:
            nn.init.xavier_uniform_(p.data)
        else:
            p.data = torch.zeros(p.data.shape)

    return model


def MNIST(data="./.data", batch_size=32, shuffle=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])

    train_datasets = datasets.MNIST(data,
                                    train=True,
                                    download=True,
                                    transform=transform)
    train_loader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    test_datasets = datasets.MNIST(data, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_datasets,
                                              batch_size=batch_size)
    print((iter(train_loader).next())[0].shape)
    print((iter(train_loader).next())[1].shape)
    return train_loader, test_loader


def CIFAR10(data="./.data", batch_size=32, shuffle=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_datasets = datasets.CIFAR10(data,
                                      train=True,
                                      download=True,
                                      transform=transform)
    train_loader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=batch_size,
                                               shuffle=shuffle)
    test_datasets = datasets.CIFAR10(data, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_datasets,
                                              batch_size=batch_size)
    print((iter(train_loader).next())[0].shape)
    print((iter(train_loader).next())[1].shape)
    return train_loader, test_loader


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    train_loader, test_loader = CIFAR10(args.data, args.batch_size,
                                        args.shuffle)
    model = get_model(784, 10)
    model.to(args.device)

    inputs, targets = iter(train_loader).next()
    mgr = asdl.fisher_for_cross_entropy(model,
                                        fisher_type=asdl.FISHER_EMP,
                                        fisher_shapes=[asdl.SHAPE_UNIT_WISE],
                                        inputs=inputs,
                                        targets=targets)


if __name__ == "__main__":
    main()
