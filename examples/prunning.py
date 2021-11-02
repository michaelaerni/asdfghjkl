import argparse
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import asdfghjkl as asdl
from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG


def parse_args():
    parser = argparse.ArgumentParser(description="prunning")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log", type=str, default="./.log")

    parser.add_argument("--model",
                        type=str,
                        default="resnet18",
                        choices=["resnet18", "toy"])

    parser.add_argument("--dataset",
                        type=str,
                        default="CIFAR10",
                        choices=["CIFAR10"])
    parser.add_argument("--data_dir", type=str, default="./.data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)

    parser.add_argument("--finetune", dest="finetune", action="store_true")
    parser.add_argument("--no-finetune", dest="finetune", action="store_false")
    parser.set_defaults(finetune=False)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()


def get_data(input_size,
             dataset=None,
             data_dir="./.data",
             batch_size=32,
             shuffle=True):
    transforms = {
        "train":
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(input_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]),
        "test":
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]),
    }

    datasets = {
        k: getattr(torchvision.datasets,
                   dataset)(data_dir,
                            train=True if k == "train" else False,
                            download=True,
                            transform=v)
        for k, v in transforms.items()
    }

    loaders = {
        k:
        torch.utils.data.DataLoader(v,
                                    batch_size=batch_size,
                                    shuffle=shuffle if k == "train" else False)
        for k, v in datasets.items()
    }

    return loaders


class Toy(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 1, 5)
        self.fc1 = nn.Linear(1 * 5 * 5, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(name, n_classes):
    if name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, n_classes)
        input_size = 224
        return model, input_size
    elif name == "toy":
        model = Toy(n_classes)
        input_size = 32
        return model, input_size

    return None


def train(model, loader, criterion, optimizer, device="cpu"):
    model.train()

    loss = 0.0
    corrects = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)

        loss += batch_loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)

    loss = loss / len(loader.dataset)
    acc = corrects.double() / len(loader.dataset)

    print("Train Loss: {:.4f} Acc: {:.4f}".format(loss, acc))


def test(model, loader, criterion, device="cpu"):
    model.eval()

    loss = 0.0
    corrects = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss += batch_loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)

    loss = loss / len(loader.dataset)
    acc = corrects.double() / len(loader.dataset)

    print("Test Loss: {:.4f} Acc: {:.4f}".format(loss, acc))
    return acc


def get_number_of_w_in_model(model):
    n = 0
    for x in model.parameters():
        n += torch.numel(x)
    return n


def get_flatten_w_from_model(model):
    w = [torch.flatten(x) for x in model.parameters()]
    w = torch.cat(w)
    return w


def assign_flatten_w_to_model(model, w):
    head = 0
    for x in model.parameters():
        x.data = w[head:head + torch.numel(x)].reshape(x.shape)
        head += torch.numel(x)


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, input_size = get_model(args.model, 10)
    model.to(args.device)
    loaders = get_data(input_size, args.dataset, args.data_dir,
                       args.batch_size, args.shuffle)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Fine tune
    if args.finetune:
        best_acc = 0.0
        best_model_metadata = copy.deepcopy(model.state_dict())
        for e in range(args.e):
            print("Epoch {}/{}".format(e + 1, args.e))
            train(model, loaders["train"], criterion, opt, args.device)
            acc = test(model, loaders["test"], criterion, args.device)
            if acc > best_acc:
                best_acc = acc
                best_model_metadata = copy.deepcopy(model.state_dict())
        torch.save(best_model_metadata,
                   "{}/best-{}-{}".format(args.log, args.model, args.dataset))

    model.load_state_dict(
        torch.load("{}/best-{}-{}".format(args.log, args.model, args.dataset)))
    print("Final Acc: {:.4f}".format(
        test(model, loaders["test"], criterion, args.device)))

    n = get_number_of_w_in_model(model)
    processed = []
    for j in range(n):
        fisher_for_cross_entropy(model,
                                 fisher_type=FISHER_EMP,
                                 fisher_shapes=[SHAPE_FULL],
                                 data_loader=loaders["test"])
        model.fisher_emp.update_inv()

        w = get_flatten_w_from_model(model)
        p = w * w / torch.diagonal(model.fisher_emp.inv)
        _, indices = torch.sort(p)
        for i in indices:
            if i not in processed:
                break
        processed.append(i)
        print("{}th {} iteration choose {} {}".format(j, j/n, i, w[i]))
        d = -w[i] * model.fisher_emp.inv[:, i] / model.fisher_emp.inv[i, i]

        w += d
        torch.isclose(w[i], torch.tensor([0.0]))
        assign_flatten_w_to_model(model, w)
        test(model, loaders["test"], criterion, args.device)


if __name__ == "__main__":
    main()
