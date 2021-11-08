import argparse
import copy
#from functools import reduce

import matplotlib.pyplot as plt
import torch
from torch import nn
#from torch.nn.utils import prune
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
                        default="toy1",
                        choices=["resnet18", "toy1", "toy2"])

    parser.add_argument("--dataset",
                        type=str,
                        default="MNIST",
                        choices=["MNIST", "CIFAR10"])
    parser.add_argument("--data_dir", type=str, default="./.data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)

    parser.add_argument("--pretrain", dest="pretrain", action="store_true")
    parser.add_argument("--no-pretrain", dest="pretrain", action="store_false")
    parser.set_defaults(pretrain=True)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()


def calc_moments(loader):
    n = 0
    s = None
    sq = None
    for x, _ in loader:
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        n += b * h * w
        x2 = x.pow(2)
        x = x.sum((0, 2, 3))
        x2 = x2.sum((0, 2, 3))
        s = x if s is None else s + x
        sq = x2 if sq is None else sq + x2
    mean = s / n
    std = ((sq - n * mean.pow(2)) / (n - 1)).pow(0.5)
    print(f"n: {n/h/w}, mean: {mean}, std: {std}")
    return n / h / w, mean, std


def get_dataset_metadata(dataset):
    return {
        "MNIST": {
            "img_shape": (1, 32, 32),
            "n_classes": 10,
            "mean": (0.1307, ),
            "std": (0.3081, )
        },
        "CIFAR10": {
            "img_shape": (3, 32, 32),
            "n_classes": 10,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    }[dataset]


def get_data(input_size,
             dataset=None,
             data_dir="./.data",
             batch_size=32,
             shuffle=True):
    metadata = get_dataset_metadata(dataset)

    transforms = {
        "train":
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(input_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(metadata["mean"],
                                             metadata["std"]),
        ]),
        "test":
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(metadata["mean"],
                                             metadata["std"]),
        ]),
    }
    trains = {"train": True, "test": False}
    shuffles = {"train": shuffle, "test": False}

    dataset = getattr(torchvision.datasets, dataset)
    datasets = {
        k: dataset(data_dir,
                   train=trains[k],
                   download=True,
                   transform=transforms[k])
        for k in ["train", "test"]
    }

    loaders = {
        k: torch.utils.data.DataLoader(datasets[k],
                                       batch_size=batch_size,
                                       shuffle=shuffles[k])
        for k in ["train", "test"]
    }

    return loaders


class Toy1(nn.Module):
    def __init__(self, img_shape, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(img_shape[0], 1, 3)
        self.pool = nn.MaxPool2d(10, 10)
        self.conv2 = nn.Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(1, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class Toy2(nn.Module):
    def __init__(self, img_shape, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(img_shape[0], 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 1, 5)
        self.fc1 = nn.Linear(1 * 5 * 5, 16)
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(name, dataset):
    metadata = get_dataset_metadata(dataset)
    if name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, metadata["n_classes"])
        input_size = 224
        return model, input_size
    elif name == "toy1":
        model = Toy1(metadata["img_shape"], metadata["n_classes"])
        input_size = 32
        return model, input_size
    elif name == "toy2":
        model = Toy2(metadata["img_shape"], metadata["n_classes"])
        input_size = 32
        return model, input_size
    else:
        return None


def train(model, loader, optimizer, criterion, device="cpu"):
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


def pretrain(model, loaders, opt, criterion, args):
    best_acc = 0.0
    best_model_metadata = copy.deepcopy(model.state_dict())
    for e in range(args.e):
        print("Epoch {}/{}".format(e + 1, args.e))
        train(model, loaders["train"], opt, criterion, args.device)
        acc = test(model, loaders["test"], criterion, args.device)
        if acc > best_acc:
            best_acc = acc
            best_model_metadata = copy.deepcopy(model.state_dict())
    torch.save(best_model_metadata,
               "{}/best-{}-{}".format(args.log, args.model, args.dataset))


class Scope(object):
    def __init__(self, m_name, m, p_name, p, l, r):
        self.m_name = m_name
        self.m = m
        self.p_name = p_name
        self.p = p
        self.l = l
        self.r = r
        self.n = r - l
        self.n_zero = 0

    def has(self, index):
        return self.l <= index and index < self.r

    @property
    def sparsity(self):
        return self.n_zero / self.n


class OptimalBrainSurgeon(object):
    def __init__(self, model, scopes, fisher_shapes, fisher_type=FISHER_EMP):
        self.model = model
        self.scopes = scopes
        self.fisher_shapes = fisher_shapes
        self.fisher_type = fisher_type
        self.init_mask()
        self.n = len(self.parameters)
        self.n_zero = 0
        print(self)

    ##### Parameter #####
    @property
    def parameters(self):
        return self.get_flatten_parameters()

    def get_flatten_parameters(self):
        # TODO(sxwang): Does named_parameters yield in order? We should
        # guarantee this order is the same as the order in fisher. OrderedDict
        # looks like ok.
        return nn.utils.parameters_to_vector([
            v for k, v in self.model.named_parameters()
            if k in self.scopes.keys()
        ])

    @parameters.setter
    def parameters(self, p):
        self.assign_flatten_parameters(p)

    def assign_flatten_parameters(self, p):
        head = 0
        for x in self.model.parameters():
            x.data = p[head:head + torch.numel(x)].reshape(x.shape)
            head += torch.numel(x)

    ##### Mask #####
    def mask_name(self, p):
        return p + "_mask"

    def init_mask(self):
        for _, v in self.scopes.items():
            v.m.register_buffer(self.mask_name(v.p_name), torch.ones_like(v.p))
            # ASDL doesn't use this gradient
            #v.p.register_hook(
            #    lambda grad: grad * getattr(v.m, self.mask_name(v.p_name)))

    def get_mask_by_scope(self, scope):
        return getattr(scope.m, self.mask_name(scope.p_name))

    @property
    def mask(self):
        return self.get_flatten_mask()

    def get_flatten_mask(self):
        # TODO(sxwang): Does named_parameters yield in order? We should
        # guarantee this order is the same as the order in fisher. OrderedDict
        # looks like ok.
        return nn.utils.parameters_to_vector([
            v for k, v in self.model.named_buffers()
            if k[:-5] in self.scopes.keys()
        ])

    ##### Scope #####
    def get_scope_by_index(self, index):
        for v in self.scopes.values():
            if v.has(index):
                return v

    ##### Prunning #####
    def prune(self, loader):
        fisher_for_cross_entropy(self.model,
                                 fisher_type=self.fisher_type,
                                 fisher_shapes=self.fisher_shapes,
                                 data_loader=loader)
        fisher = getattr(self.model, self.fisher_type)

        mask = self.mask
        fisher.data *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
        fisher.update_inv(1e-3)

        scores = self.parameters.pow(2) / torch.diagonal(fisher.inv)
        scores -= scores.max()
        scores *= mask

        _, indices = torch.sort(scores)
        i = indices[0].item()

        scope = self.get_scope_by_index(i)
        self.get_mask_by_scope(scope).view(-1)[i - scope.l] = 0.0
        scope.n_zero += 1

        parameters = self.parameters
        d = -parameters[i] * fisher.inv[:, i] / fisher.inv[i, i]
        d[i] = -parameters[i]
        parameters += d
        self.parameters = parameters

        self.n_zero += 1

        mask = self.mask
        torch.isclose(mask[i], torch.tensor([0.0]))
        parameters = self.parameters
        torch.allclose(parameters.masked_select(mask < 1),
                       torch.zeros(self.n_zero))
        print(f"=============================================================="
              f"\nPrunning No.{i} parameter in {scope.m_name}.{scope.p_name}"
              f"\nscores: {scores[i]}\n{scores}"
              f"\nd: \n{d}"
              f"\nparameters: {parameters[i]}\n{parameters}"
              f"\nmask: {mask[i]}\n{mask}"
              f"\nsparsity: {self.n_zero}/{self.n}={self.n_zero/self.n}")

    @property
    def sparsity(self):
        return self.n_zero / self.n

    def __str__(self):
        info = ["Pruning Scope:"]
        fmt = "{:<15} {:<5} {:<5} {:<5} {:<5} {:<5}"
        info += [fmt.format("parameter", "l", "r", "n", "zero", "sparsity")]
        info += [
            fmt.format(k, v.l, v.r, v.n, v.n_zero, v.sparsity)
            for k, v in self.scopes.items()
        ]
        info += [f"Total sparsity: {self.n_zero}/{self.n}={self.sparsity}"]
        return "\n".join(info)


def list_model(module, prefix="", condition=lambda _: True):
    modules = {}
    has_children = False
    for name, x in module.named_children():
        has_children = True
        new_prefix = prefix + ("" if prefix == "" else ".") + name
        modules.update(list_model(x, new_prefix, condition))
    if not has_children and condition(module):
        modules[prefix] = module
    return modules


def get_global_prunning_scope(model):
    modules = list_model(model, condition=lambda x: hasattr(x, "weight"))
    scopes = {}
    l = 0
    for m_name, m in modules.items():
        for p_name, p in m.named_parameters():
            r = l + torch.numel(p)
            scopes[f"{m_name}.{p_name}"] = Scope(m_name, m, p_name, p, l, r)
            l = r
    return scopes
    #return {k: v for k, v in model.named_parameters()}


def prunning(model, loaders, criterion, args):
    obs = OptimalBrainSurgeon(model,
                              get_global_prunning_scope(model),
                              fisher_shapes=[SHAPE_FULL])
    x = []
    y = []
    for i in range(obs.n):
        obs.prune(loaders["train"])
        acc = test(model, loaders["test"], criterion, args.device)
        x.append(obs.sparsity)
        y.append(acc)
        torch.save(copy.deepcopy(model.state_dict()),
                   f"{args.log}/pruned-{args.model}-{args.dataset}-{i}-{acc}")
    print(obs)
    plt.plot(x, y, label=f"{args.model}-{args.dataset}")
    plt.xlabel("sparsity")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig(f"{args.log}/pruned-{args.model}-{args.dataset}.svg")
    #plt.show()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, input_size = get_model(args.model, args.dataset)
    model.to(args.device)
    loaders = get_data(input_size, args.dataset, args.data_dir,
                       args.batch_size, args.shuffle)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if not args.pretrain:
        pretrain(model, loaders, opt, criterion, args)

    model.load_state_dict(
        torch.load("{}/best-{}-{}".format(args.log, args.model, args.dataset)))
    model.to(args.device)

    prunning(model, loaders, criterion, args)


if __name__ == "__main__":
    main()
