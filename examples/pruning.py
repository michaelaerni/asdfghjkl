import argparse
import copy
#from functools import reduce
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
#from torch.nn.utils import prune
import torch.nn.functional as F
import torchvision
import wandb

import asdfghjkl as asdl
from asdfghjkl import fisher_for_cross_entropy
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG
from asdfghjkl.utils import add_value_to_diagonal, cholesky_inv


def parse_args():
    parser = argparse.ArgumentParser(description="prnning")
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
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)

    parser.add_argument("--pretrain", dest="pretrain", action="store_true")
    parser.add_argument("--no-pretrain", dest="pretrain", action="store_false")
    parser.set_defaults(pretrain=True)
    parser.add_argument("--e", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-1)

    parser.add_argument("--pruning_strategy",
                        type=str,
                        default="oneshot",
                        choices=["oneshot", "gradual"])
    parser.add_argument(
        "--fisher_type",
        type=str,
        default="fisher_emp",
        choices=["fisher_exact", "fisher_mc", "fisher_emp", "wood_fisher"])
    parser.add_argument(
        "--fisher_shape",
        type=str,
        default="full",
        choices=["full", "layer_wise", "kron", "unit_wise", "none"])
    parser.add_argument("--kfac_fast_inv",
                        dest="kfac_fast_inv",
                        action="store_true")
    parser.set_defaults(kfac_fast_inv=False)
    parser.add_argument("--layer_normalize",
                        dest="layer_normalize",
                        action="store_true")
    parser.set_defaults(layer_normalize=False)

    parser.add_argument("--check", dest="check", action="store_true")
    parser.set_defaults(check=False)

    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--n_recompute", type=int, default=10)
    parser.add_argument("--n_recompute_samples", type=int, default=4096)
    parser.add_argument("--test_intvl", type=float, default=0.05)

    args = parser.parse_args()

    args.model_dir = f"{args.log}/{args.model}-{args.dataset}"
    args.log = f"{args.log}/{args.model}-{args.dataset}/" \
               f"{args.pruning_strategy}/{args.fisher_shape}/" \
               f"{args.n_recompute}-{args.n_recompute_samples}"
    Path(args.log).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(f"{args.log}/log"),
                            logging.StreamHandler()
                        ])

    wandb.init(project="pruning")
    wandb.run.name = args.log

    return args


def to_vector(parameters):
    return nn.utils.parameters_to_vector(parameters)


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
    logging.info(f"n: {n/h/w}, mean: {mean}, std: {std}")
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

    logging.info(f"Train Loss: {loss:.4f} Acc: {acc:.4f}")


def test(model, loader, criterion, device="cpu", prefix=""):
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

    logging.info(f"{prefix}Test Loss: {loss:.4f} Acc: {acc:.4f}")
    return acc


def pretrain(model, loaders, opt, criterion, args):
    best_acc = 0.0
    best_model_metadata = copy.deepcopy(model.state_dict())
    for e in range(args.e):
        logging.info(f"Epoch {e + 1}/{args.e}")
        train(model, loaders["train"], opt, criterion, args.device)
        acc = test(model, loaders["test"], criterion, args.device)
        if acc > best_acc:
            best_acc = acc
            best_model_metadata = copy.deepcopy(model.state_dict())
    torch.save(best_model_metadata, f"{args.model_dir}/best-{best_acc}")
    torch.save(best_model_metadata, f"{args.model_dir}/best")


def polynomial_schedule(start, end, i, n):
    scale = end - start
    progress = min(float(i) / n, 1.0)
    remaining_progress = (1.0 - progress)**2
    return end - scale * remaining_progress


class Scope(object):
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.n_weight = torch.numel(self.weight)
        self.n_bias = torch.numel(self.bias) if self.has_bias else 0
        self.n = self.n_weight + self.n_bias
        self.n_zero = 0
        self.init_mask()
        self.ifisher = None
        self.ifisher_diag = None
        self.pruned = set()

    @property
    def weight(self):
        return self.module.weight

    @weight.setter
    def weight(self, w):
        self.module.weight.data = w.reshape(self.module.weight.shape)

    def weight_iadd(self, w):
        self.module.weight.data += w.reshape(self.module.weight.shape)

    @property
    def has_bias(self):
        return self.module.bias is not None

    @property
    def bias(self):
        return self.module.bias

    @bias.setter
    def bias(self, b):
        self.module.bias.data = b.reshape(self.module.bias.shape)

    def bias_iadd(self, b):
        self.module.bias.data += b.reshape(self.module.bias.shape)

    @property
    def parameters(self):
        if self.has_bias:
            return to_vector([self.weight, self.bias])
        else:
            return to_vector([self.weight])

    @parameters.setter
    def parameters(self, p):
        self.weight = p[:self.n_weight]
        if self.has_bias:
            self.bias = p[self.n_weight:]

    def parameters_iadd(self, p):
        self.weight_iadd(p[:self.n_weight])
        if self.has_bias:
            self.bias_iadd(p[self.n_weight:])

    def init_mask(self):
        self.module.register_buffer("weight_mask",
                                    torch.ones_like(self.module.weight))
        self.module.weight.register_hook(
            lambda grad: grad * getattr(self.module, "weight_mask"))
        if self.has_bias:
            self.module.register_buffer("bias_mask",
                                        torch.ones_like(self.module.bias))
            self.module.bias.register_hook(
                lambda grad: grad * getattr(self.module, "bias_mask"))

    @property
    def weight_mask(self):
        return getattr(self.module, "weight_mask")

    @property
    def bias_mask(self):
        return getattr(self.module, "bias_mask")

    @property
    def mask(self):
        if self.has_bias:
            return to_vector([self.weight_mask, self.bias_mask])
        else:
            return to_vector([self.weight_mask])

    @property
    def grad(self):
        if self.has_bias:
            return to_vector([self.weight.grad, self.bias.grad])
        else:
            return to_vector([self.weight.grad])

    def score(self, diag_fisher_inv):
        scores = self.parameters.pow(2) / diag_fisher_inv
        return scores.masked_fill(self.mask == 0.0, float("inf"))

    def prune(self, i, d=None, check=False, log=False):
        assert i not in self.pruned
        assert i < self.n
        assert self.mask[i] == 1.0
        self.pruned.add(i)

        with torch.no_grad():
            if d is not None:
                self.parameters_iadd(d)
            if i < self.n_weight:
                self.weight.view(-1)[i] = 0.0
                self.weight_mask.view(-1)[i] = 0.0
            else:
                self.bias.view(-1)[i - self.n_weight] = 0.0
                self.bias_mask.view(-1)[i - self.n_weight] = 0.0
        self.n_zero += 1

        if check:
            assert self.parameters[i] == 0.0
            assert self.mask[i] == 0.0
            self.check()

        if log:
            logging.info(self)

    def check(self):
        masked = self.parameters.masked_select(self.mask < 1)
        zeros = torch.zeros(self.n_zero).to(masked.device)
        torch.testing.assert_close(masked, zeros)

    @property
    def sparsity(self):
        return self.n_zero / self.n

    def __str__(self):
        return "\n".join([
            #"=" * 80,
            f"{self.name} sparsity: {self.n_zero}/({self.n_weight}+{self.n_bias})={self.sparsity}",
            #"parameters:", f"{self.parameters}", "mask:", f"{self.mask}"
        ])


class OptimalBrainSurgeon(object):
    def __init__(self, model, scopes, fisher_type, check=False):
        self.model = model
        self.scopes = scopes
        offset = 0
        for i, s in enumerate(self.scopes):
            s.l = offset
            s.r = s.l + s.n
            s.index = i
            offset = s.r
        self.n = sum([s.n for s in self.scopes])
        self.n_zero = 0
        self.fisher_type = fisher_type
        self.ifisher = None
        self.ifisher_diag = None
        self.pruned = set()
        self._device = next(self.model.parameters()).device
        self._check = check
        logging.info(self)

    def _get_scope_by_indice(self, i):
        for s in self.scopes:
            if s.l <= i and i < s.r:
                return s
        assert False

    @property
    def parameters(self):
        return to_vector([s.parameters for s in self.scopes])

    @parameters.setter
    def parameters(self, p):
        if isinstance(p, list):
            for s, v in zip(self.scopes, p):
                if v is not None:
                    s.parameters = v
        else:
            for s in self.scopes:
                s.parameters = p[s.l:s.r]

    def parameters_iadd(self, p):
        if isinstance(p, list):
            for s, v in zip(self.scopes, p):
                if v is not None:
                    s.parameters += v
        else:
            for s in self.scopes:
                s.parameters += p[s.l:s.r]

    @property
    def mask(self):
        return to_vector([s.mask for s in self.scopes])

    @property
    def grad(self):
        return to_vector([s.grad for s in self.scopes])

    def _gen_samples(self, loader, n_samples):
        for inputs, targets in loader:
            if n_samples != -1 and len(inputs) > n_samples:
                inputs = inputs[:n_samples]
                targets = targets[:n_samples]
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            yield inputs, targets
            if n_samples != -1:
                n_samples -= len(inputs)
                if n_samples <= 0:
                    break

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        for inputs, targets in self._gen_samples(loader, n_samples):
            fisher_for_cross_entropy(self.model,
                                     fisher_type=self.fisher_type,
                                     fisher_shapes=[self.fisher_shape],
                                     inputs=inputs,
                                     targets=targets,
                                     accumulate=False)

    def _prune_one(self, i, cb):
        assert i not in self.pruned
        self.pruned.add(i)

        scope = self._get_scope_by_indice(i)
        d = self._pruning_direction(i)
        if d is None:
            scope.prune(i - scope.l, check=self._check, log=False)
        elif len(d) == scope.n:
            scope.prune(i - scope.l, d, check=self._check, log=False)
        elif len(d) == self.n:
            self.parameters_iadd(d)
            scope.prune(i - scope.l, check=self._check, log=False)
        else:
            assert False

        self.check()

        self.n_zero += 1
        if cb is not None:
            cb(i)

    def prune(self,
              loader,
              sparsity,
              damping=1e-3,
              n_recompute=1,
              n_recompute_samples=4096,
              cb=None):
        init_n_zero = self.n_zero
        target_n_zero = int(self.n * sparsity)

        if n_recompute == -1:
            n_recompute = target_n_zero - init_n_zero
            schedule = lambda i: self.n_zero + 1
        else:
            schedule = lambda i: polynomial_schedule(
                init_n_zero, target_n_zero, i, n_recompute)

        for i in range(1, n_recompute + 1):
            # We are accumulating fisher across recompute iteration
            # Should we clear fisher at beginning of the iteration?
            self._calc_fisher(loader, n_recompute_samples, damping)
            with torch.no_grad():
                n_pruned = int(schedule(i)) - self.n_zero
                scores = self._get_scores()
                _, indices = torch.sort(scores)
                indices = indices[:n_pruned]
                for j in indices:
                    self._prune_one(j.item(), cb)

        logging.info(self)

    def check(self):
        if self._check:
            mask = torch.ones(self.n)
            mask[list(self.pruned)] = 0.0
            torch.testing.assert_close(mask, self.mask)

    @property
    def sparsity(self):
        return self.n_zero / self.n

    def __str__(self):
        info = [str(s) for s in self.scopes]
        info += [f"Total sparsity: {self.n_zero}/{self.n}={self.sparsity}"]
        return "\n".join(info)


class FullOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type, check):
        super().__init__(model, scopes, fisher_type, check=check)
        self.fisher_shape = SHAPE_FULL

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super()._calc_fisher(loader, n_samples, damping)
        fisher = getattr(self.model, self.fisher_type)
        mask = self.mask
        fisher.data *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
        fisher.update_inv(damping)
        self.ifisher = fisher.inv
        self.ifisher_diag = torch.diagonal(self.ifisher)

    def _get_scores(self):
        scores = self.parameters.pow(2) / self.ifisher_diag
        scores = scores.masked_fill(self.mask == 0.0, float("inf"))
        return scores

    def _pruning_direction(self, i):
        s = self._get_scope_by_indice(i)
        pi = s.parameters[i - s.l]
        return -pi * self.ifisher[:, i] / self.ifisher_diag[i] * self.mask


class LayerOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type, check):
        super().__init__(model, scopes, fisher_type, check=check)
        self.fisher_shape = SHAPE_LAYER_WISE
        self.normalize = False

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super()._calc_fisher(loader, n_samples, damping)
        for s in self.scopes:
            fisher = getattr(s.module, self.fisher_type)
            mask = s.mask
            fisher.data *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
            fisher.update_inv(damping)
            s.ifisher = fisher.inv
            s.ifisher_diag = torch.diagonal(s.ifisher)

    def _get_scores(self):
        flatten_scores = []
        for s in self.scopes:
            scores = s.parameters.pow(2) / s.ifisher_diag
            if self.normalize:
                scores = scores.masked_fill(s.mask == 0.0, 0.0)
                scores /= torch.sum(scores)
            scores = scores.masked_fill(s.mask == 0.0, float("inf"))
            flatten_scores.append(scores)
        return to_vector(flatten_scores)

    def _pruning_direction(self, i):
        s = self._get_scope_by_indice(i)
        pi = s.parameters[i - s.l]
        return -pi * s.ifisher[:, i - s.l] / s.ifisher_diag[i - s.l] * s.mask


class KronOBS(LayerOBS):
    def __init__(self, model, scopes, fisher_type, check):
        super().__init__(model, scopes, fisher_type, check=check)
        self.fisher_shape = SHAPE_KRON
        self.normalize = False
        self.fast_inv = False

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        super(LayerOBS, self)._calc_fisher(loader, n_samples, damping)
        for s in self.scopes:
            fisher = getattr(s.module, self.fisher_type).kron
            if self.fast_inv:
                # This method is wrong
                fisher.update_inv(damping)
                fisher.inv = torch.kron(fisher.A_inv, fisher.B_inv)
                mask = s.mask
                fisher.inv *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
            else:
                f = torch.kron(fisher.A, fisher.B)
                mask = s.mask
                f *= mask.reshape([1, -1]) * mask.reshape([-1, 1])
                fisher.inv = cholesky_inv(add_value_to_diagonal(f, damping))
            s.ifisher = fisher.inv
            s.ifisher_diag = torch.diagonal(s.ifisher)


class NoneOBS(OptimalBrainSurgeon):
    def __init__(self, model, scopes, fisher_type, check):
        super().__init__(model, scopes, fisher_type, check=check)
        self.fisher_shape = "none"

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        return None

    def _get_scores(self):
        return torch.abs(self.parameters).masked_fill(self.mask == 0.0,
                                                      float("inf"))

    def _pruning_direction(self, i):
        return None


class FullWoodOBS(FullOBS):
    def __init__(self, model, scopes, check):
        super().__init__(model, scopes, "wood_fisher", check=check)

    def _calc_fisher(self, loader, n_samples, damping=1e-3):
        N = None
        fisher_inv = torch.eye(self.n) / (damping**2)

        for inputs, targets in self._gen_samples(loader, n_samples):
            if N is None:
                N = math.ceil(n_samples / len(inputs))
            nn.CrossEntropyLoss()(self.model(inputs), targets).backward()
            with torch.no_grad():
                g = self.grad * self.mask
                fg = fisher_inv @ g
                fisher_inv -= torch.outer(fg, fg) / (N + g.T @ fg)

        self.ifisher = fisher_inv
        self.ifisher_diag = torch.diagonal(self.ifisher)


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


def get_global_prnning_scope(model):
    modules = list_model(model, condition=lambda x: hasattr(x, "weight"))
    return [Scope(k, v) for k, v in modules.items()]


def create_obs(model, scopes, args):
    if args.fisher_type == "wood_fisher":
        obs = {
            SHAPE_FULL: FullWoodOBS,
        }[args.fisher_shape](model, scopes, args.check)
    else:
        obs = {
            SHAPE_FULL: FullOBS,
            SHAPE_LAYER_WISE: LayerOBS,
            SHAPE_KRON: KronOBS,
            "none": NoneOBS
        }[args.fisher_shape](model, scopes, args.fisher_type, args.check)
        if args.fisher_shape in [SHAPE_KRON, SHAPE_LAYER_WISE]:
            obs.normalize = args.layer_normalize
        if args.fisher_shape == SHAPE_KRON:
            obs.fast_inv = args.kfac_fast_inv
    return obs


def one_shot_pruning(model, loaders, criterion, args):
    scopes = get_global_prnning_scope(model)
    obs = create_obs(model, scopes, args)

    def _cb(i):
        n_zero = obs.n_zero
        if n_zero > 1:
            s = obs._get_scope_by_indice(i)
            wandb.log({"p_dist": i, "l_dist": s.index}, step=n_zero)
        if n_zero % int(obs.n * args.test_intvl) == 0 or n_zero == obs.n:
            acc = test(model,
                       loaders["test"],
                       criterion,
                       args.device,
                       prefix=f"[{n_zero:10}] ")
            #torch.save(model.state_dict(), f"{args.log}/{n_zero}-{acc}")
            wandb.log({"acc": acc}, step=n_zero)

    _cb(None)
    obs.prune(loaders["train"], args.sparsity, args.damping, args.n_recompute,
              args.n_recompute_samples, _cb)


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
        torch.load(f"{args.model_dir}/best", map_location=args.device))

    if args.pruning_strategy == "oneshot":
        one_shot_pruning(model, loaders, criterion, args)


if __name__ == "__main__":
    main()
