import argparse
import glob

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="pruning")
    parser.add_argument("--log", type=str, default="./.log")

    parser.add_argument("--model",
                        type=str,
                        default="toy1",
                        choices=["resnet18", "toy1", "toy2"])

    parser.add_argument("--dataset",
                        type=str,
                        default="MNIST",
                        choices=["MNIST", "CIFAR10"])

    args = parser.parse_args()
    args.log = f"{args.log}/{args.model}-{args.dataset}"
    return args

def main():
    args = parse_args()
    data = {}
    for model in glob.glob(f"{args.log}/pruned/*"):
        parts = model.split("-")
        data[int(parts[-2])] = float(parts[-1])
    x = []
    y = []
    for k, v in sorted(data.items()):
        x.append(k/len(data))
        y.append(v)
    plt.plot(x, y, label=f"{args.model}-{args.dataset}")
    plt.xlabel("sparsity")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig(f"{args.log}/pruned.svg")
    #plt.show()

if __name__ == "__main__":
    main()
