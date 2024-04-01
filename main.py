import torch.backends.mps

from train import Training


def main():
    Training().train()


if __name__ == '__main__':
    main()
