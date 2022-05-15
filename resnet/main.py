import os
import sys
sys.path.append(os.pardir)
from conv.main import get_dataloader, get_dataset, train, get_device, check_acc, check_history, check_model, check_my_image
import torch.nn as nn
import torch.nn.functional as F
import torch


def main() -> None:
    # train_loader, test_loader = get_dataloader(get_dataset)
    # device = get_device()
    check_model()

    raise UnimplementedError


class UnimplementedError(Exception):
    pass

if __name__ == "__main__":
    main()
