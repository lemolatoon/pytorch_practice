from __future__ import annotations
import matplotlib
import numpy as np
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import Tensor
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset
from tqdm import tqdm


def data_process() -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
    data_root: str = "./data"

    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),  # PIL Image or ndarray to Tensor (?) : Unknown -> Tensor
        transforms.Normalize(0.5, 0.5),  # Normalize : Tensor -> Tensor
        # Apply lambda : Tensor -> Tensor
        # view(-1, **args), reshape automatically
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    # do experiment compose `Callables`
    # data: torch.Tensor = datasets.MNIST(root=data_root, download=True).data
    # print(data.shape)
    # print(type(data))
    # # data: torch.Tensor = transforms.ToTensor()(data)
    # data: torch.Tensor = transforms.Lambda(lambda x: 1.0 * x)(data)
    # print(type(data))
    # data: torch.Tensor = transforms.Normalize(0.5, 0.5)(data)
    # print(type(data))
    # print(data.shape)
    # data: torch.Tensor = transforms.Lambda(lambda x: x.view(-1))(data)
    # print(type(data))
    # print(data.shape)

    train_set: datasets.VisionDataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    print(train_set)

    test_set: datasets.VisionDataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    print(test_set)
    return (train_set, test_set)


def check_data(train_set: datasets.VisionDataset,
               test_set: datasets.VisionDataset) -> None:
    image: Tensor
    label: int
    image, label = train_set[0]  # (Tensor, int)
    print(f"number of train_set: {len(train_set)}")
    print(f"image type: {type(image)}")
    print(f"image shape: {image.shape}")

    image, label = test_set[0]
    print(f"[test] number of test_set: {len(test_set)}")
    print(f"[test] image type: {type(image)}")
    print(f"[test] image shape: {image.shape}")

    print(f"image value min: {image.min()}")  # なぜ image.data.min() ?
    print(f"image value max: {image.max()}")  # なぜ image.data.min() ?

    show_img(image, label)


def show_img(image: Tensor, label: int,
             figsize: Tuple[int, int] = (2, 2)) -> None:
    plt.figure(figsize=figsize)
    plt.title(f"{label}")

    length: int = int(math.sqrt(image.shape[0]))
    image_square: Tensor = image.reshape((length, length))
    print(image_square.shape)
    plt.imshow(image_square, cmap="gray_r")
    plt.savefig("./image/num_image.png")


def process_mini_batch(train_set: Dataset[Tensor],
                       test_set: Dataset[Tensor],
                       batch_size: int = 500) -> Tuple[DataLoader[Tensor],
                                                       DataLoader[Tensor]]:
    # Expression of type "Tensor" cannot be assigned to declared type "int"
    # "Tensor" is incompatible with "int
    # generic argument means return type of __getitem__(self, index) -> T_co:
    # test: int = train_set[0]

    train_loader: DataLoader[Tensor] = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    print(type(train_loader))

    test_loader: DataLoader[Tensor] = DataLoader(
        test_set, batch_size=batch_size, shuffle=True)

    print(f"train_loader len: {len(train_loader)}")
    print(f"test_loader len: {len(test_loader)}")
    print(f"cf.) 60000 / 500 == 120, ", end="")
    print(f"10000 / 500 == 20")

    image: Tensor
    label: Tensor
    image, label = next(iter(train_loader))
    print(f"image.shape: {image.shape}")
    print(f"label.shape: {label.shape}")

    return train_loader, test_loader


def new_optimizer(lr: float, net: nn.Module) -> optim.Optimizer:
    return optim.SGD(net.parameters(), lr)


def train(train_loader: DataLoader[Tensor],
          test_loader: DataLoader[Tensor],
          optimizer: optim.Optimizer,
          criterion: nn.CrossEntropyLoss,
          net: nn.Module,
          device: torch.device,
          num_epoch: int = 100) -> np.ndarray[Any,
                                              np.dtype[Any]]:

    batch_size: int = next(iter(train_loader))[0].shape[0]
    history: np.ndarray[Any, np.dtype[Any]] = np.zeros(5)

    for epoch in range(num_epoch):
        train_acc: float
        train_loss: float
        val_acc: float
        val_loss: float
        n_train: int
        n_test: int
        train_acc, train_loss = 0, 0
        val_acc, val_loss = 0, 0
        n_train, n_test = 0, 0

        inputs: Tensor
        labels: Tensor
        for inputs, labels in tqdm(test_loader):
            n_train += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # fill 0 in grads which was retained by optimizer
            optimizer.zero_grad()
            # forward
            outputs: Tensor = net(inputs)
            loss: Tensor = criterion(outputs, labels)

            # backward (differential)
            loss.backward()

            # update params by rules of optimizer.
            # Note: optimizer has ptrs to params of net
            optimizer.step()

            # Returns a namedtuple (values, indices) where values is the maximum value
            # of each row of the input tensor in the given dimension dim. And indices
            # is the index location of each maximum value found (argmax).
            # eg.) outputs = Tensor ([[1, 2, 3, 4, 5, 6, 7, 8, 9, 100]])
            #   -> torch.max(outputs, dim=1) == (100, 9)
            predicted = torch.max(outputs, dim=1)[1]  # take index

            train_loss += loss.item()
            train_acc += (predicted == labels).sum().item()

        # check loss and acc on test data
        inputs_test: Tensor
        labels_test: Tensor
        for inputs_test, labels_test in test_loader:
            n_test += len(labels_test)

            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            outputs_test: Tensor = net(inputs_test)
            loss_test: Tensor = criterion(outputs_test, labels_test)
            predicted_test = torch.max(outputs_test, dim=1)[1]

            val_loss += loss_test.item()
            val_acc += (predicted_test == labels_test).sum().item()

        # calc average of each value
        train_acc = train_acc / n_train
        val_acc = val_acc / n_test
        train_loss = train_loss * batch_size / n_train  # ?
        val_loss = val_loss * batch_size / n_test

        print(
            f"Epoch [{epoch + 1}/{num_epoch}], loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_acc:{val_acc:.5f}"
        )
        items: np.ndarray[Any, np.dtype[Any]] = np.array(
            [epoch + 1, train_loss, train_acc, val_loss, val_acc]
        )
        history = np.vstack((history, items))

    return history


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    train_set: datasets.VisionDataset
    test_set: datasets.VisionDataset
    train_set, test_set = data_process()

    check_data(train_set, test_set)

    train_loader: DataLoader[Tensor]
    test_loader: DataLoader[Tensor]
    train_loader, test_loader = process_mini_batch(train_set, test_set)

    n_input: int = 784
    n_output: int = 10
    n_hidden: int = 128

    device: torch.device = get_device()
    net: Net = Net(n_input, n_output, n_hidden).to(device)

    # optimizer has ptr to all of parameters of the net
    optimizer = new_optimizer(0.01, net)

    criterion = nn.CrossEntropyLoss()
    print(device)
    history = train(
        train_loader,
        test_loader,
        optimizer,
        criterion,
        net,
        device)
    print(history)
    print(history.shape)
    net.save("./params/param.pt")


class Net(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int):
        super().__init__()
        self.l1: nn.Linear = nn.Linear(n_input, n_hidden)
        self.l2: nn.Linear = nn.Linear(n_hidden, n_output)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self: Net, x: Tensor) -> Tensor:
        x1: Tensor = self.l1(x)
        x2: Tensor = self.relu(x1)
        x3: Tensor = self.l2(x2)
        return x3

    def save(self: Net, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self: Net, path: str):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    main()
