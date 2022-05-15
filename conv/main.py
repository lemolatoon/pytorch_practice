from __future__ import annotations
from PIL import Image
import torch
from torchvision import datasets
from torchvision import transforms
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import matplotlib.pyplot as plt
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import Optional
import sys


def main() -> None:
    (train_set, test_set) = get_dataset()
    check_data(train_set)
    train_loader, test_loader = get_dataloader(train_set, test_set)
    print(len(train_loader))
    print(len(test_loader))

    device = get_device()
    print(device)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    history = train(train_loader, test_loader, optimizer,
                    criterion, net, device)
    print(history.shape)
    torch.save(net.state_dict(), "param.pt")
    pd.to_pickle(history, "history.pkl")


Loader = DataLoader[Tuple[Tensor, Tensor]]


def train(train_loader: Loader, test_loader: Loader, optimizer: optim.Adam, criterion: nn.CrossEntropyLoss, net: Net, device, num_epoch: int = 20) -> np.ndarray:
    batch_size: int = iter(train_loader).next()[0].shape[0]
    history = np.zeros(5)
    for epoch in range(num_epoch):
        train_acc, train_loss = 0.0, 0.0  # train data
        val_acc, val_loss = 0.0, 0.0  # validation data
        num_train, num_test = 0, 0  # the number of data

        inputs: Tensor
        labels: Tensor
        for inputs, labels in tqdm(train_loader):
            num_train += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs).to(device)
            loss: Tensor = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            _, predicted = torch.max(outputs, 1)

            train_loss += loss.item()
            train_acc += (predicted == labels).sum().item()

        inputs_test: Tensor
        labels_test: Tensor
        for inputs_test, labels_test in test_loader:
            num_test += len(labels_test)
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            outputs_test = net(inputs_test)
            loss_test: Tensor = criterion(outputs_test, labels_test)
            _, predicted_test = torch.max(outputs_test, 1)

            val_loss += loss_test.item()
            val_acc += (predicted_test == labels_test).sum().item()

        train_acc = train_acc / num_train
        val_acc = val_acc / num_test
        train_loss = train_loss * batch_size / num_train
        val_loss = val_loss * batch_size / num_test

        print(
            f"Epoch [{epoch + 1}/{num_epoch}], loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}"
        )
        items = np.array(
            [epoch + 1, train_loss, train_acc, val_loss, val_acc]
        )
        history = np.vstack((history, items))
    # [[index, train_loss, train_acc, val_loss, val_acc]], len(history) == num_epoch + 1, len(history[0]) == 5
    return history


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.drop2 = nn.Dropout2d(p=0.25)

        self.fc1 = nn.Linear(in_features=64*32*32, out_features=512)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drop1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.drop2(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)

        return x


def get_dataset() -> Tuple[Dataset[Tuple[Tensor, int]],
                           Dataset[Tuple[Tensor, int]]]:
    data_root = "./data"

    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    train_set: Dataset[Tuple[Tensor, int]] = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )

    test_set = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform
    )

    return (train_set, test_set)


def get_dataloader(train_set: Dataset[Tuple[Tensor, int]], test_set: Dataset[Tuple[Tensor, int]],
                   batch_size=128) -> Tuple[DataLoader[Tuple[Tensor, Tensor]], DataLoader[Tuple[Tensor, Tensor]]]:
    train_loader: DataLoader[Tuple[Tensor, Tensor]] = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)  # type: ignore

    test_loader: DataLoader[Tuple[Tensor, Tensor]] = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)  # type: ignore

    return (train_loader, test_loader)


def check_data(data: Dataset[Tuple[Tensor, int]]):
    for i in range(10):
        image, _ = data[i]
        plt.subplot(2, 5, i + 1)
        # (rgb, height, length) -> (length, height, rgb) -> (height, length, rgb)
        plt.imshow(image.transpose(0, 2).transpose(0, 1))
    plt.savefig("pic.png")


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_history():
    # [[index, train_loss, train_acc, val_loss, val_acc]], len(history) == num_epoch + 1, len(history[0]) == 5
    history: np.ndarray = pd.read_pickle("history.pkl")
    plt.plot(history[1:, 0], history[1:, 1], "b", label="train")
    plt.plot(history[1:, 0], history[1:, 3], "k", label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    plt.plot(history[:, 0], history[:, 2], "b", label="train")
    plt.plot(history[:, 0], history[:, 4], "k", label="test")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("accuracy")
    plt.legend()
    plt.savefig("acc_curve.png")


def check_model(test_loader: Optional[Loader] = None, net: Optional[Net] = None, index: int = 0):
    device = get_device()
    if net is None:
        net = load_model(device)
    if test_loader is None:
        _, test_loader = get_dataloader(*get_dataset())
    images, labels = list(iter(test_loader))[index]
    print(images.shape)
    images: Tensor = images.to(device)
    labels: Tensor = labels.to(device)
    outputs: Tensor = net(images)
    predicteds: Tensor = torch.max(outputs, 1)[1]

    acc = (labels == predicteds).sum().item() / len(labels)
    print(acc)
    if acc == 1:
        check_model(test_loader, net, index + 1)
        sys.exit(0)

    plt.figure(figsize=(30, 24))
    acc: float = 0
    for i in range(50):
        ax = plt.subplot(5, 10, i + 1)

        image = images[i].cpu()
        label = labels[i].cpu()
        predicted = predicteds[i].cpu()
        if (predicted == label):
            c = "k"
            acc += 1
        else:
            c = "b"

        plt.imshow(image.reshape(3, 32, 32).transpose(0, 2).transpose(0, 1))
        label_name = ["airplane", "automobile", "bird", "cat",
                      "deer", "dog", "frog", "horse", "ship", "truck"]
        ax.set_title(f"{label_name[label]}:{label_name[predicted]}", c=c)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    acc = acc / 50
    print(f"acc: {acc}")
    plt.savefig("pred.png")


def check_my_image(path="my.png"):
    img = Image.open(path)
    img = img.resize((32, 32))
    print(np.array(img).shape)
    img = transforms.Normalize(0.5, 0.5)(transforms.ToTensor()(img))
    input = Tensor(img)
    device = get_device()
    print(input.shape)
    # (height, length, rgb) -> (rgb, height, length)
    # input = input.transpose(0, 2).transpose(1, 2)
    input = input[:3, ]  # for image that have 4 dim color
    plt.imshow(input.transpose(0, 2).transpose(0, 1))
    plt.savefig("conversion.png")
    print(input.shape)
    input = input.reshape(1, 3, 32, 32).to(device)
    print(input.shape)
    label_name = ["airplane", "automobile", "bird", "cat",
                  "deer", "dog", "frog", "horse", "ship", "truck"]
    net = load_model(device)
    output = net(input)
    pred = torch.max(output, 1)
    print(pred)
    print(label_name[pred[1]])


def load_model(device: torch.device) -> Net:
    net = Net().to(device)
    net.load_state_dict(torch.load("param.pt"))
    net.eval()
    return net


if __name__ == "__main__":

    # predict image in conv/`path`
    # check_my_image()

    # check acc in eval mode
    # check_model(my.jpg)

    # create png which was generated while train phase
    # check_history()

    # train
    main()
