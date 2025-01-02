import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        # self.fc = nn.Linear(in_features=28 * 28,
        #                     out_features=10)
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=10,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.fc = nn.Linear(in_features=28 * 28 * 10 // (2 * 2),
                            out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class MyNet(nn.Module):
    """
    Experiment with all possible settings mentioned in the CW page
    """
    def __init__(self):
        super(MyNet, self).__init__()
        self.Layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                                              stride=1, padding=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.Layer2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                                              stride=1, padding=2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.Layer3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                              stride=1, padding=2),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.Layer4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3,
                                              stride=1, padding=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))

        self.Linear1 = nn.Linear(in_features=576, out_features=128)
        self.Linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        output = F.log_softmax(x, dim=1)
        return output


def classify(model, x):
    """
    :param model:    network model object
    :param x:        (batch_size, 1, 28, 28) tensor - batch of images to classify

    :return labels:  (batch_size, ) torch tensor with class labels
    """
    labels = torch.argmax(model(x), dim=1)
    return labels


def visualize_data(data, legend=None, title=None, xlabel=None, ylabel=None,
                   save_filepath=None, show=True, hline=None, hlinelabel=None):
    """
    visualize_data(data, legend, title, xlabel, ylabel, save_filepath, show)

    :param data:            list of 1D input data
    :param legend:          list of data labels (same size as data, optional)
    :param title:           figure title, string (optional)
    :param xlabel:          x-axis label, string (optional)
    :param ylabel:          y-axis label, string (optional)
    :param save_filepath:   name and path for saving (optional)
    :param show:            showing figure, boolean
    :param hline:           show horizontal line
    :return:
    """

    if title:
        plt.title(title)
    for d in data:
        plt.plot(np.arange(len(d)), d)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)
    if hline:
        plt.axhline(linewidth=2, y=hline, color='red', ls='--')
    if hlinelabel:
        plt.text(0, hline, hlinelabel, fontsize=10, va='bottom', ha='left', c='red')
    if save_filepath:
        plt.savefig(save_filepath)
    if show:
        plt.show()


def visualize_xy(data, legend=None, title=None, xlabel=None, ylabel=None, save_filepath=None, show=True, **kwargs):
    """
    visualize_data(data, legend, title, xlabel, ylabel, save_filepath, show)

    :param data:            list of 2D input data tuples
                            i.e.: data = [(xdata, ydata)]
    :param legend:          list of data labels (same size as data, optional)
    :param title:           figure title, string (optional)
    :param xlabel:          x-axis label, string (optional)
    :param ylabel:          y-axis label, string (optional)
    :param save_filepath:   name and path for saving (optional)
    :param show:            showing figure, boolean
    :return:
    """
    axis = kwargs.get('axis', None)
    grid = kwargs.get('grid', None)
    linestyle = kwargs.get('linestyle', '-')

    if title:
        plt.title(title)
    for d in data:
        plt.plot(d[0], d[1], linestyle)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)

    if axis:
        plt.axis(axis)
    if grid:
        plt.grid()

    if save_filepath:
        plt.savefig(save_filepath)
    if show:
        plt.show()


def main():
    batch_size = 16
    learning_rate = 0.01
    epochs = 30

    dataset = datasets.FashionMNIST('data', train=True, download=True,
                                    transform=transforms.ToTensor())

    trn_size = int(0.09 * len(dataset))
    val_size = int(0.01 * len(dataset))
    add_size = len(dataset) - trn_size - val_size  # you don't need ADDitional dataset to pass

    trn_dataset, val_dataset, add_dataset = torch.utils.data.random_split(dataset, [trn_size,
                                                                                    val_size,
                                                                                    add_size])
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

    device = torch.device("cpu")
    model = MyNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    validation_accuracies = []
    for epoch in range(1, epochs + 1):
        # training
        model.train()
        for i_batch, (x, y) in enumerate(trn_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            net_output = model(x)
            loss = F.nll_loss(net_output, y)
            loss.backward()
            optimizer.step()

            if i_batch % 100 == 0:
                print('[TRN] Train epoch: {}, batch: {}\tLoss: {:.4f}'.format(
                    epoch, i_batch, loss.item()))

        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                prediction = classify(model, x)
                correct += prediction.eq(y).sum().item()
        val_accuracy = correct / len(val_loader.dataset)
        validation_accuracies.append(100. * val_accuracy)
        print('[VAL] Validation accuracy: {:.2f}%'.format(100 * val_accuracy))

    print('Training completed, final accuracy: {:.2f}%'.format(100 * val_accuracy))
    torch.save(model.state_dict(), "model.pt")

    visualize_data([validation_accuracies], legend=['validation_accuracy'], xlabel='epoch', ylabel='%',
                   save_filepath='pytorch_cnn_training.png', hline=75.0, hlinelabel='Accuracy threshold (on test set)')


if __name__ == '__main__':
    main()
