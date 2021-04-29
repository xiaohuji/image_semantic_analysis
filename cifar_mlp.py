import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torch.utils.data
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from torch.autograd import Variable
from torchvision.transforms import transforms


def sklearn_mlp(x_train, x_test, y_train, y_test):
    # split = len(x_train)*2/5
    x_train_data, y_train_data = x_train, y_train
    # x_valid_data, y_valid_data = x_train[split:], y_train[split:]
    x_test_data, y_test_data = x_test, y_test
    # 合并训练集,验证集
    # param_grid = {"hidden_layer_sizes": [(100,), (100, 30)], "solver": ['adam', 'sgd', 'lbfgs'],
    #               "max_iter": [20], "verbose": [True], "early_stopping": [True]}
    param_grid = {"hidden_layer_sizes": [(100, 20)], "solver": ['adam', 'sgd', 'lbfgs'],
                  "max_iter": [20], "verbose": [True], "early_stopping": [True]}
    mlp = MLPClassifier()
    clf = GridSearchCV(mlp, param_grid, n_jobs=-1)
    print(mlp.n_layers_)
    clf.fit(x_train_data, y_train_data)
    print(clf.score(x_test_data, y_test_data))
    print(clf.get_params().keys())
    print(clf.best_params_)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3072, 120)
        self.d1 = nn.Dropout(p=0.5)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.d1(x)
        x = self.r1(x)
        x = self.fc2(x)
        return x


def torch_mlp(train_n, test_n):
    transform = transforms.Compose([
        # 把灰度范围从0 - 255变换到0 - 1之间
        transforms.ToTensor(),
        # 标准化变换到-1 - 1之间
        # 数据如果分布在(0, 1)之间，可能实际的bias，就是神经网络的输入b会比较大，而模型初始化时b = 0
        # 这样会导致神经网络收敛比较慢，经过Normalize后，可以加快模型的收敛速度。
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 创建测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset[train_n], batch_size=1, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset[test_n], batch_size=1, shuffle=False)

    net = Net()
    # 权值初始化
    for n in net.modules():
        if isinstance(n, nn.Linear):
            nn.init.xavier_uniform_(n.weight)
            nn.init.kaiming_normal(n.weight.data)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    for epoch in range(1):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Finished Training")

    print("Beginning Testing")
    correct = 0
    total = 0

    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        predicted = torch.max(outputs, 1)[1].data.numpy()
        total += labels.size(0)
        correct += (predicted == labels.data.numpy()).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False)
    cifar_train = trainset.data.reshape((50000, 3072))
    cifar_train_label = trainset.targets
    cifar_test = testset.data.reshape((10000, 3072))
    cifar_test_label = testset.targets
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # sklearn发现隐层不好初始化
    # t1 = time.clock()
    # sklearn_mlp(cifar_train[:100], cifar_test[:20], cifar_train_label[:100], cifar_test_label[:20])
    # t2 = time.clock()
    # print('svm_c_time:', t2 - t1)
    # ---------------------------------------------------------------------------------
    # 改用torch
    t1 = time.clock()
    torch_mlp(train_n=10, test_n=10)
    t2 = time.clock()
    print('svm_c_time:', t2 - t1)
