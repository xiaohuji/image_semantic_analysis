import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from torchvision.transforms import transforms


class AlexNet(nn.Module):  # 定义网络，推荐使用Sequential，结构清晰
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input_size = 227*227*3
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)  # output_size = 27*27*96
        )
        self.conv2 = torch.nn.Sequential(  # input_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)  # output_size = 13*13*256
        )
        self.conv3 = torch.nn.Sequential(  # input_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv4 = torch.nn.Sequential(  # input_size = 13*13*384
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv5 = torch.nn.Sequential(  # input_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)  # output_size = 6*6*256
        )

        # 网络前向传播过程
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 50)
        )

    def forward(self, x):  # 正向传播过程
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out

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

        net = AlexNet()
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