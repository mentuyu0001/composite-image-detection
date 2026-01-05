import torch
import torch.nn.functional as f

class MyNet(torch.nn.Module):
    def __init__(self, input_dim=(3, 224, 224)):
        super(MyNet, self).__init__()

        hidden_size = 100
        filter_size = 5

        self.conv1 = torch.nn.Conv2d(in_channels=input_dim[0], out_channels=3, kernel_size=filter_size, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=filter_size, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=filter_size, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=filter_size, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=filter_size, stride=1, padding=2)

        self.relu = torch.nn.ReLU()
        pooling_size = 2
        self.pool = torch.nn.MaxPool2d(pooling_size, stride=pooling_size)

        # 最終の特徴マップサイズに合わせてfc1を修正します。
        self.fc1 = torch.nn.Linear(3 * 7 * 7, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 2)  # ノード2つに出力

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        # 修正後のサイズに合わせます。
        x = x.view(-1, 3 * 7 * 7)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x