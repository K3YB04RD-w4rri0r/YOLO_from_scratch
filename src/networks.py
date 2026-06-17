import torch
import torch.nn as nn



class YOLO(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride= 2)
        self.lrelu = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=64, kernel_size=7, padding=3, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=192, kernel_size=3, padding=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1),
        )

        self.block2 = nn.Sequential(
            
            nn.Conv2d(in_channels = 512, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels = 512, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1),


            nn.Conv2d(in_channels = 512, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1),


            nn.Conv2d(in_channels = 512, out_channels=256, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1),

            
            nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )


        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels=512, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels = 1024, out_channels=512, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),          
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels = 1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(4096, 7 * 7 * 30)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.maxpool(x)

        x = self.block2(x)
        x = self.maxpool(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.head(x)

        return x








