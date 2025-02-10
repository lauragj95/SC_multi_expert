
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=16, tokenH=16, num_labels=5):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        # self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))
        self.upsampling = torch.nn.Upsample(scale_factor=4)
        
        self.conv1 = torch.nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(24, 12, kernel_size=1)
        self.conv5 = torch.nn.Conv2d(12, num_labels, kernel_size=3, stride=1, padding=1)
        self.norm4 = torch.nn.BatchNorm2d(12)
        self.norm1 = torch.nn.BatchNorm2d(96)
        self.norm2 = torch.nn.BatchNorm2d(48)
        self.norm3 = torch.nn.BatchNorm2d(24)


    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)
        
        out4 = self.upsampling(embeddings)
        out5=self.norm1(F.leaky_relu(self.conv1(out4)))
        out6= self.upsampling(out5)
        out7=self.norm2(F.leaky_relu(self.conv2(out6)))
        out8= self.upsampling(out7)
        out9=self.norm3(F.leaky_relu(self.conv3(out8)))
        out10= self.upsampling(out9)
        out11=self.norm4(F.leaky_relu(self.conv4(out10)))
        out12 = self.conv5(out11)

        return out12