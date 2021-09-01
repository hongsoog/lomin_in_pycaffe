# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # weight/bias is loaded from trained model
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))

        # running_mean/running var is always constant array of zeros and ones
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary

        # x.dtype == torch.float32 (confirmed with debug)
        if x.dtype == torch.float16:  # following part never executed
            #print(f"FrozenBatchNorm2d::forward() x.dtype : {x.dtype}")
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        #print(f"FBN weight: of shape {self.weight.shape}\n{self.weight}\n")
        #print(f"FBN bias: of shape {self.bias.shape}\n{self.bias}\n")
        #print(f"FBN running_mean: of shape {self.running_mean.shape}\n{self.running_mean}\n")
        #print(f"FBN running_var: of shape {self.running_var.shape}\n{self.running_var}\n")
        # in fact,
        # running_var is list of 1s of size given by n
        # running_mean is list of 0s of size given by n
        scale = self.weight * self.running_var.rsqrt()
        # finally scale = self.weight since runnin_var is array of all element is 1 and rsqrt(1)=1
        #print(f"FBN scale: of shape {scale.shape}\n{scale}\n")

        bias = self.bias - self.running_mean * scale
        # bias = self.bais since rnumming_mean is array of all zero elements,
        #print(f"FBN bias: of shape {bias.shape}\n{bias}\n")

        scale = scale.reshape(1, -1, 1, 1)    # just reshape of weight
        #print(f"FBN scale after reshape: of shape {scale.shape}\n{scale}\n")
        bias = bias.reshape(1, -1, 1, 1)      # just reshape of bias
        #print(f"FBN bias after reshape: of shape {bias.shape}\n{bias}\n")

        # finally return x * scale + bias
        # i.e x * weight + bias
        return x * scale + bias    # i.e., x * weight + bias