import torch.nn as nn

class TimeIndependentValue(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, y=None):
        if y is not None:
            return self.net(x, y)
        else:
            return self.net(x)

    def load_pretrained(self, ckpt):
        self.net.load_pretrained(ckpt)