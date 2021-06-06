import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

import config as cfg


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Recurrent(nn.Module):
    def __init__(
        self,
        layer_num: int,
        state_shape,
        action_shape,
        device = "cpu") -> None:
        super().__init__()
        self.device = device
       
        c, h, w = cfg.state_shape
        
        self.extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten())
        with torch.no_grad():
            self.output_dim = np.prod(
                self.extractor(torch.zeros(1, c, h, w)).shape[1:])

        self.lstm = nn.LSTM(
            input_size=self.output_dim,
            hidden_size=self.output_dim,
            num_layers=layer_num,
            batch_first=True)
       
    def forward(self, s, state = None, info = {}):
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)

        bsz = s.size(0)
        if len(s.shape) == 3:
            s = s.unsqueeze(1)
        elif len(s.shape) == 4:
            s = s.view(-1, *cfg.state_shape)
        # if s.ndim == 5:
        #     s = s.view(-1, 3, *cfg.image_size)

        s = self.extractor(s)
        if s.size(0) != bsz:
            s = s.view(bsz, cfg.n_stack, -1)
        if len(s.shape) == 2:
            s = s.unsqueeze(1)

        self.lstm.flatten_parameters()
        if state is None:
            s, (h, c) = self.lstm(s)
        else:
            s, (h, c) = self.lstm(s, (state["h"].transpose(0, 1).contiguous(),
                                    state["c"].transpose(0, 1).contiguous()))
        s = s[:, -1]
        return s, {"h": h.transpose(0, 1).detach(),
                   "c": c.transpose(0, 1).detach()}

