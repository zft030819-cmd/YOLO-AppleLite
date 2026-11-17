"""Historical EMA group-unrolling function, copied without mathematical changes."""
import torch

def ema_forward(self, x):
    b, c, h, w = x.shape
    outputs = []
    for group_x in x.chunk(self.groups, dim=1):
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b, 1, h, w)
        outputs.append(group_x * weights.sigmoid())
    return torch.cat(outputs, dim=1)
