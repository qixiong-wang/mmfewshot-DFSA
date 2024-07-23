import time
import torch.nn as nn
import torch

Linear = nn.Linear(1024, 1024).cuda()
Conv2d = nn.Conv2d(1024, 1024, 1, bias=False).cuda()
s = time.time()

x = torch.rand((4096 * 15, 1024)).cuda()
s = time.time()
for _ in range(3):
    x = Linear(x)
e = time.time()
print(e - s)