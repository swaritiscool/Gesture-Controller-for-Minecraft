"""
Pytorch implementation of the model. This should be way more stable and easier to feed into via pandas which I will introduce in train.py so that I have backups and can refer back here.
"""

import torch
import torch.nn as nn

# will fill with Pandas later
X = torch.Tensor()
y = torch.Tensor()

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 128
        self.lin1 = nn.Linear(42, self.hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.hidden, 3)
    
    def forward(self, x):
        return self.lin2(self.relu(self.lin1(x)))

model = Model()
# a = torch.randn(5, 42)
# print(model(a).shape)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

# will implement batches later as well
model.train()
for epoch in range(0, epochs):
    A = model(X)
    loss = loss(A, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

