"""
Pytorch implementation of the model. Training is working as expected.
"""

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("data/hand_gestures.csv")
X_df = df.drop(columns=['sample_id', 'label'])
y_df = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.2, stratify=y_df, random_state=42
)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 128
        self.lin1 = nn.Linear(42, self.hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.hidden, self.hidden)
        self.lin3 = nn.Linear(self.hidden, 3)
    
    def forward(self, x):
        return self.lin3(self.relu(self.lin2(self.relu(self.lin1(x)))))

model = Model()
# a = torch.randn(5, 42)
# print(model(a).shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs+1):
    print(f"===============Epoch: {epoch}===============")
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = model(xb)
        loss_val = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()

    print(f"\n\nepoch {epoch} | loss {total_loss / len(train_loader):.4f}\n\n\n\n")


print("\nEvaluating on test data...")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    preds = model(X_test)
    predicted_labels = torch.argmax(preds, dim=1)

    correct += (predicted_labels == y_test).sum().item()
    total += y_test.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), "gesture_model.pth")

print("Saved Model")
