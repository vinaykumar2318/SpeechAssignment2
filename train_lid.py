import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

from extract_features import load_dataset
from lid_model import LIDModel

X_hi, y_hi = load_dataset("data/commonvoice_hi", 0)

X_en, y_en = load_dataset("data/commonvoice_en", 1)

X = np.array(X_hi + X_en)
y = np.array(y_hi + y_en)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = LIDModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

with torch.no_grad():
    preds = model(X_test)
    predicted = torch.argmax(preds, dim=1)

f1 = f1_score(y_test, predicted)
print("F1 Score:", f1)


torch.save(model.state_dict(), "lid_model.pth")
print("Model saved successfully!")