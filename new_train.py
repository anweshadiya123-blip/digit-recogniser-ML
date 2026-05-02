import matplotlib
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
# 📦 Load data
train = pd.read_csv("dataset/mnist_train.csv")

X = train.iloc[:, 1:].values / 255.0
y = train.iloc[:, 0].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
# 🧠 Model
def get_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

model = get_model()

# ⚙️ Training setup
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🚀 DataLoader (THE BIG UPGRADE)
train_loader = DataLoader(
    TensorDataset(X, y),
    batch_size=64,
    shuffle=True
)

# 🧠 Training
best_loss = float('inf')
if __name__ == "__main__":
    print(X.shape, y.shape)
    for epoch in range(10):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            output = model(xb)
            loss = loss_fn(output, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss}")

    # 💾 Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")

# 🧪 Accuracy check (fast now)
    model.eval()
    model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in train_loader:
        output = model(xb)
        preds = output.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print("Accuracy:", acc)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in train_loader:
            output = model(xb)
            preds = output.argmax(dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

    print("Accuracy:", correct / total)
    # 🖼️ Visual check
    idx = 0
    plt.imshow(X[idx].reshape(28,28), cmap="gray")
    plt.title(f"Label: {y[idx].item()}")
    plt.axis("off")
    plt.show()
# 🔁 Load best model and predict
    model = get_model()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    for i in range(10):
        idx = torch.randint(0, len(X), (1,)).item()
        pred = model(X[idx].unsqueeze(0)).argmax().item()
        print("Actual:", y[idx].item(), "Pred:", pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
