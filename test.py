import torch
import pandas as pd
from new_train import get_model
test = pd.read_csv("dataset/mnist_test.csv")
X_test = test.values / 255.0
X_test = torch.tensor(X_test, dtype=torch.float32)

print("X_test shape:", X_test.shape)  # should be [28000, 784]

model = get_model()
model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    preds = outputs.argmax(dim=1)
    print(torch.unique(preds))


print("Predictions:", preds[:20])
import matplotlib.pyplot as plt

for i in range(20):
    plt.imshow(X_test[i].reshape(28,28), cmap="gray")
    plt.title(f"Pred: {preds[i].item()}")
    plt.show()