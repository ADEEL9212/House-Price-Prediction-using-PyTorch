# 🏡 House Price Prediction using PyTorch

## 📌 Overview
This repository contains a **regression model** built using **PyTorch** to predict **house prices** based on features like:
- 📏 **Square Footage**
- 🏠 **Number of Bedrooms**
- 📍 **Location**

The project follows a step-by-step approach to training a neural network for regression tasks. We also visualize the training process and model performance using graphs. 📊

---

## 📂 Dataset
We use a housing dataset, such as:
- 🏙️ **California Housing Dataset** (from `sklearn.datasets`)
- 🏘️ **Any CSV file** with relevant house price features

---

## 🛠️ Installation & Setup
1️⃣ Clone this repository:
```bash
 git clone https://github.com/your-username/house-price-prediction.git
 cd house-price-prediction
```

2️⃣ Install dependencies:
```bash
 pip install -r requirements.txt
```

3️⃣ Run the training script:
```bash
 python train.py
```

---

## 📜 Steps to Build the Model
### 1️⃣ Import Libraries
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

### 2️⃣ Load the Dataset
```python
df = pd.read_csv("housing.csv")
print(df.head())
```

### 3️⃣ Preprocess Data
- Normalize features using `StandardScaler`
- Split into **training** and **testing** sets
```python
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['Price']))
y = df['Price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4️⃣ Convert Data to PyTorch Tensors
```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

### 5️⃣ Build the Regression Model
```python
class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
```

### 6️⃣ Define Loss Function & Optimizer
```python
model = HousePriceModel(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

### 7️⃣ Train the Model
```python
epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```

### 8️⃣ Evaluate the Model
```python
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    test_loss = criterion(y_pred_test, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')
```

---

## 📊 Visualizing Training Loss
```python
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()
```

📈 **Example Graph:**
![Loss Graph](https://via.placeholder.com/600x300?text=Loss+Graph)

---

## 🚀 Future Enhancements
- 🔍 **Hyperparameter tuning** (learning rate, batch size, etc.)
- 🏗️ **Experiment with deeper networks**
- 📊 **Feature Engineering** (adding more relevant house attributes)

---

## 📝 License
This project is **MIT licensed**. Feel free to use and modify it.

👩‍💻 **Contributions are welcome!** Feel free to submit pull requests. 🤝

---

⭐ **If you found this useful, don't forget to star the repo!** ⭐
