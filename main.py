import pandas as pd
df = pd.read_csv('water_potability.csv', sep=',')

# Usable Supervised Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Usable Unsupervised Learning Algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Data: ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity,Potability
# Objective: Potability prediction with different algorithms

# Data Preprocessing
df.fillna(value=0, inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Data Splitting
from sklearn.model_selection import train_test_split
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Data Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training Supervised Learning
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
svm_model = SVC()
svm_model.fit(X_train, y_train)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Model Training Unsupervised Learning
kmeans_model = KMeans(n_clusters=2)
kmeans_model.fit(X_train)
gmm_model = GaussianMixture(n_components=2)
gmm_model.fit(X_train)

# Model Testing Supervised Learning
log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# Model Testing Unsupervised Learning
kmeans_pred = kmeans_model.predict(X_test)
gmm_pred = gmm_model.predict(X_test)

# Model Evaluation Supervised Learning
from sklearn.metrics import classification_report
print('Logistic Regression:')
print(classification_report(y_test, log_pred))
print('Random Forest:')
print(classification_report(y_test, rf_pred))
print('Support Vector Machine:')
print(classification_report(y_test, svm_pred))
print('K Nearest Neighbors:')
print(classification_report(y_test, knn_pred))

# Model Evaluation Unsupervised Learning
print('K-Means:')
print(classification_report(y_test, kmeans_pred))
print('Gaussian Mixture Model:')
print(classification_report(y_test, gmm_pred))

# Here's the CNN approach with pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train.to_numpy()).long()
y_test = torch.from_numpy(y_test.to_numpy()).long()

# Data Loader
class WaterDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.X)
train_dataset = WaterDataset(X_train, y_train)
test_dataset = WaterDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 9 -> 28 -> 28 -> 40 -> 20 -> 10 -> 5 -> 2

        self.starting = nn.Linear(9, 24)
        
        self.hid1 = nn.Linear(24, 28)
        self.hid2 = nn.Linear(28, 24)
        self.hid3 = nn.Linear(24, 9)

        self.output = nn.Linear(9, 2)
        
    def forward(self, x):
        x = F.relu(self.starting(x))
        x = F.relu(self.hid1(x))
        x = F.relu(self.hid2(x))
        x = F.relu(self.hid3(x))
        return F.sigmoid(self.output(x))
    
model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

# Training
epochs = 90000
for epoch in range(epochs):
    for data in train_loader:
        X, y = data
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} Loss: {loss.item():.16f}')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                X, y = data
                X = X.to(device)
                y = y.to(device)
                output = model(X)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(f'Accuracy: {100 * correct / total:.2f}%')
