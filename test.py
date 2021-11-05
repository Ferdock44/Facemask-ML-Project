from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

data = np.loadtxt("auto-mpg_removed_missing_values.data", usecols=range(0,8))
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
x = data[:, 1:]
y = data[:, 0]

w = np.array([0, 0, 0, 0, 0, 0, 0])
b = 0
alpha = 0.01

for i in range(50000):
    w = w - alpha * (1 / len(data)) * np.dot(np.transpose(np.dot(x, w)+b - y), x)
    b = b - alpha * (1 / len(data)) * sum(np.dot(x, w)+b - y)
print(w, b)
