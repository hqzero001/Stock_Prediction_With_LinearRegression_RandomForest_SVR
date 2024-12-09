import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Bước 1: Load dữ liệu từ hai file
file1 = "D:\VisualStudio\SaveVS\Stock_Prediction_With_LinearRegression_RandomForest_SVR\SVR\AMD (1980 -11.07.2023).csv"
file2 = "D:\VisualStudio\SaveVS\Stock_Prediction_With_LinearRegression_RandomForest_SVR\SVR\AMD (2023 - 08.04.2024).csv"

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# Bước 2: Tiền xử lý dữ liệu
# Kiểm tra và hợp nhất dữ liệu
data = pd.concat([data1, data2], ignore_index=True)

# Đảm bảo cột 'Date' có định dạng thời gian và sắp xếp theo ngày
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

# Chọn cột 'Close' để dự đoán
data['Close_shifted'] = data['Close'].shift(-1)  # Dự đoán giá ngày hôm sau
data.dropna(inplace=True)

# Chuẩn bị dữ liệu đầu vào và đầu ra
X = data[['Close']].values
y = data['Close_shifted'].values

# Chuẩn hóa dữ liệu đầu vào
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

# Bước 3: Huấn luyện mô hình SVR
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Bước 4: Dự đoán trên toàn bộ dữ liệu để vẽ biểu đồ
data['Predicted_Close'] = model.predict(X_scaled)

# Vẽ biểu đồ
plt.figure(figsize=(15, 8))
plt.plot(data.index, data['Close'], label='Actual Price', color='blue', linewidth=1)
plt.plot(data.index, data['Predicted_Close'], label='Predicted Price', color='red', linewidth=1)
plt.title('SVR Stock Price Prediction for AMD')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
