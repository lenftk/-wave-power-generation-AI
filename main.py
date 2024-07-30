import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# CSV
csv_path = 'data.csv'

# CSV file lode
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# data
df['PowerGeneration'].plot(figsize=(12, 6))
plt.title('Power Generation Over Time')
plt.xlabel('Date')
plt.ylabel('Power Generation')
plt.show()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['PowerGeneration'].values.reshape(-1, 1))

# create data set
def create_dataset(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

time_steps = 10  # 시계열의 타임 스텝 수
X, y = create_dataset(scaled_data, time_steps)

# train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Power Generation')
plt.plot(y_pred, label='Predicted Power Generation')
plt.title('Power Generation Prediction')
plt.xlabel('Time')
plt.ylabel('Power Generation')
plt.legend()
plt.show()
