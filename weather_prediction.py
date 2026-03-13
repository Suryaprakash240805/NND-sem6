# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load dataset
data = pd.read_csv(r"C:\Users\Suriya Prakash\Downloads\archive (2)\weather_prediction_dataset.csv")

# Select temperature column for prediction
temperature = data[['BASEL_temp_mean']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_temp = scaler.fit_transform(temperature)

# Function to create sequences for RNN
def create_sequences(data, time_steps=10):
    X = []
    y = []

    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])

    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(scaled_temp, time_steps)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build RNN model
model = Sequential()

model.add(SimpleRNN(50, activation='tanh', input_shape=(time_steps,1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Make predictions
predictions = model.predict(X_test)

# Convert predictions back to original scale
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label="Actual Temperature")
plt.plot(predictions, label="Predicted Temperature")
plt.legend()
plt.title("Temperature Prediction using RNN")
plt.show()

last_days = scaled_temp[-time_steps:]   # last 10 days
last_days = last_days.reshape(1, time_steps, 1)
tomorrow_scaled = model.predict(last_days)
tomorrow_temp = scaler.inverse_transform(tomorrow_scaled)
print("Predicted temperature for tomorrow:", tomorrow_temp[0][0])