# traffic_prediction.py
# Complete working LSTM Traffic Prediction System

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚦 LSTM Traffic Volume Prediction System")
print("=" * 60)

# ===============================================================
# 1️⃣ GENERATE SYNTHETIC TRAFFIC DATA (since no CSV file)
# ===============================================================
print("\n📊 Generating synthetic traffic data...")

np.random.seed(42)
hours = 24 * 365 * 2  # 2 years of hourly data
dates = pd.date_range(start='2023-01-01', periods=hours, freq='H')

# Create realistic traffic patterns
# Base pattern: rush hours (8-9am and 5-6pm) have higher traffic
hour_of_day = np.array([d.hour for d in dates])
day_of_week = np.array([d.dayofweek for d in dates])

# Morning rush (7-10am): high traffic
morning_rush = np.exp(-((hour_of_day - 8) ** 2) / 8) * 0.8
# Evening rush (4-7pm): high traffic
evening_rush = np.exp(-((hour_of_day - 17) ** 2) / 8) * 0.8
# Night time (11pm-5am): low traffic
night_time = np.exp(-((hour_of_day - 2) ** 2) / 20) * 0.2

# Weekend effect (lower traffic on Sat/Sun)
weekend_factor = np.where(day_of_week >= 5, 0.6, 1.0)

# Seasonal effect
seasonal = 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 24))

# Noise
noise = np.random.normal(0, 0.05, len(dates))

# Combine to create volume (50-500 range)
vol = 100 + morning_rush * 300 + evening_rush * 250 + night_time * 50
vol = vol * weekend_factor
vol = vol + seasonal * 100 + noise * 50
vol = np.clip(vol, 30, 550)

# Create DataFrame
data = pd.DataFrame({
    'datetime': dates,
    'vol': vol,
    'hour': hour_of_day,
    'day_of_week': day_of_week,
    'month': [d.month for d in dates]
})

print(f"✅ Generated {len(data):,} hourly records")
print(f"📈 Traffic volume range: {data['vol'].min():.0f} - {data['vol'].max():.0f}")
print(f"📊 Average traffic volume: {data['vol'].mean():.0f}")

# ===============================================================
# 2️⃣ PREPARE DATA FOR LSTM
# ===============================================================
print("\n🔧 Preparing data for LSTM model...")

# Features: use hour and day_of_week to predict volume
features = ['hour', 'day_of_week', 'month']
target = 'vol'

# Scale the data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(data[features])
y_scaled = target_scaler.fit_transform(data[[target]])

# Create sequences
window_size = 24  # Use past 24 hours to predict next hour
horizon = 1

def create_sequences(X, y, window_size, horizon=1):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size - horizon + 1):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size + horizon - 1])
    return np.array(X_seq), np.array(y_seq)

X, y = create_sequences(X_scaled, y_scaled, window_size, horizon)

# Train/test split (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")
print(f"📐 Input shape: {X_train.shape[1]} timesteps × {X_train.shape[2]} features")

# ===============================================================
# 3️⃣ BUILD LSTM MODEL
# ===============================================================
print("\n🏗️ Building LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, X.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ===============================================================
# 4️⃣ TRAIN MODEL
# ===============================================================
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=64,
    verbose=1
)

# ===============================================================
# 5️⃣ EVALUATE MODEL
# ===============================================================
print("\n📊 Evaluating model...")

# Make predictions
y_pred_scaled = model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_actual = target_scaler.inverse_transform(y_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

print(f"\n📈 Evaluation Results:")
print(f"   RMSE: {rmse:.2f} vehicles")
print(f"   MAE:  {mae:.2f} vehicles")
print(f"   MAPE: {mape:.1f}%")

# ===============================================================
# 6️⃣ VISUALIZATION
# ===============================================================
print("\n📉 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training History
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_title('Model Training History', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (last 500 samples)
ax2 = axes[0, 1]
test_indices = range(len(y_actual))[-500:]
ax2.plot(test_indices, y_actual[-500:], label='Actual Traffic', alpha=0.7, linewidth=1)
ax2.plot(test_indices, y_pred[-500:], label='Predicted', alpha=0.7, linewidth=1)
ax2.set_title('Actual vs Predicted Traffic (Last 500 Hours)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time Step (Hours)')
ax2.set_ylabel('Traffic Volume')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Scatter Plot
ax3 = axes[1, 0]
ax3.scatter(y_actual, y_pred, alpha=0.5, s=10)
ax3.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='Perfect Prediction')
ax3.set_title('Prediction Scatter Plot', fontsize=12, fontweight='bold')
ax3.set_xlabel('Actual Volume')
ax3.set_ylabel('Predicted Volume')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Error Distribution
ax4 = axes[1, 1]
errors = (y_pred - y_actual).flatten()
ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax4.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Prediction Error')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('traffic_prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'traffic_prediction_results.png'")

# ===============================================================
# 7️⃣ FUTURE PREDICTIONS (Next 7 days)
# ===============================================================
print("\n🔮 Generating 7-day forecast...")

# Get last window and predict next 168 hours (7 days)
last_window = X[-1].copy()
future_predictions = []

for i in range(168):  # 7 days * 24 hours
    pred_scaled = model.predict(last_window[np.newaxis, :, :], verbose=0)[0, 0]
    future_predictions.append(pred_scaled)
    
    # Update window: remove first, add prediction
    new_row = last_window[-1].copy()
    new_row[0] = pred_scaled  # Update the target position
    last_window = np.vstack([last_window[1:], new_row])

# Inverse transform predictions
future_volumes = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create forecast dates
last_date = data['datetime'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=168, freq='H')

# Plot forecast
plt.figure(figsize=(14, 6))
plt.plot(future_dates, future_volumes, color='red', linewidth=2, label='Forecasted Traffic')
plt.fill_between(future_dates, 
                 future_volumes.flatten() - rmse, 
                 future_volumes.flatten() + rmse, 
                 alpha=0.3, color='red', label='Confidence Interval (±1 RMSE)')
plt.title('7-Day Traffic Volume Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Date & Time')
plt.ylabel('Predicted Traffic Volume')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('traffic_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Forecast saved as 'traffic_forecast.png'")

# ===============================================================
# 8️⃣ HOURLY PATTERN ANALYSIS
# ===============================================================
print("\n📊 Analyzing hourly traffic patterns...")

# Get average traffic by hour from model predictions
hourly_avg = data.groupby('hour')['vol'].mean()

plt.figure(figsize=(12, 5))
plt.bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7)
plt.title('Average Traffic Volume by Hour of Day', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Average Traffic Volume')
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(hourly_avg.values):
    plt.text(i, v + 5, f'{v:.0f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('hourly_pattern.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Hourly pattern saved as 'hourly_pattern.png'")

# ===============================================================
# 9️⃣ SAVE MODEL
# ===============================================================
print("\n💾 Saving model...")
model.save('traffic_lstm_model.h5')
print("✅ Model saved as 'traffic_lstm_model.h5'")

# ===============================================================
# 🔟 SUMMARY
# ===============================================================
print("\n" + "=" * 60)
print("🎉 TRAFFIC PREDICTION SYSTEM - COMPLETE!")
print("=" * 60)
print(f"\n📁 Generated Files:")
print(f"   - traffic_lstm_model.h5 (LSTM model)")
print(f"   - traffic_prediction_results.png (visualization)")
print(f"   - traffic_forecast.png (7-day forecast)")
print(f"   - hourly_pattern.png (hourly patterns)")
print(f"\n📊 Model Performance:")
print(f"   - RMSE: {rmse:.2f} vehicles")
print(f"   - MAE: {mae:.2f} vehicles")
print(f"   - MAPE: {mape:.1f}%")
print("\n✨ The model is ready for real-time predictions!")
print("=" * 60)