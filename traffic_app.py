# Create the fixed interactive app

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Traffic Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .input-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        background: #ff4b4b;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 5px;
        width: 100%;
    }
    .stButton > button:hover {
        background: #ff6b6b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚦 Smart Traffic Volume Predictor</h1>
    <p>Enter your input parameters to get real-time traffic predictions</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Sidebar for model training
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    
    st.markdown("### 📊 Training Data")
    years_data = st.slider("Years of historical data", 1, 5, 2, help="More years = better accuracy")
    
    if st.button("🔄 Train/Retrain Model", use_container_width=True):
        with st.spinner("Training model with historical data..."):
            # Generate training data
            np.random.seed(42)
            hours = 24 * 365 * years_data
            dates = pd.date_range(start='2020-01-01', periods=hours, freq='H')
            
            hour_of_day = np.array([d.hour for d in dates])
            day_of_week = np.array([d.dayofweek for d in dates])
            month = np.array([d.month for d in dates])
            
            # Create weekend indicator (0 for weekday, 1 for weekend)
            is_weekend_array = (day_of_week >= 5).astype(int)
            
            # Create realistic traffic patterns
            base_traffic = 150
            morning_peak = np.exp(-((hour_of_day - 8) ** 2) / 8) * 200
            evening_peak = np.exp(-((hour_of_day - 18) ** 2) / 8) * 180
            night_low = np.exp(-((hour_of_day - 2) ** 2) / 20) * 80
            
            traffic = base_traffic + morning_peak + evening_peak + night_low
            
            # Apply weekend factor (vectorized operation)
            weekend_factor = np.where(is_weekend_array == 1, 0.7, 1.0)
            traffic = traffic * weekend_factor
            
            # Apply seasonal factor
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
            traffic = traffic * seasonal_factor
            
            # Add noise
            traffic = traffic + np.random.normal(0, 15, len(traffic))
            traffic = np.clip(traffic, 20, 600)
            
            # Train Random Forest model
            X_train = np.column_stack([hour_of_day, day_of_week, month, is_weekend_array])
            y_train = traffic
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            st.session_state.model = model
            st.session_state.model_trained = True
            
            st.success("✅ Model trained successfully!")
            st.balloons()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **How it works:**
    This model predicts traffic volume based on:
    - 📍 Hour of day (0-23)
    - 📅 Day of week (Monday-Sunday)
    - 📆 Month (January-December)
    - 🎉 Weekend/Holiday status
    
    **Weather Impact:**
    - Clear: Normal traffic
    - Light Rain: +15% traffic
    - Heavy Rain: +35% traffic
    - Snow: +45% traffic
    - Fog: +25% traffic
    """)

# Main content area - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📝 Input Parameters")
    st.markdown("Enter the details below to get traffic prediction")
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Basic inputs
        st.markdown("### 🕐 Time & Date")
        
        col_date, col_time = st.columns(2)
        with col_date:
            input_date = st.date_input("Select Date", datetime.now())
        with col_time:
            input_hour = st.slider("Hour of Day", 0, 23, 9, help="0 = midnight, 12 = noon, 18 = 6 PM")
        
        # Day type
        st.markdown("### 📅 Day Type")
        day_type = st.radio("Select day type", ["Weekday", "Weekend/Holiday"], horizontal=True)
        
        # Weather conditions
        st.markdown("### 🌤️ Weather Conditions")
        weather = st.selectbox("Current weather", 
                              ["Clear ☀️", "Light Rain 🌧️", "Heavy Rain ⛈️", "Snow ❄️", "Fog 🌫️"],
                              index=0)
        
        # Weather impact mapping
        weather_impact = {
            "Clear ☀️": 1.0,
            "Light Rain 🌧️": 1.15,
            "Heavy Rain ⛈️": 1.35,
            "Snow ❄️": 1.45,
            "Fog 🌫️": 1.25
        }
        
        # Special events
        st.markdown("### 🎪 Special Events")
        has_event = st.checkbox("Special event nearby")
        
        event_impact = 0
        event_type = ""
        if has_event:
            event_type = st.selectbox("Event type", ["Concert 🎵", "Sports Game ⚽", "Festival 🎉", "Protest/March ✊"])
            event_impact = st.slider("Expected additional vehicles", 0, 200, 50, 
                                    help="Estimated increase in traffic due to event")
            
            # Event impact multipliers
            event_multiplier = {
                "Concert 🎵": 1.2,
                "Sports Game ⚽": 1.3,
                "Festival 🎉": 1.4,
                "Protest/March ✊": 1.5
            }
            event_impact = event_impact * event_multiplier.get(event_type, 1)
        
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## 🎯 Prediction Results")
    
    if st.session_state.model_trained and st.session_state.model is not None:
        # Prepare input features
        day_of_week = input_date.weekday()
        is_weekend = 1 if (day_type == "Weekend/Holiday" or day_of_week >= 5) else 0
        month = input_date.month
        
        # Make prediction
        features = np.array([[input_hour, day_of_week, month, is_weekend]])
        base_prediction = st.session_state.model.predict(features)[0]
        
        # Apply weather impact
        weather_factor = weather_impact[weather]
        final_prediction = base_prediction * weather_factor + event_impact
        
        # Calculate confidence
        confidence = 90 - (abs(weather_factor - 1) * 30) - (event_impact / 30)
        confidence = max(50, min(98, confidence))
        
        # Store prediction
        st.session_state.prediction_result = {
            'volume': final_prediction,
            'base': base_prediction,
            'confidence': confidence,
            'weather_factor': weather_factor,
            'event_impact': event_impact
        }
        
        # Display prediction cards
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        # Color code based on traffic level
        if final_prediction < 150:
            emoji = "🟢"
            level = "LIGHT"
        elif final_prediction < 300:
            emoji = "🟡"
            level = "MODERATE"
        elif final_prediction < 450:
            emoji = "🟠"
            level = "HEAVY"
        else:
            emoji = "🔴"
            level = "SEVERE"
        
        st.markdown(f"## {emoji} **{final_prediction:.0f}** vehicles/hour")
        st.markdown(f"### {level} TRAFFIC")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Metrics row
        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Base Traffic", f"{base_prediction:.0f}", help="Normal conditions without weather/events")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_met2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            weather_percent = (weather_factor - 1) * 100
            st.metric("Weather Impact", f"+{weather_percent:.0f}%", 
                     delta_color="inverse" if weather_factor > 1 else "normal")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_met3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.0f}%", 
                     help="Prediction confidence level")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Traffic status bar
        st.markdown("### 📊 Traffic Status")
        traffic_percent = min(100, (final_prediction / 600) * 100)
        st.progress(traffic_percent / 100)
        
        # Recommendations based on traffic level
        st.markdown("### 💡 Recommendations")
        
        if final_prediction < 150:
            st.success("✅ **Great time to travel!**")
            st.write("• Smooth flow expected")
            st.write("• No major delays anticipated")
            st.write("• Enjoy your journey!")
        elif final_prediction < 300:
            st.info("🚗 **Moderate traffic expected**")
            st.write("• Allow extra 15-20 minutes")
            st.write("• Consider alternative routes")
            st.write("• Use navigation apps for real-time updates")
        elif final_prediction < 450:
            st.warning("⚠️ **Heavy traffic expected**")
            st.write("• Allow extra 30-45 minutes")
            st.write("• Strongly consider alternative routes")
            st.write("• Public transport recommended if available")
            st.write("• Avoid peak hours if possible")
        else:
            st.error("🚨 **Severe traffic expected!**")
            st.write("• Avoid travel if possible")
            st.write("• If necessary, allow 1+ hour extra time")
            st.write("• Consider rescheduling your trip")
            st.write("• Use train/subway as alternative")
        
        # Prediction factors explanation
        with st.expander("🔍 View detailed prediction factors"):
            st.markdown(f"**📅 Date:** {input_date.strftime('%A, %B %d, %Y')}")
            st.markdown(f"**🕐 Hour:** {input_hour}:00")
            if input_hour in [7, 8, 9, 17, 18, 19]:
                st.markdown("• **Peak hour** → Higher traffic expected")
            elif input_hour in [22, 23, 0, 1, 2, 3, 4, 5]:
                st.markdown("• **Late night** → Lower traffic expected")
            
            st.markdown(f"**📅 Day type:** {day_type}")
            if is_weekend == 1:
                st.markdown("• **Weekend/Holiday** → Generally 30% less traffic")
            
            st.markdown(f"**🌤️ Weather:** {weather} (+{weather_percent:.0f}% impact)")
            
            if has_event:
                st.markdown(f"**🎪 Event:** {event_type} (+{event_impact:.0f} vehicles)")
                st.markdown("• Events cause localized congestion")
            
            st.markdown("---")
            st.markdown("**🎯 Prediction Formula:**")
            st.markdown(f"Base: {base_prediction:.0f} × Weather: {weather_factor:.2f} + Event: {event_impact:.0f} = **{final_prediction:.0f}**")
        
    else:
        st.warning("⚠️ **Model not trained yet!**")
        st.info("👈 Please click 'Train/Retrain Model' in the sidebar to start")
        st.image("https://via.placeholder.com/400x200?text=Train+Model+First", use_column_width=True)

# Multi-hour forecast section
st.markdown("---")
st.markdown("## 🔮 Multi-Hour Forecast")

if st.session_state.model_trained and st.session_state.model is not None:
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        forecast_hours = st.slider("How many hours to forecast?", 1, 24, 12)
    
    with col_f2:
        forecast_date = st.date_input("Forecast date", 
                                     datetime.now() + timedelta(days=1),
                                     help="Select date for forecast")
    
    with col_f3:
        forecast_weather = st.selectbox("Forecast weather", 
                                       ["Clear ☀️", "Light Rain 🌧️", "Heavy Rain ⛈️"], 
                                       index=0)
    
    if st.button("📈 Generate Forecast", use_container_width=True):
        # Generate forecast
        forecast_hours_list = []
        forecast_volumes = []
        
        start_hour = input_hour if 'input_hour' in locals() else 9
        forecast_day = forecast_date.weekday()
        forecast_month = forecast_date.month
        forecast_is_weekend = 1 if forecast_day >= 5 else 0
        
        for i in range(forecast_hours):
            hour = (start_hour + i) % 24
            features = np.array([[hour, forecast_day, forecast_month, forecast_is_weekend]])
            pred = st.session_state.model.predict(features)[0]
            
            # Apply weather factor
            weather_factor = weather_impact[forecast_weather]
            pred = pred * weather_factor
            
            forecast_hours_list.append(f"{hour}:00")
            forecast_volumes.append(pred)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Hour': forecast_hours_list,
            'Predicted Volume': [f"{v:.0f}" for v in forecast_volumes],
            'Status': ['🟢' if v < 150 else '🟡' if v < 300 else '🟠' if v < 450 else '🔴' for v in forecast_volumes]
        })
        
        # Plot forecast
        fig, ax = plt.subplots(figsize=(12, 5))
        hours_range = range(len(forecast_volumes))
        
        # Color code the line based on traffic level
        colors = ['green' if v < 150 else 'orange' if v < 300 else 'red' for v in forecast_volumes]
        
        for i in range(len(forecast_volumes) - 1):
            ax.plot([i, i+1], [forecast_volumes[i], forecast_volumes[i+1]], 
                   color=colors[i], linewidth=2)
        
        ax.scatter(hours_range, forecast_volumes, c=colors, s=100, zorder=5)
        ax.fill_between(hours_range, forecast_volumes, alpha=0.2, color='gray')
        
        ax.set_xlabel('Hour from now', fontsize=12)
        ax.set_ylabel('Predicted Traffic Volume', fontsize=12)
        ax.set_title(f'Traffic Forecast for {forecast_date.strftime("%A, %B %d")}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(hours_range)
        ax.set_xticklabels(forecast_hours_list, rotation=45)
        
        # Add horizontal lines for traffic levels
        ax.axhline(y=150, color='green', linestyle='--', alpha=0.5, label='Light traffic threshold')
        ax.axhline(y=300, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')
        ax.axhline(y=450, color='red', linestyle='--', alpha=0.5, label='Heavy threshold')
        
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show table
        st.dataframe(forecast_df, use_container_width=True)
        
        # Best time to travel
        best_hour_idx = np.argmin(forecast_volumes)
        best_volume = forecast_volumes[best_hour_idx]
        
        if best_volume < 150:
            st.success(f"✨ **Best time to travel:** {forecast_hours_list[best_hour_idx]} with only {best_volume:.0f} vehicles/hour 🟢")
        elif best_volume < 300:
            st.info(f"✨ **Best time to travel:** {forecast_hours_list[best_hour_idx]} with {best_volume:.0f} vehicles/hour 🟡")
        else:
            st.warning(f"⚠️ All forecasted hours have moderate-heavy traffic. Consider postponing travel.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🚦 Smart Traffic Prediction System | Powered by Random Forest ML Model</p>
    <p>Enter your inputs and get instant traffic predictions with recommendations!</p>
    <p style='font-size: 0.8rem;'>⚠️ Predictions are estimates based on historical patterns and weather conditions</p>
</div>
""", unsafe_allow_html=True)


