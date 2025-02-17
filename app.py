import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

# Streamlit Page Config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
)

# Custom Styling
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
        }
        .stApp {
            background: linear-gradient(to right, #141e30, #243b55);
        }
        h1, h2, h3 {
            color: #FFA500;
            text-align: center;
        }
        .stButton>button {
            color: white !important;
            background: #007BFF !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
        }
        .stFileUploader {
            border: 2px dashed #FFA500 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Title
st.title("📈 AI-Based Stock Price Prediction")
st.subheader("Upload a CSV file to analyze and predict stock trends")

# Sidebar Information
st.sidebar.title("⚙️ Model Settings")
st.sidebar.info(
    "🔹 **LSTM Model Used**: 2 Layers\n"
    "🔹 **Optimizer**: Adam\n"
    "🔹 **Epochs**: 5\n"
    "🔹 **Batch Size**: 1"
)

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Dataset
    dataset = pd.read_csv(uploaded_file)
    
    # Display Dataset Information
    st.success("File uploaded successfully! ✅")
    st.write(f"🔹 **Number of Rows:** {dataset.shape[0]}")
    st.write(f"🔹 **Number of Columns:** {dataset.shape[1]}")
    st.write("🔹 **Column Names:**", list(dataset.columns))
    
    # Show First Few Rows
    st.subheader("🔍 Preview of Dataset")
    st.dataframe(dataset.head())

    # Process Dataset for Prediction
    dataset = dataset.iloc[:, 1:5]  # Assuming OHLC columns are 2nd to 5th
    dataset = dataset.reindex(index=dataset.index[::-1])
    
    # Prepare Data
    OHLC_avg = dataset.mean(axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg = scaler.fit_transform(np.reshape(OHLC_avg.values, (len(OHLC_avg), 1)))
    
    # Train-Test Split
    train_size = int(len(OHLC_avg) * 0.75)
    train_data, test_data = OHLC_avg[:train_size], OHLC_avg[train_size:]

    # Prepare Time-Series Data
    def create_dataset(data, step=1):
        X, Y = [], []
        for i in range(len(data) - step - 1):
            X.append(data[i:(i+step), 0])
            Y.append(data[i + step, 0])
        return np.array(X), np.array(Y)
    
    trainX, trainY = create_dataset(train_data, 1)
    testX, testY = create_dataset(test_data, 1)
    
    # Reshape Data for LSTM
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 1), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train Model
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

    # Predict Stock Prices
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Inverse Transform Predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    # Get Today's Price & Prediction
    today_price = testPredict[-1]
    last_val_scaled = today_price / today_price
    next_day_prediction = model.predict(np.reshape(last_val_scaled, (1, 1, 1)))
    next_day_price = (today_price * next_day_prediction).item()

    # Investment Suggestion Logic
    if next_day_price > today_price:
        suggestion = "📈 **Suggestion:** Stock price is increasing! It's a good time to **BUY** ✅"
        suggestion_color = "green"
    else:
        suggestion = "📉 **Suggestion:** Stock price is decreasing. **Hold for now** ⏳"
        suggestion_color = "red"

    # Visualization
    st.subheader("📊 Stock Price Predictions")
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(OHLC_avg), label="Original Data", color="green")
    plt.plot(np.arange(len(trainPredict)), trainPredict, label="Training Prediction", color="red")
    plt.plot(np.arange(len(trainPredict), len(trainPredict) + len(testPredict)), testPredict, label="Test Prediction", color="blue")
    plt.legend()
    st.pyplot(fig)

    # Display Future Prediction
    st.subheader("🔮 Future Prediction")
    st.write(f"📌 **Today's Price:** {today_price.item():.2f}")
    st.write(f"📌 **Next Day Prediction:** {next_day_price:.2f}")
    
    # Display Suggestion
    st.markdown(
        f"<h3 style='color:{suggestion_color}; text-align:center;'>{suggestion}</h3>",
        unsafe_allow_html=True
    )

    st.success("✅ Prediction & Suggestion Generated!")

# Footer
st.markdown(
    "<h4 style='text-align: center; color: #228B22;'>"
    "🚀 Developed with ❤️ by Islavath Raghunaik</h4>",
    unsafe_allow_html=True
)
