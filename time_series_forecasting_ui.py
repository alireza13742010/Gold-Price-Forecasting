import streamlit as st
import pandas as pd
import numpy as np
import os 
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO
import base64

st.title(":red[Gold Price Forecast & Backtesting UI]")

tabs = st.tabs(["Live Forecast", "Backtesting"])


def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
     )
set_bg_hack('/home/avidmech/Image_Generation_DeepSeek_1B pro/Janus/get.png')
##############################################
# Live Forecast
##############################################
with tabs[0]:
    st.header(":blue[Live Forecast]")

    # آپلود فایل CSV برای Live Forecast (اختیاری)
    uploaded_file_live = st.file_uploader("Upload CSV file for Live Forecast", type=["csv"], key="live")
    if uploaded_file_live is not None:
        df_live = pd.read_csv(uploaded_file_live)
        st.subheader("CSV Data Preview for Live Forecast")
        st.dataframe(df_live.head())

    # ورودی‌های دستی جهت وارد کردن اعداد
    high = st.number_input("Enter High", value=0.0)
    open_price = st.number_input("Enter Open", value=0.0)
    low = st.number_input("Enter Low", value=0.0)
    close_price = st.number_input("Enter Close", value=0.0)
    ma20 = st.number_input("Enter MA20", value=0.0)
    

    if st.button("Run Live Forecast", key="live_run"):
        global_path = os.getcwd()
        #model3_days_1_hourforcast(2025-3-12)(ELU).keras
        #model3_days_1_hourforcast(2025-3-12)(leaky_relu).keras
        model_path = f"{global_path}/model3_days_1_hourforcast(2025-3-12)(leaky_relu).keras"
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        features_live = np.array([[high, open_price, low, close_price, ma20]])
        features_live = features_live.reshape((1, 5, 1))
        pred_live = model.predict(features_live)

        st.subheader(":green[Live Forecast Result]")
        st.write(f":green[Predicted Close: {pred_live[0][0]:.4f}]")

##############################################
# Backtesting
##############################################
with tabs[1]:
    st.header(":blue[Backtesting]")

    uploaded_file_backtest = st.file_uploader("Upload CSV file for Backtesting", type=["csv"], key="backtest")
    if uploaded_file_backtest is not None:
        df_back = pd.read_csv(uploaded_file_backtest)
        st.subheader("CSV Data Preview")
        st.dataframe(df_back.head())

        required_cols = ["date", "time", "high", "open", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df_back.columns]
        if missing_cols:
            st.error(f"CSV file must contain the following columns: {missing_cols}")
            st.stop()

        try:
            df_back["datetime"] = pd.to_datetime(
                df_back["date"] + " " + df_back["time"],
                format="%Y.%m.%d %H:%M",
                errors="raise"
            )
        except Exception as e:
            st.error(f"Error converting date/time columns to datetime: {e}")
            st.stop()

        if "ma20" not in df_back.columns:
            df_back["ma20"] = df_back["open"].rolling(window=20, min_periods=1).mean()

        df_back = df_back.sort_values(by="datetime").reset_index(drop=True)

        st.subheader("Specify Backtesting Date/Time Range")
        min_datetime = df_back["datetime"].min()
        max_datetime = df_back["datetime"].max()

        start_date = st.date_input("Start Date", value=min_datetime.date(), min_value=min_datetime.date(), max_value=max_datetime.date(), key="backtest_start_date")
        start_time = st.time_input("Start Time", value=min_datetime.time(), key="backtest_start_time")

        end_date = st.date_input("End Date", value=max_datetime.date(), min_value=min_datetime.date(), max_value=max_datetime.date(), key="backtest_end_date")
        end_time = st.time_input("End Time", value=max_datetime.time(), key="backtest_end_time")

        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)

        filtered_df = df_back[(df_back["datetime"] >= start_datetime) & (df_back["datetime"] <= end_datetime)].copy()

        if len(filtered_df) < 2:
            st.warning("Not enough data in selected range to perform backtesting.")
        else:
            if st.button("Run Backtesting", key="backtest_run"):
                global_path = os.getcwd()
                #model3_days_1_hourforcast_hyperparamtertuned.keras
                #model3_days_1_hourforcast1.keras
                #model3_days_1_hourforcast.keras
                #model3_days_1_hourforcast(2025-3-12)(ELU).keras
                model_path = f"{global_path}/model3_days_1_hourforcast(2025-3-12)(leaky_relu).keras"
                try:
                    model = load_model(model_path)
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.stop()

                predictions = []
                actuals = []
                datetimes = []
                datetimes_pred = []
			
                for i in range(len(filtered_df)):
                    row = filtered_df.iloc[i]
                    features = np.array([[row["high"], row["open"], row["low"], row["close"], row["ma20"]]])
                    features = features.reshape((1, 5, 1))
                    pred = model.predict(features)
                    predictions.append(pred[0][0])
                    actuals.append(filtered_df["close"].iloc[i])
                    datetimes.append(filtered_df["datetime"].iloc[i])
                    datetimes_pred.append(filtered_df["datetime"].iloc[i] + timedelta(hours=1))

                mae = mean_absolute_error(actuals[1:], predictions[:-1])
                mse = mean_squared_error(actuals[1:], predictions[:-1])
                r2 = r2_score(actuals[1:], predictions[:-1])

                st.subheader("Backtesting Results")
                st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.write(f"R² Score: {r2:.4f}")
                
                RMSE = (np.array(actuals[1:]) - np.array(predictions[:-1]))**2/(int(np.array(actuals[1:]).shape[0]))
                fig0 = go.Figure()
                fig0.add_trace(go.Scatter(x=datetimes_pred[:-1], y=RMSE, mode='lines+markers', name='RMSE', fillcolor = 'rosybrown'))
                fig0.update_layout(
                    title='Backtesting: The RMSE for each sample',
                    xaxis_title='Date/Time',
                    yaxis_title='RMSE')
     
                st.plotly_chart(fig0, use_container_width=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=datetimes, y=actuals, mode='lines+markers', name='Actual Close'))
                fig.add_trace(go.Scatter(x=datetimes_pred, y=predictions, mode='lines+markers', name='Predicted Close'))
                fig.update_layout(
                    title='Backtesting: Actual vs Predicted Close Prices',
                    xaxis_title='Date/Time',
                    yaxis_title='Close Price',
                    legend=dict(x=0, y=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                results_df = pd.DataFrame({
                    "datetime": datetimes,
                    "actual": actuals,
                    "predicted": predictions
                })
                results_df["datetime"] = results_df["datetime"].astype(str)

                towrite = BytesIO()
                results_df.to_excel(towrite, index=False, sheet_name="Backtesting_Results")
                towrite.seek(0)

                st.download_button(
                    label="Download Backtesting Results as Excel",
                    data=towrite,
                    file_name="backtesting_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
