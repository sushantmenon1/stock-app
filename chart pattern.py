import io
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import mplfinance as mpf
from nixtlats import TimeGPT
import time

# Function to download stock data
timegpt = TimeGPT(
    # defaults to os.environ.get("TIMEGPT_TOKEN")
    token='Wm9zzEqSJzFxaKzeQuNVpq2ikAVlnvSezB8YbE54yWQFARswm0WfG6U8bIrUaBQNKG1PLuMSvn0QFhacZygi2p9YhwVDy4dwasXzQN3JdDeMBINb4sO11l5nkRFeKztWZsjJcxdtyK9NmNfHMWwgzmCN0248weZGoIPgf3EYUBS8FUZg0qKRKwWh1SCynyhBUWSnakrFBoQtWVlrKRB9ucR7afHWup5c5lRjkmuRdGshCJMlgT9P29oXr3WZTaPl'
)


def download_stock_data(ticker, interval):
    stock_prices = (
        yf.Ticker(ticker=ticker)
        .history(period="1d", interval=interval)
        .ffill()
    ).dropna().drop(["Volume", "Dividends", "Stock Splits"], axis=1)
    stock_prices.index = stock_prices.index.tz_localize(None)
    return stock_prices


def get_prediction(stock_prices):
    start_time = time.time()
    stock_prices['ds'] = stock_prices.index
    stock_prices.reset_index(level=0, inplace=True, drop=True)
    stock_prices = stock_prices.melt(
        id_vars='ds', var_name='unique_id', value_name='y')
    # add_history=True fewshot_steps=100
    timegpt_fcst_df = timegpt.forecast(
        df=stock_prices, h=5, freq="min", fewshot_steps=100)

    # Convert back to original form using pivot
    pred_df = timegpt_fcst_df.pivot(
        index='ds', columns='unique_id', values='TimeGPT')
    pred_df.index = pd.to_datetime(pred_df.index)
    pred_df.index.name = "Datetime"
    end_time = time.time()
    print(f"Time taken by the predict function is {end_time-start_time}")
    return pred_df

# Create a function to convert plot to image array


def plot_to_image_array(plot):
    buf = io.BytesIO()
    plot.savefig(buf, format='jpg')
    buf.seek(0)
    image = Image.open(buf)
    image_array = np.array(image)
    return image_array


# Streamlit app
st.title("Live Stock Data and Candlestick Chart")

# Parameters
companies = {"Google": "GOOGL",
             "Apple": "AAPL",
             "Tesla": "TSLA",
             "Meta": "META",
             "Microsoft": "MSFT",
             "Bank of America": "BAC",
             "Nvidia": "NVDA",
             "Amazon": "AMZN",
             "Visa": "V",
             "Netflix": "NFLX"}

tickers = st.selectbox("Select Company", companies.keys(), index=0)
ticker = companies[tickers]
interval = st.selectbox("Select interval", [
                        '1m', '2m', '5m', '15m', '30m'], index=0)

placeholder = st.empty()
st.sidebar.title("Predictions")
sidebar_placeholder = st.sidebar.empty()

while True:
    with placeholder.container():
        # Download initial stock data
        data = download_stock_data(ticker, interval)
        original_df = data.copy(deep=True)

        # Check if data has changed
        if st.session_state.get('counter') and st.session_state.get('company') and len(data) == st.session_state['counter'] and tickers == st.session_state.get('company'):
            continue

        # Plot figures
        fig, ax = plt.subplots()
        pred_data = get_prediction(data)

        # Show predictions in sidebar
        sidebar_placeholder.table(pred_data)

        combined_data = pd.concat([original_df, pred_data])

        # Display the initial candlestick chart
        mpf.plot(combined_data.iloc[-30:, :], type='candle', style='charles', volume=False,
                 ax=ax)

        # Convert the plot to image
        st.image(plot_to_image_array(fig), use_column_width=True)
        plt.close()

        # Counter to keep track of updated data length
        st.session_state['counter'] = len(original_df)
        st.session_state['company'] = tickers
