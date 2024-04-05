from polygon import WebSocketClient, RESTClient
from polygon.websocket.models import WebSocketMessage
from typing import List
import pandas as pd
import datetime
from nixtlats import TimeGPT
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
import io
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

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

st.set_page_config(layout="wide")
st.title('Real-time and Predicted Candlestick Charts')

tickers = st.selectbox("Select Company", companies.keys(), index=0)
ticker = companies[tickers]

client = WebSocketClient(
    api_key="XRcvi1wdOpza2JAdKVigSjL5fzXtwhEG", feed='delayed.polygon.io', market='stocks', subscriptions=[f"A.{ticker}"])
rest_client = RESTClient(api_key="XRcvi1wdOpza2JAdKVigSjL5fzXtwhEG")

timegpt = TimeGPT(
    token='kV8Ye5yTL0jeGkUEYpawjwLLiHUuoq0uZPelYOzNUyUODxfDKgRKT5AozNZk4LFBhOCUKie2kv8eCMMBRRoIrvrYQwR3LDXkm4xzCoaXEKatOUK6oGCXZJ9wfj4LqUlGbB05ZHXdKpxO0HANSU3nWiJSMh3xQbhB52R53tMIAXRimP1EaxSSONcgqrW6QTigCwTzxi8UxtN4sD0nV4rZfkP2o5i84gnKv9YfK8h9B44ETLfo6E8JnZkSxHQAphYk'
)

executor = ThreadPoolExecutor()


def predict(df):
    stock_prices = df.copy(deep=True)
    stock_prices['ds'] = stock_prices.index
    stock_prices.reset_index(level=0, inplace=True, drop=True)
    stock_prices = stock_prices.melt(
        id_vars='ds', var_name='unique_id', value_name='y')
    # add_history=True fewshot_steps=100
    timegpt_fcst_df = timegpt.forecast(
        df=stock_prices, h=5, freq="min")

    # Convert back to original form using pivot
    global pred_df
    pred_df = timegpt_fcst_df.pivot(
        index='ds', columns='unique_id', values='TimeGPT')
    pred_df.index = pd.to_datetime(pred_df.index)
    pred_df.index.name = "Datetime"


def plot_to_image_array(plot):
    buf = io.BytesIO()
    plot.savefig(buf, format='jpg')
    buf.seek(0)
    image = Image.open(buf)
    image_array = np.array(image)
    return image_array


def handle_msg(msgs: List[WebSocketMessage]):
    realtime_df.index.name = "Datetime"
    for m in msgs:
        timestamp = datetime.datetime.fromtimestamp(
            (int(m.start_timestamp)/1000) - 34200).replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
        if (handle_msg.old_timestampt and timestamp != handle_msg.old_timestampt) or handle_msg.old_timestampt is None:
            handle_msg.open = m.open
        handle_msg.high = max(handle_msg.high, m.high)
        handle_msg.low = min(handle_msg.low, m.low)

        realtime_df.loc[timestamp] = [handle_msg.open,
                                      handle_msg.low, handle_msg.high, m.close]
        realtime_df.index = pd.to_datetime(realtime_df.index)
        with col1.container():
            with realtime_fig.container():
                # Plot real-time candlestick
                fig, ax = plt.subplots(figsize=(8, 6))
                mpf.plot(realtime_df.iloc[-30:, :], ax=ax, type='candle',
                         style='charles', volume=False, datetime_format='%H:%M')
                # Convert the plot to image
                st.image(plot_to_image_array(fig), use_column_width=True)
                # plt.close()

        if (datetime.datetime.now() - handle_msg.last_prediction_time).seconds >= 10:
            print("Running thread")
            executor.submit(predict, realtime_df)
            handle_msg.last_prediction_time = datetime.datetime.now()

        if pred_df is not None:
            with col2.container():
                with predicted_fig.container():
                    # Plot real-time candlestick
                    fig, ax = plt.subplots(figsize=(8, 6))

                    mpf.plot(pred_df, ax=ax, type='candle',
                             style='charles', volume=False, datetime_format='%H:%M:%S')

                    # Convert the plot to image
                    st.image(plot_to_image_array(fig), use_column_width=True)
                    # plt.close()

            with pred_df_container.container():
                st.table(pred_df)
        handle_msg.old_timestampt = timestamp


col1, col2 = st.columns(2)

with col1:
    realtime_fig = st.empty()

with col2:
    predicted_fig = st.empty()

pred_df_container = st.empty()

pred_df = None

current_date = datetime.datetime.now().date()
yesterday_date = current_date - datetime.timedelta(days=1)

start = str(yesterday_date)
end = str(current_date)
# List Aggregates (Bars)
realtime_df = pd.DataFrame(columns=['open', 'low', 'high', 'close'])
for a in rest_client.list_aggs(ticker=ticker, multiplier=1, timespan="minute", from_=start, to=end):
    timestamp = datetime.datetime.fromtimestamp(int(a.timestamp)/1000 - 34200)
    realtime_df.loc[timestamp] = [a.open, a.low, a.high, a.close]

# realtime_df = pd.DataFrame(columns=['open', 'low', 'high', 'close'])
handle_msg.last_prediction_time = datetime.datetime.now()
handle_msg.old_timestampt = None
handle_msg.open, handle_msg.high, handle_msg.low = 0, 0, np.inf
# print messages
client.run(handle_msg)
