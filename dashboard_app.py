import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Book3.csv")
df.columns = [col.strip() for col in df.columns]
df["Time"] = pd.to_datetime(df["Time"], format="%b-%Y")

st.sidebar.title("🗂️ Retail Dashboard Controls")
category = st.sidebar.selectbox(
    "Select a Retail Category",
    [
        "Turnover ;  Total (State) ;  Food retailing ;",
        "Turnover ;  Total (State) ;  Household goods retailing ;",
        "Turnover ;  Total (State) ;  Clothing, footwear and personal accessory retailing ;",
        "Turnover ;  Total (State) ;  Department stores ;",
        "Turnover ;  Total (State) ;  Other retailing ;",
        "Turnover ;  Total (State) ;  Cafes, restaurants and takeaway food services ;",
    ]
)

min_date = df["Time"].min().to_pydatetime()
max_date = df["Time"].max().to_pydatetime()
start_date, end_date = st.slider("Select Time Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

filtered_df = df[(df["Time"] >= start_date) & (df["Time"] <= end_date)]

title_clean = category.split(";")[-2].strip()
st.markdown(f"### 🛍️ Monthly Turnover for **{title_clean}** ({start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')})")

color_map = {
    "Turnover ;  Total (State) ;  Food retailing ;": "forestgreen",
    "Turnover ;  Total (State) ;  Household goods retailing ;": "goldenrod",
    "Turnover ;  Total (State) ;  Clothing, footwear and personal accessory retailing ;": "deeppink",
    "Turnover ;  Total (State) ;  Department stores ;": "dodgerblue",
    "Turnover ;  Total (State) ;  Other retailing ;": "orangered",
    "Turnover ;  Total (State) ;  Cafes, restaurants and takeaway food services ;": "slateblue"
}
selected_color = color_map.get(category, "royalblue")

y = df[category].astype(float)
model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit(disp=False)
forecast_steps = 12
future_dates = pd.date_range(df["Time"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df["Time"],
    y=filtered_df[category],
    mode='lines',
    name='Actual Turnover',
    line=dict(color=selected_color, width=1, shape='spline')
))
fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast_mean,
    mode='lines',
    name='Forecast',
    line=dict(color='gray', dash='dot', width=2)
))
fig.add_trace(go.Scatter(
    x=future_dates,
    y=conf_int.iloc[:, 0],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=future_dates,
    y=conf_int.iloc[:, 1],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(200,200,200,0.3)',
    line=dict(width=0),
    name='95% Confidence Interval'
))
fig.update_layout(
    title=f"{title_clean} with Forecast (12 Months Ahead)",
    xaxis_title='Time',
    yaxis_title='Turnover ($ Millions)',
    template='plotly_dark',
    height=600
)
st.plotly_chart(fig, use_container_width=True)

change_col = f"Change in {title_clean}"
if change_col in df.columns:
    try:
        df[change_col] = df[change_col].astype(str).str.replace('%', '').astype(float)
        st.markdown(f"### 🔄 Monthly Percentage Change in **{title_clean}**")
        bar_fig = px.bar(
            filtered_df,
            x="Time",
            y=change_col,
            title=f"Percentage Change in {title_clean}",
            labels={change_col: "% Change"},
            template="plotly_white",
            color=change_col,
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    except:
        st.warning("⚠️ Percentage change data not formatted correctly.")

st.markdown("###### 📌 Summary")
st.metric("📆 First Month", df["Time"].iloc[0].strftime('%b %Y'))
st.metric("🛒 Latest Turnover", f"${df[category].iloc[-1]:,.2f} Million AUD")
st.metric("📈 Forecast for Next Month", f"${forecast_mean.iloc[0]:,.2f} Million AUD")

st.markdown("---")
st.caption("📊 Data Source: Australian Bureau of Statistics (ABS)")
st.caption("🛠️ Dashboard built with Streamlit and Plotly by Khush")