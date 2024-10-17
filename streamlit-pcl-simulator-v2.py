import streamlit as st
import math
from decimal import Decimal, getcontext
import numpy as np
import plotly.graph_objs as go

# Set high precision for decimal calculations
getcontext().prec = 30

def geometric_mean(x, y):
    return (Decimal(x) * Decimal(y)).sqrt()

def calc_y(x, xy, A):
    return xy / x

def compute_swap(x1, y1, dx, A):
    xy = x1 * y1
    x2 = x1 + dx
    y2 = calc_y(x2, xy, A)
    dy = y1 - y2
    return dy

def calculate_trade_impact(x, y, dx, A):
    dy = compute_swap(x, y, dx, A)
    expected_dy = dx * (y / x)
    return (expected_dy - dy) / expected_dy * 100

st.title('Astroport PCL Pool Simulator')

# Sidebar for inputs
st.sidebar.header('Pool Parameters')
total_liquidity = st.sidebar.slider('Total Pool Size', 1_000_000, 1_000_000_000, 50_000_000, step=1_000_000, format="%d")
sale_amount = st.sidebar.number_input('eclipASTRO Sale Amount', 1, 1_000_000, 10_000, step=1_000)

st.sidebar.header('PCL Parameters')
A = st.sidebar.number_input('Amp (A)', 1.0, 1000.0, 500.0, step=1.0)
gamma = st.sidebar.number_input('Gamma', 0.0, 1.0, 0.01, step=0.01)
mid_fee = st.sidebar.number_input('Mid Fee', 0.0, 1.0, 0.0003, step=0.0001, format="%.4f")
out_fee = st.sidebar.number_input('Out Fee', 0.0, 1.0, 0.0045, step=0.0001, format="%.4f")
fee_gamma = st.sidebar.number_input('Fee Gamma', 0.0, 1.0, 0.3, step=0.1)

# Display initial pool split
half_pool = total_liquidity // 2
st.write(f'Initial xASTRO: {half_pool:,}')
st.write(f'Initial eclipASTRO: {half_pool:,}')

# Simulate and plot
initial_liquidity_float = float(total_liquidity)
sale_amount_decimal = Decimal(str(sale_amount))
A = Decimal(str(A))
gamma = Decimal(str(gamma))
mid_fee = Decimal(str(mid_fee))
out_fee = Decimal(str(out_fee))
fee_gamma = Decimal(str(fee_gamma))

x_range = np.linspace(0.01, 0.99, 100) * initial_liquidity_float
impacts = []

for x in x_range:
    y = initial_liquidity_float - x
    impact = calculate_trade_impact(Decimal(str(x)), Decimal(str(y)), sale_amount_decimal, A)
    impacts.append(float(impact))

# Create the Plotly figure
fig = go.Figure()

# Add the main line
fig.add_trace(go.Scatter(x=x_range / initial_liquidity_float * 100, y=impacts, mode='lines', name='Price Impact'))

# Add vertical line at 50%
fig.add_shape(
    type="line",
    x0=50, y0=0, x1=50, y1=max(impacts),
    line=dict(color="Grey", width=2, dash="dot"),
)

# Add annotation for the 50% line
fig.add_annotation(
    x=50, y=max(impacts),
    text="Balanced Pool",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="Grey",
    ax=50,
    ay=-40
)

fig.update_layout(
    title=f'Price Impact Across Liquidity Compositions<br>Total Pool Size: {total_liquidity:,} Tokens',
    xaxis_title='% of Pool as eclipASTRO',
    yaxis_title=f'% Price Impact on {sale_amount:,} eclipASTRO Sale',
    width=800,
    height=600
)

fig.update_xaxes(range=[0, 100])
fig.update_yaxes(range=[0, max(impacts) * 1.1])

st.plotly_chart(fig)

# Display the price impact at 50% (balanced pool)
balanced_impact = calculate_trade_impact(Decimal(str(initial_liquidity_float / 2)), Decimal(str(initial_liquidity_float / 2)), sale_amount_decimal, A)
st.write(f"Price Impact at balanced state (50/50): {balanced_impact:.4f}%")
