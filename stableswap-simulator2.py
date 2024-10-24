import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from decimal import Decimal

def calculate_d(amounts, amp):
    """Calculate the invariant D."""
    sum_x = sum(amounts)
    if sum_x == 0:
        return 0
    
    d = sum_x
    ann = amp * len(amounts)
    
    for _ in range(255):
        d_prev = d
        k = d ** len(amounts) // (len(amounts) ** len(amounts))
        for x in amounts:
            k = k * d // (x * len(amounts))
        d = ((ann * sum_x + k * len(amounts)) * d) // ((ann - 1) * d + (len(amounts) + 1) * k)
        if abs(d - d_prev) <= 1:
            return d
    
    raise ValueError("D calculation did not converge")

def get_y(amp, x, d):
    """Calculate y given x in the StableSwap invariant."""
    c = d ** 3 // (x * 4)
    b = x + d // (amp * 2)
    
    y = d
    for _ in range(255):
        y_prev = y
        y = (y ** 2 + c) // (2 * y + b - d)
        if abs(y - y_prev) <= 1:
            return y
    
    raise ValueError("Y calculation did not converge")

def simulate_sell_pressure(initial_balance, sell_amount, amp):
    """Simulate sell pressure and calculate resulting peg deviation"""
    d = calculate_d([initial_balance, initial_balance], amp)
    new_x = initial_balance + sell_amount
    new_y = get_y(amp, new_x, d)
    peg_deviation = abs(new_y - new_x) / initial_balance * 100
    return peg_deviation, new_x, new_y

def simulate_buy_pressure(initial_x, initial_y, target_deviation, amp):
    """Simulate buy pressure needed to restore peg within target deviation"""
    d = calculate_d([initial_x, initial_y], amp)
    target_balance = (initial_x + initial_y) / 2
    
    def deviation(buy_amount):
        new_y = initial_y - buy_amount
        new_x = get_y(amp, new_y, d)
        return abs(new_x - new_y) / target_balance * 100 - target_deviation
    
    # Binary search to find required buy amount
    low, high = 0, initial_y
    while high - low > 1:
        mid = (low + high) / 2
        if deviation(mid) > 0:
            low = mid
        else:
            high = mid
    
    return high

def analyze_pool_dynamics(pool_depths, sell_pressure, amp_values, target_deviation):
    results = []
    for depth in pool_depths:
        for amp in amp_values:
            peg_dev, new_x, new_y = simulate_sell_pressure(depth, sell_pressure, amp)
            buy_pressure = simulate_buy_pressure(new_x, new_y, target_deviation, amp)
            results.append({
                'Pool Depth': depth,
                'Amp': amp,
                'Sell Pressure': sell_pressure,
                'Resulting Peg Deviation': peg_dev,
                'Buy Pressure to Restore': buy_pressure
            })
    return pd.DataFrame(results)

def find_min_sell_pressure(depth, amp, target_deviation):
    low, high = 0, depth
    while high - low > 1:
        mid = (low + high) / 2
        peg_dev, _, _ = simulate_sell_pressure(depth, mid, amp)
        if peg_dev < target_deviation:
            low = mid
        else:
            high = mid
    return high

st.title("Stableswap Pool Dynamics Simulator")

# Input widgets
st.sidebar.header("Simulation Parameters")

min_depth = st.sidebar.number_input("Minimum Pool Depth", value=10_000_000, step=1_000_000)
max_depth = st.sidebar.number_input("Maximum Pool Depth", value=50_000_000, step=1_000_000)
depth_steps = st.sidebar.number_input("Number of Pool Depth Steps", value=5, min_value=2, step=1)

sell_pressure = st.sidebar.number_input("Sell Pressure", value=20_000_000, step=1_000_000)

min_amp = st.sidebar.number_input("Minimum Amp Value", value=50, step=10)
max_amp = st.sidebar.number_input("Maximum Amp Value", value=1000, step=10)
amp_steps = st.sidebar.number_input("Number of Amp Value Steps", value=5, min_value=2, step=1)

target_deviation = st.sidebar.number_input("Acceptable Peg Deviation (%)", value=1.0, step=0.1)

# Generate parameter ranges
pool_depths = np.linspace(min_depth, max_depth, depth_steps)
amp_values = np.linspace(min_amp, max_amp, amp_steps, dtype=int)

# Run analysis
results = analyze_pool_dynamics(pool_depths, sell_pressure, amp_values, target_deviation)

# Visualize results
st.header("Peg Deviation vs Pool Depth for Different Amp Values")
fig = go.Figure()

for amp in amp_values:
    df = results[results['Amp'] == amp]
    fig.add_trace(go.Scatter(
        x=df['Pool Depth'],
        y=df['Resulting Peg Deviation'],
        mode='lines+markers',
        name=f'Amp = {amp}'
    ))

fig.update_layout(
    xaxis_title='Pool Depth',
    yaxis_title='Resulting Peg Deviation (%)',
    legend_title='Amp Value'
)

st.plotly_chart(fig)

# Display table of results
st.header("Detailed Results")
st.dataframe(results)

# Find minimum sell pressure to exceed target deviation
st.header("Minimum Sell Pressure to Exceed Target Deviation")
min_sell_pressures = []
for depth in pool_depths:
    for amp in amp_values:
        min_sell = find_min_sell_pressure(depth, amp, target_deviation)
        min_sell_pressures.append({
            'Pool Depth': depth,
            'Amp': amp,
            'Min Sell Pressure': min_sell
        })

min_sell_df = pd.DataFrame(min_sell_pressures)
st.dataframe(min_sell_df)

# Visualize minimum sell pressure
st.header("Minimum Sell Pressure vs Pool Depth for Different Amp Values")
fig_min_sell = go.Figure()

for amp in amp_values:
    df = min_sell_df[min_sell_df['Amp'] == amp]
    fig_min_sell.add_trace(go.Scatter(
        x=df['Pool Depth'],
        y=df['Min Sell Pressure'],
        mode='lines+markers',
        name=f'Amp = {amp}'
    ))

fig_min_sell.update_layout(
    xaxis_title='Pool Depth',
    yaxis_title='Minimum Sell Pressure',
    legend_title='Amp Value'
)

st.plotly_chart(fig_min_sell)

# Buy pressure analysis
st.header("Buy Pressure Required to Restore Peg")
buy_pressure_df = results[['Pool Depth', 'Amp', 'Buy Pressure to Restore']]
st.dataframe(buy_pressure_df)

# Visualize buy pressure
st.header("Buy Pressure vs Pool Depth for Different Amp Values")
fig_buy = go.Figure()

for amp in amp_values:
    df = buy_pressure_df[buy_pressure_df['Amp'] == amp]
    fig_buy.add_trace(go.Scatter(
        x=df['Pool Depth'],
        y=df['Buy Pressure to Restore'],
        mode='lines+markers',
        name=f'Amp = {amp}'
    ))

fig_buy.update_layout(
    xaxis_title='Pool Depth',
    yaxis_title='Buy Pressure to Restore',
    legend_title='Amp Value'
)

st.plotly_chart(fig_buy)
