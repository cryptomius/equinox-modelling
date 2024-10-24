import streamlit as st
import numpy as np
import plotly.graph_objs as go
from decimal import Decimal, getcontext

# Set high precision for decimal calculations
getcontext().prec = 30

# Custom CSS to make the content full-width
st.markdown("""
<style>
    .stMainBlockContainer {
        max-width: 100%;
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def calc_y(x, D, A, gamma):
    # PCL invariant function
    return D * D / (x * (4 * A - 4 * A * gamma + gamma * x + 4 * A * gamma * D / x))

def get_D(x0, y0, A, gamma):
    # Calculate D given x0, y0, A, and gamma
    D = (x0 + y0) / 2
    for _ in range(256):
        D_prev = D
        D = (x0 + y0) / 2 + (gamma * x0 * y0 * (x0 + y0)) / (4 * A * D * D)
        if abs(D - D_prev) < 1e-8:
            return D
    raise ValueError("D calculation did not converge")

st.title('Astroport PCL Pool Bonding Curve Simulator')

# Sidebar for inputs
st.sidebar.header('Pool Parameters')
total_liquidity = st.sidebar.slider('Total Pool Size', 1_000_000, 1_000_000_000, 50_000_000, step=1_000_000, format="%d")

st.sidebar.header('PCL Parameters')
A = st.sidebar.slider('Amp (A)', 1, 1000, 500, step=1)
gamma = st.sidebar.slider('Gamma', 0.0, 1.0, 0.01, step=0.01)

# Display initial pool split
half_pool = total_liquidity // 2
st.write(f'Initial xASTRO: {half_pool:,}')
st.write(f'Initial eclipASTRO: {half_pool:,}')

# Calculate D
x0 = y0 = Decimal(total_liquidity) / 2
D = get_D(x0, y0, Decimal(A), Decimal(gamma))

# Generate points for the bonding curve
x_values = np.logspace(np.log10(float(total_liquidity) * 1e-6), np.log10(float(total_liquidity)), 1000)
y_values = np.array([float(calc_y(Decimal(x), D, Decimal(A), Decimal(gamma))) for x in x_values])

# Calculate the price (dy/dx) at each point
prices = -np.diff(y_values) / np.diff(x_values)  # Negative sign to get xASTRO per eclipASTRO
price_x = (x_values[:-1] + x_values[1:]) / 2

# Create the Plotly figure for the bonding curve
fig_bonding = go.Figure()
fig_bonding.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Bonding Curve'))
fig_bonding.update_layout(
    title='PCL Pool Bonding Curve',
    xaxis_title='eclipASTRO Amount',
    yaxis_title='xASTRO Amount',
    width=800,
    height=600,
    xaxis_type="log",
    yaxis_type="log",
    xaxis_range=[np.log10(total_liquidity * 1e-6), np.log10(total_liquidity)],
    yaxis_range=[np.log10(total_liquidity * 1e-6), np.log10(total_liquidity)]
)

# Create the Plotly figure for the price curve
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=price_x, y=prices * 100, mode='lines', name='Price Curve'))
fig_price.update_layout(
    title='PCL Pool Price Curve',
    xaxis_title='eclipASTRO Amount',
    yaxis_title='Price (% of 1 xASTRO per eclipASTRO)',
    width=800,
    height=600,
    xaxis_type="log",
    yaxis_type="log"
)

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

# Plot bonding curve in the left column
with col1:
    st.plotly_chart(fig_bonding, use_container_width=True)

# Plot price curve in the right column
with col2:
    st.plotly_chart(fig_price, use_container_width=True)

# Display some key information
st.write(f"Amplification Parameter (A): {A}")
st.write(f"Gamma: {gamma}")
st.write(f"Invariant (D): {float(D):.2f}")