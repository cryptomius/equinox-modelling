import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

# Set precision for decimal calculations
getcontext().prec = 28

# Clear session state if structure changes
if 'version' not in st.session_state or st.session_state.version != '1.0':
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.version = '1.0'

# Initialize session state
if 'swap_history' not in st.session_state:
    st.session_state.swap_history = []
if 'current_pool' not in st.session_state:
    st.session_state.current_pool = {
        'xastro': Decimal('25000000'),
        'eclipastro': Decimal('25000000')
    }

TOKEN_PRICES = {
    'xastro': Decimal('0.035'),
    'eclipastro': Decimal('0.035')
}

@dataclass
class SwapEvent:
    timestamp: datetime
    token_in: str
    amount_in: float
    token_out: str
    amount_out: float
    price_impact: float
    peg_deviation: float
    pool_composition: dict

def calculate_d(xp: List[Decimal], amp: int) -> Decimal:
    """Calculate the invariant D for stableswap curve"""
    a = Decimal(str(amp))
    n = len(xp)
    s = sum(xp)
    if s == 0:
        return Decimal('0')
    
    d = s
    ann = a * n
    
    for _ in range(255):
        d_prev = d
        d_p = d
        for x in xp:
            d_p = d_p * d / (x * n)
        d = (ann * s + d_p * n) * d / ((ann - 1) * d + (n + 1) * d_p)
        
        if abs(d - d_prev) < Decimal('1'):
            return d
    return d

def get_y(amp: int, x: Decimal, d: Decimal) -> Decimal:
    """
    Calculate the required y balance given x to maintain the invariant D.
    Uses Newton's method with better convergence handling.
    """
    a = Decimal(str(amp))
    n = Decimal('2')  # Number of tokens
    
    # Initial guess for y using simple proportion
    y = d * d / (x * n * n)
    
    for _ in range(255):
        y_prev = y
        
        # Calculate k = D^2 / (4xy)
        k = d * d / (Decimal('4') * x * y)
        
        # Newton iteration: y = (y^2 + c) / (2y + b - D)
        b = (x + d/a)
        c = d * d * d / (n * n * x * a)
        y = (y * y + c) / (Decimal('2') * y + b - d)
        
        # Check for convergence with larger tolerance
        if abs(y - y_prev) < Decimal('0.1'):
            return y
            
    raise Exception("Y calculation did not converge")

def calculate_peg_deviation(pool: Dict[str, Decimal]) -> Decimal:
    """Calculate how far the pool is from 50:50 balance"""
    total = sum(pool.values())
    target = total / Decimal('2')
    
    max_deviation = Decimal('0')
    for amount in pool.values():
        deviation = abs(amount - target) / target * 100
        max_deviation = max(max_deviation, deviation)
    
    return max_deviation

def calculate_pool_composition(pool: Dict[str, Decimal]) -> Dict[str, Dict[str, float]]:
    """Calculate pool composition in both absolute and percentage terms"""
    total = sum(pool.values())
    composition = {}
    
    for token, amount in pool.items():
        composition[token] = {
            "amount": float(amount),
            "percentage": float(amount / total * 100)
        }
    
    return composition

def simulate_swap(token_in: str, token_out: str, amount_in: Decimal, 
                 current_pool: Dict[str, Decimal], amp: int) -> Dict:
    """Simulate a swap and return expected results with improved calculations"""
    try:
        if token_in == token_out:
            raise ValueError("Cannot swap token for itself")
        
        pool_in = current_pool[token_in]
        pool_out = current_pool[token_out]
        
        if amount_in <= 0:
            raise ValueError("Amount must be greater than 0")
            
        # Calculate initial D
        xp = [pool_in, pool_out]
        d = calculate_d(xp, amp)
        
        # Calculate new balances
        new_x = pool_in + amount_in
        new_y = get_y(amp, new_x, d)
        
        # Ensure we don't try to remove more than available
        amount_out = pool_out - new_y
        if amount_out >= pool_out or amount_out <= 0:
            raise ValueError("Invalid swap amount")
        
        # Calculate metrics
        initial_price = pool_out / pool_in
        final_price = new_y / new_x
        price_impact = (initial_price - final_price) / initial_price * Decimal('100')
        exchange_rate = amount_out / amount_in
        
        # Calculate USD value difference
        value_in = amount_in * TOKEN_PRICES[token_in]
        value_out = amount_out * TOKEN_PRICES[token_out]
        value_difference = float(value_in - value_out)
        
        # Create new pool state
        new_pool = current_pool.copy()
        new_pool[token_in] = new_x
        new_pool[token_out] = new_y
        
        peg_deviation = calculate_peg_deviation(new_pool)
        
        return {
            'amount_out': amount_out,
            'price_impact': price_impact,
            'exchange_rate': exchange_rate,
            'value_difference': value_difference,
            'peg_deviation': peg_deviation,
            'new_pool': new_pool
        }
        
    except Exception as e:
        raise Exception(f"Swap simulation error: {str(e)}")

def plot_pool_history():
    """Create two charts: 1) Peg deviation and cumulative volume delta, 2) Pool composition"""
    if not st.session_state.swap_history:
        return None
        
    # Prepare data
    trade_sequence = list(range(len(st.session_state.swap_history) + 1))
    peg_deviations = []
    cumulative_volume_delta = [0]  # Start with 0 for initial state
    xastro_percentages = []
    eclipastro_percentages = []
    
    # Include initial state
    initial_peg = calculate_peg_deviation({
        'xastro': Decimal('25000000'),
        'eclipastro': Decimal('25000000')
    })
    peg_deviations.append(float(initial_peg))
    xastro_percentages.append(50)
    eclipastro_percentages.append(50)
    
    # Add all swap events
    for event in st.session_state.swap_history:
        peg_deviations.append(event.peg_deviation)
        
        # Calculate volume delta
        if event.token_in == 'xastro':
            volume_delta = -event.amount_in  # Negative for xASTRO sells
        else:  # eclipASTRO
            volume_delta = event.amount_in  # Positive for eclipASTRO sells
        
        # Add to cumulative delta
        cumulative_volume_delta.append(cumulative_volume_delta[-1] + volume_delta)
        
        xastro_percentages.append(event.pool_composition['xastro']['percentage'])
        eclipastro_percentages.append(event.pool_composition['eclipastro']['percentage'])
    
    # Create two separate figures
    fig1 = go.Figure()
    fig2 = go.Figure()
    
    # Figure 1: Peg deviation and cumulative volume delta
    # Add cumulative volume delta
    fig1.add_trace(
        go.Scatter(
            x=trade_sequence,
            y=cumulative_volume_delta,
            name='Cumulative Volume Delta',
            line=dict(color='#1e40af', width=2),
            hovertemplate="Cumulative Delta: %{y:,.0f}<br>Trade: %{x}",
            yaxis='y2'
        )
    )
    
    # Add peg deviation area
    fig1.add_trace(
        go.Scatter(
            x=trade_sequence,
            y=peg_deviations,
            name='Peg Deviation',
            fill='tozeroy',
            fillcolor='rgba(147, 51, 234, 0.2)',  # Light purple fill
            line=dict(width=2, color='#9333ea'),  # Purple line for deviation
            hovertemplate="Peg Deviation: %{y:.2f}%<br>Trade: %{x}",
            yaxis='y'
        )
    )
    
    # Update layout for figure 1
    fig1.update_layout(
        title='Peg Deviation and Cumulative Volume Delta',
        height=300,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            title="Peg Deviation %",
            titlefont=dict(color='#9333ea'),
            tickfont=dict(color='#9333ea'),
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
        ),
        yaxis2=dict(
            title="Cumulative Volume Delta",
            titlefont=dict(color='#1e40af'),
            tickfont=dict(color='#1e40af'),
            gridcolor='lightgray',
            overlaying='y',
            side='right'
        ),
        xaxis=dict(title="Trade Sequence", gridcolor='lightgray')
    )
    
    # Figure 2: Pool composition
    colors = {'xastro': '#4f46e5', 'eclipastro': '#06b6d4'}
    fig2.add_trace(
        go.Scatter(
            x=trade_sequence,
            y=xastro_percentages,
            name='xASTRO',
            fill='tozeroy',
            fillcolor=colors['xastro'],
            line=dict(width=0),
            stackgroup='one',
            hovertemplate="xASTRO: %{y:.2f}%<br>Trade: %{x}"
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=trade_sequence,
            y=eclipastro_percentages,
            name='eclipASTRO',
            fill='tonexty',
            fillcolor=colors['eclipastro'],
            line=dict(width=0),
            stackgroup='one',
            hovertemplate="eclipASTRO: %{y:.2f}%<br>Trade: %{x}"
        )
    )
    
    # Update layout for figure 2
    fig2.update_layout(
        title='Pool Composition',
        height=250,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            title="Pool Composition %",
            gridcolor='lightgray',
            range=[0, 100],
        ),
        xaxis=dict(title="Trade Sequence", gridcolor='lightgray')
    )
    
    return fig1, fig2

# Streamlit UI
st.title("Enhanced Stableswap Pool Simulator")

# Pool Configuration
st.header("Pool Configuration")
col1, col2 = st.columns(2)

with col1:
    amp = st.slider("Amplification Parameter (A)", 1, 1000, 100,
                    help="Higher values create a more stable swap curve, reducing price impact")

# Display current pool info
with col2:
    st.write("Current Pool State")
    for token, amount in st.session_state.current_pool.items():
        value = amount * TOKEN_PRICES[token]
        st.write(f"{token}: {float(amount):,.0f} (${float(value):,.2f})")
    current_peg_deviation = calculate_peg_deviation(st.session_state.current_pool)
    st.write(f"Current Peg Deviation: {float(current_peg_deviation):.2f}%")

# Swap Interface
st.header("Swap Simulation")

# Token selection columns
col3, col4 = st.columns(2)

with col3:
    token_in = st.selectbox("Token to Sell", ['eclipASTRO', 'xASTRO'])
    amount_in = st.number_input(
        f"{token_in} Amount",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
        format="%.1f"
    )

with col4:
    # Only show the token that isn't selected as input
    available_tokens = ['eclipASTRO', 'xASTRO']
    available_tokens.remove(token_in)
    token_out = st.selectbox("Token to Buy", available_tokens)

# Preview calculations
if amount_in > 0:
    try:
        preview = simulate_swap(
            token_in.lower(),
            token_out.lower(),
            Decimal(str(amount_in)),
            st.session_state.current_pool,
            amp
        )

        # Display preview results
        st.markdown("### Swap Preview")
        
        # Metrics columns
        col5, col6 = st.columns(2)
        
        with col5:
            st.write("Expected Returns:")
            st.write(f"→ {float(preview['amount_out']):,.2f} {token_out}")
            st.write(f"Exchange Rate: {float(preview['exchange_rate']):,.6f}")
            st.write(f"Price Impact: {float(preview['price_impact']):,.4f}%")
            
        with col6:
            st.write("Impact Analysis:")
            st.write(f"Value Difference: ${float(preview['value_difference']):,.4f}")
            st.write(f"New Peg Deviation: {float(preview['peg_deviation']):,.2f}%")
            deviation_change = float(preview['peg_deviation'] - current_peg_deviation)
            st.write(f"Peg Impact: {deviation_change:+,.2f}%")

        # Execute swap button
        if st.button("Execute Swap"):
            try:
                # Calculate pool composition after swap
                current_composition = calculate_pool_composition(preview['new_pool'])
                
                # Create and store swap event
                event = SwapEvent(
                    timestamp=datetime.now(),
                    token_in=token_in.lower(),
                    amount_in=float(amount_in),
                    token_out=token_out.lower(),
                    amount_out=float(preview['amount_out']),
                    price_impact=float(preview['price_impact']),
                    peg_deviation=float(preview['peg_deviation']),
                    pool_composition=current_composition
                )
                
                # Update state
                st.session_state.swap_history.append(event)
                st.session_state.current_pool = preview['new_pool']
                
                st.success("Swap executed successfully!")
                st.rerun()  # Updated from experimental_rerun
                
            except Exception as e:
                st.error(f"Failed to execute swap: {str(e)}")
                
    except Exception as e:
        st.error(f"Failed to simulate swap: {str(e)}")

# Display pool composition and volume chart
st.header("Pool Analysis")
fig1, fig2 = plot_pool_history()
if fig1 and fig2:
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
# Display swap history
if st.session_state.swap_history:
    st.header("Swap History")
    
    history_data = []
    for event in st.session_state.swap_history:
        history_data.append({
            'Time': event.timestamp,
            'Sold': f"{event.amount_in:,.2f} {event.token_in}",
            'Received': f"{event.amount_out:,.2f} {event.token_out}",
            'Price Impact': f"{event.price_impact:.4f}%",
            'Peg Deviation': f"{event.peg_deviation:.2f}%",
            'Pool Composition': ', '.join(
                f"{token}: {comp['amount']:,.0f} ({comp['percentage']:.2f}%)" 
                for token, comp in event.pool_composition.items()
            )
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(
        history_df.sort_values('Time', ascending=False),
        height=400,
        use_container_width=True
    )

