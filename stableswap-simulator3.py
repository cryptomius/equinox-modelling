import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal, getcontext
from datetime import datetime, timedelta

# Set decimal precision
getcontext().prec = 30

class StableSwapSimulator:
    def __init__(self, initial_xastro, initial_eclip, xastro_price, eclip_price):
        self.initial_xastro = Decimal(str(initial_xastro))
        self.initial_eclip = Decimal(str(initial_eclip))
        self.xastro_price = Decimal(str(xastro_price))
        self.eclip_price = Decimal(str(eclip_price))
        self.catastrophic_event = None
        
    def compute_d(self, amp, xastro, eclip):
        """Calculate the invariant D."""
        n = Decimal('2')  # Number of tokens
        ann = Decimal(str(amp)) * (n ** n)
        s = xastro + eclip
        
        if s == 0:
            return Decimal('0')
            
        d = s
        d_prev = Decimal('0')
        
        for _ in range(256):
            d_prev = d
            k = (xastro * eclip * (n ** n)) / (d ** n)
            d = ((ann * s + n * k * d) * d) / ((ann - Decimal('1')) * d + (n + Decimal('1')) * k * d)
            
            if abs(d - d_prev) <= Decimal('1'):
                break
                
        return d
    
    def calculate_price_impact(self, amount, xastro, eclip, amp, is_selling):
        """Calculate the price impact of a trade."""
        d = self.compute_d(amp, xastro, eclip)
        
        if is_selling:
            new_xastro = xastro - amount
            new_eclip = eclip + amount
        else:
            new_xastro = xastro + amount
            new_eclip = eclip - amount
            
        new_d = self.compute_d(amp, new_xastro, new_eclip)
        
        return (d - new_d) / d
    
    def simulate(self, amp, sell_pressure_pct, total_eclip, unlock_percents, days=220):
        """Run the simulation with given parameters."""
        results = []
        current_xastro = self.initial_xastro
        current_eclip = self.initial_eclip
        self.catastrophic_event = None
        
        # Unlock events (day, percentage)
        unlock_events = [
            (2, unlock_percents[0]),
            (30, unlock_percents[1]),
            (90, unlock_percents[2]),
            (180, unlock_percents[3])
        ]
        
        start_date = datetime.now()
        
        # Initial state
        results.append({
            'day': 0,
            'date': start_date.strftime('%Y-%m-%d'),
            'xastro_price': float(self.xastro_price),
            'eclip_price': float(self.eclip_price),
            'ratio': 1.0,
            'xastro_amount': float(current_xastro),
            'eclip_amount': float(current_eclip)
        })
        
        for day in range(1, days + 1):
            current_date = start_date + timedelta(days=day)
            
            # Check for unlock event
            unlock_event = next((e for e in unlock_events if e[0] == day), None)
            
            if unlock_event:
                unlock_amount = Decimal(str(total_eclip)) * (Decimal(str(unlock_event[1])) / Decimal('100'))
                sell_amount = unlock_amount * (Decimal(str(sell_pressure_pct)) / Decimal('100'))
                
                # Check if we have enough xASTRO
                if sell_amount > current_xastro:
                    if not self.catastrophic_event:
                        self.catastrophic_event = {
                            'day': day,
                            'reason': 'xASTRO Depleted',
                            'xastro_remaining': float(current_xastro),
                            'sell_pressure': float(sell_amount)
                        }
                    sell_amount = current_xastro  # Can only sell what's available
                
                price_impact = self.calculate_price_impact(
                    sell_amount,
                    current_xastro,
                    current_eclip,
                    amp,
                    True
                )
                
                current_xastro -= sell_amount
                current_eclip += sell_amount
                
                new_ratio = float(current_xastro / current_eclip)
                xastro_price = float(self.xastro_price * (1 - price_impact))
                eclip_price = float(self.eclip_price * (1 + price_impact))
                
            else:
                # Normal day - some price recovery if depegged
                last_ratio = results[-1]['ratio']
                if last_ratio != 1:
                    recovery = (1 - last_ratio) * 0.1  # 10% recovery per day
                    new_ratio = last_ratio + recovery
                    xastro_price = results[-1]['xastro_price'] * (1 + recovery)
                    eclip_price = results[-1]['eclip_price'] * (1 - recovery)
                else:
                    new_ratio = last_ratio
                    xastro_price = results[-1]['xastro_price']
                    eclip_price = results[-1]['eclip_price']
            
            results.append({
                'day': day,
                'date': current_date.strftime('%Y-%m-%d'),
                'xastro_price': xastro_price,
                'eclip_price': eclip_price,
                'ratio': new_ratio,
                'xastro_amount': float(current_xastro),
                'eclip_amount': float(current_eclip)
            })
            
        return pd.DataFrame(results)

def plot_simulation_results(df, catastrophic_event=None):
    """Create plots for the simulation results."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Token Prices Over Time', 'Pool Composition'),
        vertical_spacing=0.15
    )
    
    # Add background color for catastrophic events
    if catastrophic_event:
        for i in range(2):
            fig.add_vrect(
                x0=catastrophic_event['day'],
                x1=df['day'].max(),
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
                row=i+1, col=1
            )
    
    # Price plot
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['xastro_price'], name='xASTRO Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['eclip_price'], name='eclipASTRO Price', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['ratio'], name='Price Ratio', line=dict(color='red')),
        row=1, col=1
    )
    
    # Pool composition plot
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['xastro_amount'], name='xASTRO Amount', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['eclip_amount'], name='eclipASTRO Amount', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="StableSwap Pool Simulation Results"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Amount", row=2, col=1)
    
    return fig

def main():
    st.title("xASTRO-eclipASTRO StableSwap Simulator")
    
    # Sidebar - Initial Parameters with configuration
    st.sidebar.header("Initial Pool Configuration")
    
    total_eclip = st.sidebar.number_input(
        "Total eclipASTRO Created",
        min_value=1_000_000,
        max_value=100_000_000,
        value=21_250_000,
        step=100_000,
        format="%d"
    )
    
    initial_xastro = st.sidebar.number_input(
        "Initial xASTRO in Pool",
        min_value=1_000_000,
        max_value=50_000_000,
        value=3_750_000,
        step=100_000,
        format="%d"
    )
    
    initial_eclip = st.sidebar.number_input(
        "Initial eclipASTRO in Pool",
        min_value=1_000_000,
        max_value=50_000_000,
        value=3_750_000,
        step=100_000,
        format="%d"
    )
    
    # Unlock Schedule Configuration
    st.sidebar.header("Unlock Schedule")
    col1, col2 = st.sidebar.columns(2)
    
    unlock_percents = []
    unlock_days = [2, 30, 90, 180]
    
    with col1:
        st.write("Day")
        for day in unlock_days:
            st.write(f"Day {day}")
    
    with col2:
        st.write("Percent")
        for i in range(4):
            unlock_percent = st.number_input(
                f"Unlock {i+1}",
                min_value=0,
                max_value=100,
                value=25,
                key=f"unlock_{i}",
                label_visibility="collapsed"
            )
            unlock_percents.append(unlock_percent)
    
    # Validate total percentage
    total_percent = sum(unlock_percents)
    if total_percent != 100:
        st.sidebar.error(f"⚠️ Total unlock percentage must be 100% (currently {total_percent}%)")
    
    # Display unlock amounts
    st.sidebar.write("Unlock Amounts:")
    unlock_data = pd.DataFrame([
        {
            "Day": day,
            "Amount": f"{total_eclip * (percent/100):,.0f}",
            "Percent": f"{percent}%"
        }
        for day, percent in zip(unlock_days, unlock_percents)
    ])
    st.sidebar.table(unlock_data)
    
    # Simulation Parameters
    st.sidebar.header("Simulation Parameters")
    amp = st.sidebar.slider("Amplification Parameter", 1, 1000, 100)
    sell_pressure = st.sidebar.slider("Sell Pressure (%)", 0, 100, 30)
    
    # Initialize simulator
    simulator = StableSwapSimulator(
        initial_xastro=initial_xastro,
        initial_eclip=initial_eclip,
        xastro_price=0.03457,
        eclip_price=0.02303
    )
    
    # Run simulation only if total percentage is 100%
    if total_percent == 100:
        # Run simulation
        results = simulator.simulate(amp, sell_pressure, total_eclip, unlock_percents)
        
        # Display catastrophic event warning if it occurred
        if simulator.catastrophic_event:
            st.error(f"""🚨 CATASTROPHIC EVENT: {simulator.catastrophic_event['reason']} on Day {simulator.catastrophic_event['day']}
                     \nxASTRO Remaining: {simulator.catastrophic_event['xastro_remaining']:,.2f}
                     \nRequired for Trade: {simulator.catastrophic_event['sell_pressure']:,.2f}""")
        
        # Display results
        st.plotly_chart(plot_simulation_results(results, simulator.catastrophic_event), use_container_width=True)
        
        # Display key metrics
        st.header("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        max_price_deviation = max(abs(1 - results['ratio']))
        recovery_time = len(results[results['ratio'] < 0.95]) if any(results['ratio'] < 0.95) else 0
        final_ratio = results['ratio'].iloc[-1]
        min_xastro = results['xastro_amount'].min()
        
        col1.metric("Max Price Deviation", f"{max_price_deviation:.2%}")
        col2.metric("Recovery Time (days)", recovery_time)
        col3.metric("Final Price Ratio", f"{final_ratio:.3f}")
        col4.metric("Minimum xASTRO", f"{min_xastro:,.0f}")
        
        # Show raw data
        if st.checkbox("Show raw data"):
            st.dataframe(results)
    else:
        st.warning("Please adjust unlock percentages to total 100% before running simulation")

if __name__ == "__main__":
    main()
