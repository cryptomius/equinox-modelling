import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal, getcontext, InvalidOperation, Overflow, DivisionByZero
from datetime import datetime, timedelta
import sys

getcontext().prec = 30

class ManagedStableSwapSimulator:
    def __init__(self, initial_xastro, initial_eclip, xastro_price, eclip_price):
        self.initial_xastro = Decimal(str(initial_xastro))
        self.initial_eclip = Decimal(str(initial_eclip))
        self.xastro_price = Decimal(str(xastro_price))
        self.eclip_price = Decimal(str(eclip_price))
        self.catastrophic_event = None
        self.AMP_PRECISION = Decimal('100')  # Added AMP_PRECISION constant
        
    def compute_d(self, amp, xastro, eclip):
        """Calculate the invariant D."""
        n = Decimal('2')
        # Apply AMP_PRECISION scaling like in the contract
        leverage = (Decimal(str(amp)) / self.AMP_PRECISION) * n
        s = xastro + eclip
        
        if s == 0:
            return Decimal('0')
            
        d = s
        d_prev = Decimal('0')
        
        for _ in range(64):
            d_prev = d
            try:
                k = (xastro * eclip * (n ** n)) / (d ** n)
                d = ((leverage * s + n * k * d) * d) / ((leverage - Decimal('1')) * d + (n + Decimal('1')) * k * d)
                
                if abs(d - d_prev) <= Decimal('1'):
                    break
            except (OverflowError, InvalidOperation, Overflow, DivisionByZero):
                # If any calculation error occurs, return the maximum possible Decimal value
                return Decimal(sys.float_info.max)
                
        return d
    
    def calculate_price_impact(self, amount, xastro, eclip, amp, is_selling):
        """Calculate the price impact of a trade."""
        try:
            d = self.compute_d(amp, xastro, eclip)
            
            if is_selling:
                new_xastro = xastro - amount
                new_eclip = eclip + amount
            else:
                new_xastro = xastro + amount
                new_eclip = eclip - amount
                
            new_d = self.compute_d(amp, new_xastro, new_eclip)
            
            return (d - new_d) / d
        except (OverflowError, InvalidOperation, Overflow, DivisionByZero):
            # If any calculation error occurs, return the maximum possible impact (100%)
            return Decimal('1')
    
    def should_intervene(self, price_ratio, intervention_threshold):
        """Determine if liquidity intervention is needed."""
        return abs(1 - price_ratio) > intervention_threshold
    
    def calculate_intervention_amount(self, current_xastro, current_eclip, target_ratio=1):
        """Calculate how much liquidity to add to rebalance the pool."""
        current_ratio = float(current_xastro / current_eclip)
        if current_ratio < target_ratio:
            # Need more xASTRO
            xastro_needed = (current_eclip * Decimal(str(target_ratio))) - current_xastro
            return (xastro_needed, Decimal('0'))
        else:
            # Need more eclipASTRO
            eclip_needed = (current_xastro / Decimal(str(target_ratio))) - current_eclip
            return (Decimal('0'), eclip_needed)

    def simulate(self, amp, sell_pressure_pct, total_eclip, unlock_percents, 
                intervention_threshold, intervention_size, days=220, proactive=False):
        """Run the simulation with liquidity management."""
        results = []
        interventions = []
        current_xastro = self.initial_xastro
        current_eclip = self.initial_eclip
        total_interventions = 0
        intervention_volume = Decimal('0')
        self.catastrophic_event = None
        self.proactive = proactive
        
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
            'eclip_amount': float(current_eclip),
            'intervention': 0
        })
        
        for day in range(1, days + 1):
            current_date = start_date + timedelta(days=day)
            intervention_made = False
            xastro_added = Decimal('0')
            eclip_added = Decimal('0')
            
            # Check for unlock event
            unlock_event = next((e for e in unlock_events if e[0] == day), None)
            
            if unlock_event:
                unlock_amount = Decimal(str(total_eclip)) * (Decimal(str(unlock_event[1])) / Decimal('100'))
                sell_amount = unlock_amount * (Decimal(str(sell_pressure_pct)) / Decimal('100'))
                
                if self.proactive:
                    # Proactive intervention before the unlock
                    xastro_add, eclip_add = self.calculate_intervention_amount(
                        current_xastro, 
                        current_eclip + sell_amount,
                        target_ratio=1
                    )
                    
                    # Apply intervention with size limit
                    max_intervention = Decimal(str(intervention_size * float(current_xastro)))
                    xastro_add = min(xastro_add, max_intervention)
                    eclip_add = min(eclip_add, max_intervention)
                    
                    if xastro_add > 0 or eclip_add > 0:
                        current_xastro += xastro_add
                        current_eclip += eclip_add
                        intervention_volume += xastro_add + eclip_add
                        total_interventions += 1
                        intervention_made = True
                        xastro_added = xastro_add
                        eclip_added = eclip_add
                
                # Process sell order
                if sell_amount > current_xastro:
                    if not self.catastrophic_event:
                        self.catastrophic_event = {
                            'day': day,
                            'reason': 'xASTRO Depleted',
                            'xastro_remaining': float(current_xastro),
                            'sell_pressure': float(sell_amount)
                        }
                    sell_amount = current_xastro
                
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
                # Check for intervention on non-unlock days (reactive mode)
                if not self.proactive:
                    current_ratio = current_xastro / current_eclip
                    if self.should_intervene(float(current_ratio), intervention_threshold):
                        xastro_add, eclip_add = self.calculate_intervention_amount(
                            current_xastro, 
                            current_eclip,
                            target_ratio=1
                        )
                        
                        # Apply intervention with size limit
                        max_intervention = Decimal(str(intervention_size * float(current_xastro)))
                        xastro_add = min(xastro_add, max_intervention)
                        eclip_add = min(eclip_add, max_intervention)
                        
                        if xastro_add > 0 or eclip_add > 0:
                            current_xastro += xastro_add
                            current_eclip += eclip_add
                            intervention_volume += xastro_add + eclip_add
                            total_interventions += 1
                            intervention_made = True
                            xastro_added = xastro_add
                            eclip_added = eclip_add
                
                last_ratio = results[-1]['ratio']
                if last_ratio != 1:
                    recovery = (1 - last_ratio) * 0.1
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
                'eclip_amount': float(current_eclip),
                'intervention': 1 if intervention_made else 0
            })
            
            if intervention_made:
                interventions.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'xastro_added': float(xastro_added),
                    'eclip_added': float(eclip_added)
                })
            
        return pd.DataFrame(results), total_interventions, float(intervention_volume), pd.DataFrame(interventions)
    
def plot_simulation_results(df, catastrophic_event=None, interventions=None):
    """Create a three-panel plot showing prices, pool composition, and interventions."""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Token Prices Over Time',
            'Pool Composition (100% Stacked)',
            'Interventions',
            'Intervention Quantities'
        ),
        vertical_spacing=0.1,
        row_heights=[0.3, 0.3, 0.2, 0.2]
    )
    
    # Add red background shading if catastrophic event occurred
    if catastrophic_event:
        for i in range(3):
            fig.add_vrect(
                x0=catastrophic_event['day'],
                x1=df['day'].max(),
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
                row=i+1, col=1
            )
    
    # Price and ratio plot (top panel)
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['xastro_price'], 
            name='xASTRO Price', 
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['eclip_price'], 
            name='eclipASTRO Price', 
            line=dict(color='green')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['ratio'], 
            name='Price Ratio', 
            line=dict(color='red', dash='dot')
        ),
        row=1, col=1
    )
    
    # Add horizontal line at ratio = 1
    fig.add_hline(
        y=1, 
        line_dash="dash", 
        line_color="gray", 
        row=1, 
        col=1,
        annotation_text="Target Ratio"
    )
    
    # Pool composition plot (100% stacked area chart)
    total_amounts = df['xastro_amount'] + df['eclip_amount']
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['xastro_amount'] / total_amounts * 100, 
            name='xASTRO %', 
            line=dict(color='blue'),
            stackgroup='one'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['eclip_amount'] / total_amounts * 100, 
            name='eclipASTRO %', 
            line=dict(color='green'),
            stackgroup='one'
        ),
        row=2, col=1
    )
    
    # Interventions plot (bottom panel)
    fig.add_trace(
        go.Bar(
            x=df['date'], 
            y=df['intervention'], 
            name='Interventions',
            marker_color='purple'
        ),
        row=3, col=1
    )
    
    # Intervention quantities plot
    if interventions is not None:
        fig.add_trace(
            go.Bar(
                x=interventions['date'],
                y=interventions['xastro_added'],
                name='xASTRO Added',
                marker_color='blue'
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(
                x=interventions['date'],
                y=interventions['eclip_added'],
                name='eclipASTRO Added',
                marker_color='green'
            ),
            row=4, col=1
        )
    
    # Update layout and labels
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="StableSwap Pool Simulation Results with Liquidity Management",
        hovermode='x unified'
    )
    
    # Adjust the y-domain of each subplot
    fig.update_layout(
        yaxis=dict(domain=[0.75, 1]),
        yaxis2=dict(domain=[0.5, 0.7]),
        yaxis3=dict(domain=[0.3, 0.45]),
        yaxis4=dict(domain=[0, 0.25])
    )
    
    # Add axis labels
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Composition (%)", row=2, col=1)
    fig.update_yaxes(title_text="Intervention", row=3, col=1)
    fig.update_yaxes(title_text="Quantity Added", row=4, col=1)
    
    # Add mouseover tooltips
    fig.update_traces(
        hovertemplate="<br>".join([
            "Date: %{x}",
            "Value: %{y:.6f}",
            "<extra></extra>"
        ])
    )
    
    return fig

def plot_metrics_summary(df, total_interventions, intervention_volume):
    """Create a summary metrics visualization."""
    metrics_fig = go.Figure()
    
    # Calculate key metrics
    max_deviation = max(abs(1 - df['ratio']))
    min_ratio = min(df['ratio'])
    max_ratio = max(df['ratio'])
    avg_ratio = df['ratio'].mean()
    recovery_periods = len(df[df['ratio'] < 0.95])
    
    # Create metrics table
    metrics_fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    ['Maximum Deviation', 'Minimum Ratio', 'Maximum Ratio', 
                     'Average Ratio', 'Recovery Periods', 'Total Interventions',
                     'Intervention Volume'],
                    [f"{max_deviation:.2%}", f"{min_ratio:.3f}", f"{max_ratio:.3f}",
                     f"{avg_ratio:.3f}", f"{recovery_periods} days", 
                     f"{total_interventions}", f"{intervention_volume:,.0f}"]
                ],
                align='left'
            )
        )
    )
    
    metrics_fig.update_layout(
        height=300,
        title_text="Simulation Metrics Summary"
    )
    
    return metrics_fig

def plot_daily_stats(df):
    """Create daily statistics visualization."""
    daily_fig = go.Figure()
    
    # Calculate daily statistics
    daily_stats = pd.DataFrame({
        'date': df['date'],
        'price_volatility': df['ratio'].rolling(window=7).std(),
        'intervention_frequency': df['intervention'].rolling(window=7).mean(),
    })
    
    # Add traces
    daily_fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['price_volatility'],
            name='7-Day Price Volatility'
        )
    )
    
    daily_fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['intervention_frequency'],
            name='7-Day Intervention Frequency',
            yaxis='y2'
        )
    )
    
    # Update layout
    daily_fig.update_layout(
        height=400,
        title_text="Daily Statistics",
        yaxis=dict(title="Price Volatility"),
        yaxis2=dict(title="Intervention Frequency", overlaying='y', side='right')
    )
    
    return daily_fig

def plot_amp_comparison(initial_xastro, initial_eclip, sell_size_range):
    """Create visualization comparing price impact across different Amp values."""
    amp_values = [50, 100, 200, 500]
    
    # Create simulator instance
    simulator = ManagedStableSwapSimulator(
        initial_xastro=initial_xastro,
        initial_eclip=initial_eclip,
        xastro_price=0.03457,
        eclip_price=0.02303
    )
    
    # Calculate price impacts for different sell sizes and Amp values
    sell_sizes = np.linspace(0, sell_size_range, 100)
    price_impacts = {}
    
    for amp in amp_values:
        impacts = []
        for sell_size in sell_sizes:
            try:
                impact = simulator.calculate_price_impact(
                    Decimal(str(sell_size)),
                    Decimal(str(initial_xastro)),
                    Decimal(str(initial_eclip)),
                    amp,
                    True
                )
                impacts.append(min(float(impact) * 100, 100))  # Convert to percentage, cap at 100%
            except OverflowError:
                impacts.append(100)  # Set to 100% if overflow occurs
        price_impacts[amp] = impacts
    
    # Create comparison plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Price Impact vs. Sell Size',
            'Price Impact Difference from Base Case (Amp=100)'
        ),
        vertical_spacing=0.15
    )
    
    # Price impact curves
    for amp in amp_values:
        fig.add_trace(
            go.Scatter(
                x=sell_sizes,
                y=price_impacts[amp],
                name=f'Amp = {amp}',
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Add difference from base case (Amp=100)
        if amp != 100:
            diff = np.array(price_impacts[amp]) - np.array(price_impacts[100])
            fig.add_trace(
                go.Scatter(
                    x=sell_sizes,
                    y=diff,
                    name=f'Amp {amp} vs 100',
                    mode='lines'
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Amp Parameter Impact Analysis",
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Sell Size", row=2, col=1)
    fig.update_yaxes(title_text="Price Impact (%)", row=1, col=1)
    fig.update_yaxes(title_text="Difference in Impact (%)", row=2, col=1)
    
    return fig

def add_amp_comparison_section():
    """Add Amp comparison section to the main UI."""
    st.header("Amp Parameter Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compare_initial_xastro = st.number_input(
            "Initial xASTRO for Comparison",
            min_value=1_000_000,
            max_value=50_000_000,
            value=3_750_000,
            step=100_000,
            format="%d"
        )
        
    with col2:
        compare_initial_eclip = st.number_input(
            "Initial eclipASTRO for Comparison",
            min_value=1_000_000,
            max_value=50_000_000,
            value=3_750_000,
            step=100_000,
            format="%d"
        )
    
    sell_size_range = st.slider(
        "Maximum Sell Size for Comparison",
        min_value=100_000,
        max_value=10_000_000,
        value=1_000_000,
        step=100_000,
        format="%d",
        help="Maximum sell size to analyze for price impact"
    )
    
    comparison_plot = plot_amp_comparison(
        compare_initial_xastro,
        compare_initial_eclip,
        sell_size_range
    )
    
    st.plotly_chart(comparison_plot, use_container_width=True)
    
    # Add key observations
    st.write("""
    ### Key Observations:
    1. Higher Amp values result in lower price impact for the same sell size
    2. The relationship between Amp and price impact is non-linear
    3. The effect of Amp is most pronounced during larger trades
    4. Diminishing returns on stability as Amp increases
    """)
    
    # Add specific examples
    col3, col4, col5 = st.columns(3)
    
    simulator = ManagedStableSwapSimulator(
        initial_xastro=compare_initial_xastro,
        initial_eclip=compare_initial_eclip,
        xastro_price=0.03457,
        eclip_price=0.02303
    )
    
    test_sell = Decimal(str(sell_size_range * 0.2))  # 20% of max sell size
    
    with col3:
        impact_50 = float(simulator.calculate_price_impact(
            test_sell,
            Decimal(str(compare_initial_xastro)),
            Decimal(str(compare_initial_eclip)),
            50,
            True
        )) * 100
        st.metric("Price Impact at Amp=50", f"{impact_50:.2f}%")
        
    with col4:
        impact_100 = float(simulator.calculate_price_impact(
            test_sell,
            Decimal(str(compare_initial_xastro)),
            Decimal(str(compare_initial_eclip)),
            100,
            True
        )) * 100
        st.metric("Price Impact at Amp=100", f"{impact_100:.2f}%")
        
    with col5:
        impact_200 = float(simulator.calculate_price_impact(
            test_sell,
            Decimal(str(compare_initial_xastro)),
            Decimal(str(compare_initial_eclip)),
            200,
            True
        )) * 100
        st.metric("Price Impact at Amp=200", f"{impact_200:.2f}%")

def main():
    st.title("Managed xASTRO-eclipASTRO StableSwap Simulator")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Initial Pool Setup
    st.sidebar.header("Initial Pool Setup")
    total_eclip = st.sidebar.number_input(
        "Total eclipASTRO Created",
        min_value=1_000_000,
        max_value=100_000_000,
        value=21_250_000,
        step=100_000,
        format="%d",
        help="Total amount of eclipASTRO tokens that will be created"
    )
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        initial_xastro = st.number_input(
            "Initial xASTRO",
            min_value=1_000_000,
            max_value=50_000_000,
            value=3_750_000,
            step=100_000,
            format="%d",
            help="Initial amount of xASTRO in the pool"
        )
    
    with col2:
        initial_eclip = st.number_input(
            "Initial eclipASTRO",
            min_value=1_000_000,
            max_value=50_000_000,
            value=3_750_000,
            step=100_000,
            format="%d",
            help="Initial amount of eclipASTRO in the pool"
        )
    
    # Pool Parameters
    st.sidebar.header("Pool Parameters")
    amp = st.sidebar.number_input(
        "Amplification Parameter",
        min_value=1,
        max_value=1000,
        value=100,
        help="Higher values lead to lower price impact but harder arbitrage"
    )
    
    # Unlock Schedule
    st.sidebar.header("Unlock Schedule")
    unlock_percents = []
    unlock_days = [2, 30, 90, 180]
    
    for i, day in enumerate(unlock_days):
        unlock_percent = st.sidebar.number_input(
            f"Day {day} Unlock %",
            min_value=0,
            max_value=100,
            value=25,
            help=f"Percentage of total eclipASTRO to unlock on day {day}"
        )
        unlock_percents.append(unlock_percent)
    
    # Validate total percentage
    total_percent = sum(unlock_percents)
    if total_percent != 100:
        st.sidebar.error(f"⚠️ Total unlock percentage must be 100% (currently {total_percent}%)")
    
    # Market Pressure
    st.sidebar.header("Market Pressure")
    sell_pressure = st.sidebar.slider(
        "Sell Pressure %",
        min_value=0,
        max_value=100,
        value=30,
        help="Percentage of unlocked tokens that will be sold"
    )
    
    # Intervention Settings
    st.sidebar.header("Intervention Settings")
    intervention_threshold = st.sidebar.slider(
        "Intervention Threshold %",
        min_value=1,
        max_value=20,
        value=5,
        help="Price deviation that triggers intervention"
    ) / 100
    
    intervention_size = st.sidebar.slider(
        "Max Intervention Size %",
        min_value=1,
        max_value=50,
        value=10,
        help="Maximum size of each intervention as % of pool"
    ) / 100
    
    # Add proactive toggle
    proactive = st.sidebar.checkbox("Proactive Liquidity Management", value=False)
    
    # Main content area
    st.write("""
    ## Simulation Overview
    This simulator models the behavior of a managed StableSwap pool with active liquidity management.
    It includes unlock events, market pressure, and strategic interventions.
    """)
    
    # Initialize simulator
    simulator = ManagedStableSwapSimulator(
        initial_xastro=initial_xastro,
        initial_eclip=initial_eclip,
        xastro_price=0.03457,
        eclip_price=0.02303
    )
    
    # Only run simulation if unlock percentages total 100%
    if total_percent == 100:
        # Run simulation
        results, total_interventions, intervention_volume, interventions = simulator.simulate(
            amp=amp,
            sell_pressure_pct=sell_pressure,
            total_eclip=total_eclip,
            unlock_percents=unlock_percents,
            intervention_threshold=intervention_threshold,
            intervention_size=intervention_size,
            proactive=proactive
        )
        
        # Display catastrophic event warning if it occurred
        if simulator.catastrophic_event:
            st.error(f"""
            🚨 CATASTROPHIC EVENT DETECTED
            
            Type: {simulator.catastrophic_event['reason']}
            Day: {simulator.catastrophic_event['day']}
            xASTRO Remaining: {simulator.catastrophic_event['xastro_remaining']:,.2f}
            Sell Pressure: {simulator.catastrophic_event['sell_pressure']:,.2f}
            """)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Main Results", "Metrics", "Daily Stats", "Amp Analysis"])
        
        with tab1:
            st.plotly_chart(
                plot_simulation_results(results, simulator.catastrophic_event, interventions),
                use_container_width=True
            )
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Max Price Deviation",
                    f"{max(abs(1 - results['ratio'])):.2%}"
                )
            with col2:
                st.metric(
                    "Total Interventions",
                    f"{total_interventions}"
                )
            with col3:
                st.metric(
                    "Final Price Ratio",
                    f"{results['ratio'].iloc[-1]:.3f}"
                )
            with col4:
                st.metric(
                    "Min xASTRO Balance",
                    f"{min(results['xastro_amount']):,.0f}"
                )
        
        with tab2:
            st.plotly_chart(
                plot_metrics_summary(results, total_interventions, intervention_volume),
                use_container_width=True
            )
        
        with tab3:
            st.plotly_chart(
                plot_daily_stats(results),
                use_container_width=True
            )
        
        with tab4:
            add_amp_comparison_section()
        
        # Option to show raw data
        if st.checkbox("Show raw simulation data"):
            st.dataframe(results)
            st.dataframe(interventions)
            
            # Option to download data
            csv = results.to_csv(index=False)
            st.download_button(
                "Download simulation results",
                csv,
                "simulation_results.csv",
                "text/csv",
                key='download-csv-results'
            )
            
            csv_interventions = interventions.to_csv(index=False)
            st.download_button(
                "Download intervention data",
                csv_interventions,
                "intervention_data.csv",
                "text/csv",
                key='download-csv-interventions'
            )
    
    else:
        st.warning("Please adjust unlock percentages to total 100% before running simulation")

if __name__ == "__main__":
    main()
