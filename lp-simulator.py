import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
POOL_COLORS = {
    'dynamic_pcl': '#1f77b4',  # Blue
    'fixed_pcl': '#2ca02c',    # Green
    'stableswap': '#d62728'    # Red
}

INITIAL_PARAMS = {
    'pcl_dynamic': {
        'mid_fee': 0.0002,  # 0.02%
        'out_fee': 0.0075,  # 0.75%
        'gamma': 0.0002,
        'fee_gamma': 0.00002
    },
    'pcl_fixed': {
        'mid_fee': 0.0002,
        'out_fee': 0.0075,
        'gamma': 0.0002,
        'fee_gamma': 0.00002
    },
    'stableswap': {
        'fee': 0.0004  # 0.04%
    }
}

# Core calculation functions
def calculate_pcl_price(x_balance, e_balance, amp, gamma):
    """
    Calculate PCL pool price given current balances and parameters
    Price = (x/y)^(1/A) where A is the amplification parameter
    """
    if x_balance <= 0 or e_balance <= 0:
        return 0
    try:
        return (x_balance / e_balance) ** (1 / amp)
    except:
        return 0

def calculate_stableswap_price(x_balance, e_balance, amp):
    """
    Calculate StableSwap pool price given current balances and parameters
    Similar to PCL but without gamma factor
    """
    if x_balance <= 0 or e_balance <= 0:
        return 0
    try:
        return (x_balance / e_balance) ** (1 / amp)
    except:
        return 0

def adjust_pcl_parameters(imbalance, current_params):
    """Adjust PCL parameters based on pool imbalance"""
    new_params = current_params.copy()
    
    if imbalance > 0.03:  # 3% imbalance threshold
        new_params['amp'] = max(50, current_params['amp'] * 0.9)
        new_params['out_fee'] = min(0.015, current_params['out_fee'] * 1.2)
        return new_params, True, f"High imbalance ({imbalance:.2%}) detected - Reducing amp to {new_params['amp']:.0f}, increasing out_fee to {new_params['out_fee']:.4f}"
    
    return new_params, False, ""

def calculate_rebalancing_opportunity_value(x_balance, e_balance, x_price_usd, e_price_usd, amp):
    """Calculate the value of rebalancing opportunity"""
    current_ratio = x_balance / (x_balance + e_balance)
    target_ratio = 0.5
    imbalance = abs(target_ratio - current_ratio)
    
    if imbalance < 0.001:  # Less than 0.1% imbalance
        return 0
    
    total_pool_value = (x_balance * x_price_usd) + (e_balance * e_price_usd)
    opportunity_value = total_pool_value * imbalance * (1 / amp)
    return opportunity_value

def calculate_rebalancing_profit(x_balance, e_balance, x_price_usd, e_price_usd, amp, pool_type):
    """
    Calculate the profit opportunity from rebalancing the pool
    Returns the profit and the optimal trade size
    """
    # Calculate pool price vs market price
    pool_ratio = x_balance / e_balance
    market_ratio = 1.0  # Assuming 1:1 target ratio
    
    # If pool_ratio < market_ratio, xASTRO is scarce in pool
    # If pool_ratio > market_ratio, eclipASTRO is scarce in pool
    
    if abs(pool_ratio - market_ratio) < 0.001:
        return {
            'profit_usd': 0,
            'trade_size': 0,
            'direction': None,
            'steps': []
        }

    # Calculate optimal trade size (this could be optimized further)
    total_pool_tokens = x_balance + e_balance
    imbalance = abs(0.5 - (x_balance / total_pool_tokens))
    trade_size = total_pool_tokens * imbalance * 0.1  # Start with 10% of imbalance

    steps = []
    
    if pool_ratio < market_ratio:
        # Need to add xASTRO to pool
        # 1. Buy xASTRO at market
        cost = trade_size * x_price_usd
        steps.append(f"Buy {trade_size:,.0f} xASTRO at market price (${x_price_usd:.5f}) = ${cost:,.2f}")
        
        # 2. Add to pool, get eclipASTRO out
        # Pool price will be higher than market, so we get more eclipASTRO out
        eclip_received = trade_size * (1 + imbalance)  # Simplified price impact
        steps.append(f"Add {trade_size:,.0f} xASTRO to pool, receive {eclip_received:,.0f} eclipASTRO")
        
        # 3. Sell received eclipASTRO at market
        revenue = eclip_received * e_price_usd
        steps.append(f"Sell {eclip_received:,.0f} eclipASTRO at market price (${e_price_usd:.5f}) = ${revenue:,.2f}")
        
        profit = revenue - cost
        
    else:
        # Need to add eclipASTRO to pool
        # 1. Buy eclipASTRO at market
        cost = trade_size * e_price_usd
        steps.append(f"Buy {trade_size:,.0f} eclipASTRO at market price (${e_price_usd:.5f}) = ${cost:,.2f}")
        
        # 2. Add to pool, get xASTRO out
        xastro_received = trade_size * (1 + imbalance)  # Simplified price impact
        steps.append(f"Add {trade_size:,.0f} eclipASTRO to pool, receive {xastro_received:,.0f} xASTRO")
        
        # 3. Sell received xASTRO at market
        revenue = xastro_received * x_price_usd
        steps.append(f"Sell {xastro_received:,.0f} xASTRO at market price (${x_price_usd:.5f}) = ${revenue:,.2f}")
        
        profit = revenue - cost

    steps.append(f"Net profit: ${profit:,.2f}")
    
    # Factor in pool fees
    if pool_type.startswith('pcl'):
        fee = INITIAL_PARAMS['pcl_dynamic']['out_fee']
    else:
        fee = INITIAL_PARAMS['stableswap']['fee']
    
    profit_after_fees = profit * (1 - fee)
    steps.append(f"Profit after {fee*100:.2f}% pool fee: ${profit_after_fees:,.2f}")

    return {
        'profit_usd': profit_after_fees,
        'trade_size': trade_size,
        'direction': 'add_xastro' if pool_ratio < market_ratio else 'add_eclip',
        'steps': steps
    }

# In the main UI, add a section to show the rebalancing opportunity details:
def show_rebalancing_details(x_balance, e_balance, x_price_usd, e_price_usd, amp, pool_type):
    opportunity = calculate_rebalancing_profit(x_balance, e_balance, x_price_usd, e_price_usd, amp, pool_type)
    
    if opportunity['profit_usd'] > 0:
        st.subheader(f"Rebalancing Opportunity for {pool_type}")
        st.write("Step-by-step profit calculation:")
        for step in opportunity['steps']:
            st.write(f"- {step}")
        
        # Show impact on pool
        st.write("\nImpact on pool after trade:")
        initial_ratio = x_balance / (x_balance + e_balance) * 100
        
        if opportunity['direction'] == 'add_xastro':
            new_x = x_balance + opportunity['trade_size']
            new_e = e_balance - (opportunity['trade_size'] * (1 + abs(0.5 - initial_ratio/100)))
        else:
            new_x = x_balance - (opportunity['trade_size'] * (1 + abs(0.5 - initial_ratio/100)))
            new_e = e_balance + opportunity['trade_size']
            
        new_ratio = new_x / (new_x + new_e) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("xASTRO Pool %", f"{initial_ratio:.1f}%", f"{new_ratio - initial_ratio:.1f}%")
        with col2:
            st.metric("Price Impact", f"{abs(1 - initial_ratio/50):.1f}%", f"{abs(1 - new_ratio/50) - abs(1 - initial_ratio/50):.1f}%")

def create_event_timeline(df, parameter_changes):
    """Create timeline visualization of pool events"""
    fig = go.Figure()
    
    # Add vertical lines for events
    for change in parameter_changes:
        if 'events' in change:
            for event in change['events']:
                color = 'red' if event['type'] == 'selloff' else 'green'
                label = f"{event['type'].title()}: {event['pool']}<br>Impact: {abs(event['new_ratio'] - event['prev_ratio'])*100:.1f}% ratio change"
                
                fig.add_vline(
                    x=change['day'],
                    line_color=color,
                    line_dash="dash",
                    annotation_text=label,
                    annotation_position="top left"
                )
    
    fig.update_layout(
        title="Pool Events Timeline",
        xaxis_title="Day",
        showlegend=False,
        height=200
    )
    
    return fig

def create_profit_tracking_plot(df):
    """Create cumulative profit tracking visualization"""
    fig = go.Figure()
    
    for pool_type, color in POOL_COLORS.items():
        # Plot cumulative profits
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df[f'{pool_type}_total_profit'],
            name=f"{pool_type.replace('_', ' ').title()} Profits",
            line=dict(color=color)
        ))
        
        # Plot cumulative fees
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df[f'{pool_type}_total_fees'],
            name=f"{pool_type.replace('_', ' ').title()} Fees",
            line=dict(color=color, dash='dot')
        ))
    
    fig.update_layout(
        title="Cumulative Profits and Fees",
        xaxis_title="Day",
        yaxis_title="USD Value",
        yaxis_tickformat='$,.2f'
    )
    
    return fig

def create_tvl_impact_plot(df):
    """Create TVL impact visualization"""
    fig = go.Figure()
    
    for pool_type, color in POOL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df[f'{pool_type}_tvl'],
            name=f"{pool_type.replace('_', ' ').title()}",
            line=dict(color=color)
        ))
    
    fig.update_layout(
        title="Total Value Locked Over Time",
        xaxis_title="Day",
        yaxis_title="TVL (USD)",
        yaxis_tickformat='$,.2f'
    )
    
    return fig

def calculate_price_impact(x_balance, e_balance, amp, gamma=None, pool_type='stableswap'):
    """Calculate price impact as deviation from 1.0 (perfect peg)"""
    if pool_type.startswith('pcl'):
        price = calculate_pcl_price(x_balance, e_balance, amp, gamma)
    else:
        price = calculate_stableswap_price(x_balance, e_balance, amp)
    
    # Calculate percentage deviation from 1.0
    return abs(1 - price) * 100  # Convert to percentage

def run_simulation(initial_liquidity, selloffs, pcl_amp, stableswap_amp, xastro_price, eclip_price, simulate_rebalancing=False):
    """Run pool simulation with corrected price impact calculation"""
    pcl_dynamic_params = INITIAL_PARAMS['pcl_dynamic'].copy()
    pcl_fixed_params = INITIAL_PARAMS['pcl_fixed'].copy()
    stableswap_params = INITIAL_PARAMS['stableswap'].copy()
    
    # Set initial amplification parameters
    pcl_dynamic_params['amp'] = pcl_amp
    pcl_fixed_params['amp'] = pcl_amp
    stableswap_params['amp'] = stableswap_amp
    
    data = []
    parameter_changes = []
    days = 200

    # Calculate rebalancing days (10 days after each selloff)
    rebalance_days = [int(day) + 10 for day in selloffs.keys()] if simulate_rebalancing else []

    # Initialize balances and tracking metrics
    balances = {
        'dynamic_pcl': {'x': initial_liquidity, 'e': initial_liquidity},
        'fixed_pcl': {'x': initial_liquidity, 'e': initial_liquidity},
        'stableswap': {'x': initial_liquidity, 'e': initial_liquidity}
    }
    
    rebalancing_metrics = {
        'dynamic_pcl': {'total_profit': 0, 'trades': 0, 'fees_earned': 0},
        'fixed_pcl': {'total_profit': 0, 'trades': 0, 'fees_earned': 0},
        'stableswap': {'total_profit': 0, 'trades': 0, 'fees_earned': 0}
    }
    
    for day in range(days + 1):
        # Track events for this day
        day_events = []
        
        # Apply selloffs
        for selloff_day, amount in selloffs.items():
            if day == int(selloff_day):
                selloff_amount = initial_liquidity * (amount / 100)
                for pool_type in balances:
                    prev_x = balances[pool_type]['x']
                    prev_e = balances[pool_type]['e']
                    balances[pool_type]['x'] -= selloff_amount
                    balances[pool_type]['e'] += selloff_amount
                    day_events.append({
                        'type': 'selloff',
                        'pool': pool_type,
                        'amount': selloff_amount,
                        'prev_ratio': prev_x / (prev_x + prev_e),
                        'new_ratio': balances[pool_type]['x'] / (balances[pool_type]['x'] + balances[pool_type]['e'])
                    })

        # Apply rebalancing if it's a rebalancing day
        if day in rebalance_days:
            for pool_type in balances:
                prev_x = balances[pool_type]['x']
                prev_e = balances[pool_type]['e']
                total_tokens = prev_x + prev_e
                target_amount = total_tokens / 2
                
                # Calculate profit from rebalancing
                if pool_type.startswith('pcl'):
                    fee = INITIAL_PARAMS['pcl_dynamic']['out_fee']
                else:
                    fee = INITIAL_PARAMS['stableswap']['fee']
                
                # Calculate the amount being moved
                amount_to_move = abs(prev_x - target_amount)
                trade_value = amount_to_move * xastro_price
                fees_earned = trade_value * fee
                
                # Calculate profit (simplified model)
                price_impact = abs(1 - (prev_x / prev_e))
                profit = trade_value * price_impact * (1 - fee)
                
                # Update metrics
                rebalancing_metrics[pool_type]['total_profit'] += profit
                rebalancing_metrics[pool_type]['trades'] += 1
                rebalancing_metrics[pool_type]['fees_earned'] += fees_earned
                
                # Apply rebalancing
                balances[pool_type]['x'] = target_amount
                balances[pool_type]['e'] = target_amount
                
                day_events.append({
                    'type': 'rebalance',
                    'pool': pool_type,
                    'amount': amount_to_move,
                    'profit': profit,
                    'fees': fees_earned,
                    'prev_ratio': prev_x / (prev_x + prev_e),
                    'new_ratio': 0.5  # Always 50/50 after rebalancing
                })

        # Calculate prices for each pool
        prices = {
            'dynamic_pcl': calculate_pcl_price(
                balances['dynamic_pcl']['x'],
                balances['dynamic_pcl']['e'],
                pcl_dynamic_params['amp'],
                pcl_dynamic_params['gamma']
            ),
            'fixed_pcl': calculate_pcl_price(
                balances['fixed_pcl']['x'],
                balances['fixed_pcl']['e'],
                pcl_fixed_params['amp'],
                pcl_fixed_params['gamma']
            ),
            'stableswap': calculate_stableswap_price(
                balances['stableswap']['x'],
                balances['stableswap']['e'],
                stableswap_params['amp']
            )
        }

        # Store daily data with enhanced metrics
        day_data = {
            'day': day,
            'pcl_dynamic_price': prices['dynamic_pcl'],
            'pcl_fixed_price': prices['fixed_pcl'],
            'stableswap_price': prices['stableswap'],
            'pcl_dynamic_amp': pcl_dynamic_params['amp'],
            'pcl_dynamic_out_fee': pcl_dynamic_params['out_fee'],
            'has_event': len(day_events) > 0
        }

        # Store actual balances
        for pool_type in balances:
            # Store token balances
            day_data[f'{pool_type}_x_balance'] = balances[pool_type]['x']
            day_data[f'{pool_type}_e_balance'] = balances[pool_type]['e']
            
            # Calculate and store ratios
            total = balances[pool_type]['x'] + balances[pool_type]['e']
            day_data[f'{pool_type}_x_pct'] = (balances[pool_type]['x'] / total) * 100
            day_data[f'{pool_type}_e_pct'] = (balances[pool_type]['e'] / total) * 100
            
            # Store TVL and price impact
            day_data[f'{pool_type}_tvl'] = (balances[pool_type]['x'] * xastro_price + 
                                           balances[pool_type]['e'] * eclip_price)
            day_data[f'{pool_type}_price_impact'] = abs(1 - (balances[pool_type]['x'] / balances[pool_type]['e']))
            
            # Store rebalancing metrics
            day_data[f'{pool_type}_total_profit'] = rebalancing_metrics[pool_type]['total_profit']
            day_data[f'{pool_type}_total_fees'] = rebalancing_metrics[pool_type]['fees_earned']
            day_data[f'{pool_type}_trade_count'] = rebalancing_metrics[pool_type]['trades']
            
            # Calculate rebalancing opportunity
            amp = pcl_amp if 'pcl' in pool_type else stableswap_amp
            opportunity_value = calculate_rebalancing_opportunity_value(
                balances[pool_type]['x'],
                balances[pool_type]['e'],
                xastro_price,
                eclip_price,
                amp
            )
            day_data[f'{pool_type}_rebalance_opportunity'] = opportunity_value

        # Store day's events
        if day_events:
            parameter_changes.append({
                'day': day,
                'events': day_events
            })

        # Adjust dynamic PCL parameters
        if day > 0:
            imbalance = abs(1 - (balances['dynamic_pcl']['x'] / balances['dynamic_pcl']['e']))
            new_params, changed, message = adjust_pcl_parameters(imbalance, pcl_dynamic_params)
            if changed:
                parameter_changes.append({
                    'day': day,
                    'message': message,
                    'type': 'parameter_change'
                })
                pcl_dynamic_params = new_params

        data.append(day_data)
    
    return pd.DataFrame(data), parameter_changes

def create_price_impact_plot(df):
    """Create price impact visualization with corrected values"""
    fig = go.Figure()
    
    for pool_type, color in POOL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df[f'{pool_type}_price_impact'],
            name=f"{pool_type.replace('_', ' ').title()}",
            line=dict(color=color)
        ))
    
    fig.update_layout(
        title="Price Impact Over Time",
        xaxis_title="Day",
        yaxis_title="Price Impact (%)",
        yaxis_tickformat='.3f%'  # Show more decimal places for precision
    )
    
    return fig

def show_pool_metrics_summary(df, pool_type):
    """Show detailed metrics for a specific pool type"""
    latest = df.iloc[-1]
    initial = df.iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Profit",
            f"${latest[f'{pool_type}_total_profit']:,.2f}",
            help="Cumulative profit from rebalancing activities"
        )
        
    with col2:
        st.metric(
            "Total Fees Earned",
            f"${latest[f'{pool_type}_total_fees']:,.2f}",
            help="Cumulative fees earned by LPs"
        )
        
    with col3:
        st.metric(
            "Trade Count",
            f"{int(latest[f'{pool_type}_trade_count']):,}",
            help="Number of rebalancing trades executed"
        )
    
    col4, col5 = st.columns(2)
    
    with col4:
        tvl_change = (latest[f'{pool_type}_tvl'] - initial[f'{pool_type}_tvl']) / initial[f'{pool_type}_tvl'] * 100
        st.metric(
            "TVL Change",
            f"${latest[f'{pool_type}_tvl']:,.2f}",
            f"{tvl_change:+.2f}%",
            help="Change in Total Value Locked"
        )
        
    with col5:
        price_impact_change = latest[f'{pool_type}_price_impact'] - initial[f'{pool_type}_price_impact']
        st.metric(
            "Current Price Impact",
            f"{latest[f'{pool_type}_price_impact']:.2%}",
            f"{price_impact_change:+.2%}",
            help="Current price impact percentage"
        )

def show_rebalancing_metrics(df, parameter_changes):
    """Show detailed rebalancing metrics"""
    st.subheader("Rebalancing Analysis")
    
    # Extract rebalancing events
    rebalancing_events = [
        change for change in parameter_changes 
        if 'events' in change and any(event['type'] == 'rebalance' for event in change['events'])
    ]
    
    if rebalancing_events:
        st.write("Rebalancing Events Summary:")
        for event_day in rebalancing_events:
            rebalance_events = [e for e in event_day['events'] if e['type'] == 'rebalance']
            
            for event in rebalance_events:
                with st.expander(f"Day {event_day['day']} - {event['pool'].replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Profit Generated", f"${event['profit']:,.2f}")
                        st.metric("Fees Generated", f"${event['fees']:,.2f}")
                    with col2:
                        ratio_improvement = abs(event['new_ratio'] - event['prev_ratio']) * 100
                        st.metric("Ratio Improvement", f"{ratio_improvement:.2f}%")
                        st.metric("Tokens Moved", f"{event['amount']:,.0f}")
    else:
        st.info("No rebalancing events have occurred yet.")

def simulate_market_rebalance(x_balance, e_balance):
    """Simulate market rebalancing action"""
    total_tokens = x_balance + e_balance
    target_amount = total_tokens / 2
    
    # Calculate how much needs to move
    if x_balance < target_amount:
        # Need to add xASTRO
        amount_to_move = target_amount - x_balance
        new_x = x_balance + amount_to_move
        new_e = e_balance - amount_to_move
    else:
        # Need to add eclipASTRO
        amount_to_move = x_balance - target_amount
        new_x = x_balance - amount_to_move
        new_e = e_balance + amount_to_move
    
    return new_x, new_e

def add_chart_explanation(title, description, key_points=None, significance=None):
    """Helper function to format chart explanations consistently with HTML"""
    
    # Convert key points to HTML list items
    key_points_html = ''.join([f'<li>{point}</li>' for point in key_points]) if key_points else ''
    
    # Use CSS for consistent styling
    css = """
        <style>
            .chart-explanation {
                word-wrap: break-word;
                margin: 10px 0;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            .chart-explanation h5 {
                color: #1f77b4;
                margin-bottom: 10px;
            }
            .chart-explanation p {
                margin-bottom: 10px;
                line-height: 1.5;
            }
            .chart-explanation ul {
                margin-top: 5px;
                margin-bottom: 10px;
                padding-left: 20px;
            }
            .chart-explanation .significance {
                margin-top: 5px;
                line-height: 1.5;
            }
        </style>
    """
    
    explanation_html = f"""
    {css}
    <div class="chart-explanation">
        <h5>{title}</h5>
        <p>{description}</p>
        <div>
            <strong>Key Points:</strong>
            <ul>
                {key_points_html}
            </ul>
        </div>
        <div>
            <strong>Significance:</strong>
            <p class="significance">{significance}</p>
        </div>
    </div>
    """
    
    st.markdown(explanation_html, unsafe_allow_html=True)


def main():
    st.title("xASTRO/eclipASTRO Pool Simulator")
    
    # Sidebar controls
    st.sidebar.header("Initial Pool Parameters")
    
    initial_xastro = st.sidebar.number_input(
        "Initial xASTRO", 
        min_value=1000000, 
        max_value=100000000, 
        value=25000000
    )
    
    initial_eclip = st.sidebar.number_input(
        "Initial eclipASTRO", 
        min_value=1000000, 
        max_value=100000000, 
        value=25000000
    )
    
    st.sidebar.header("Token Prices (USD)")
    
    xastro_price = st.sidebar.number_input(
        "xASTRO Price (USD)", 
        min_value=0.00001, 
        max_value=1000.0, 
        value=0.03457,
        step=0.00001,
        format="%.5f",
        help="Current market price of xASTRO in USD"
    )
    
    eclip_price = st.sidebar.number_input(
        "eclipASTRO Price (USD)", 
        min_value=0.00001, 
        max_value=1000.0, 
        value=0.02303,
        step=0.00001,
        format="%.5f",
        help="Current market price of eclipASTRO in USD"
    )

    initial_tvl = (initial_xastro * xastro_price) + (initial_eclip * eclip_price)
    st.sidebar.metric(
        "Initial Pool TVL (USD)", 
        f"${initial_tvl:,.2f}"
    )
    
    pcl_amp = st.sidebar.number_input(
        "PCL Amplification Parameter",
        min_value=50,
        max_value=1000,
        value=150
    )
    
    stableswap_amp = st.sidebar.number_input(
        "StableSwap Amplification Parameter",
        min_value=50,
        max_value=1000,
        value=400
    )
    
    st.sidebar.header("Sell-off Events (xASTRO amounts)")
    max_selloff = initial_xastro
    default_selloff = initial_xastro * 0.2
    
    selloffs = {
        "2": st.sidebar.number_input("Day 2 Sell-off Amount", 0, max_selloff, int(default_selloff)),
        "30": st.sidebar.number_input("Day 30 Sell-off Amount", 0, max_selloff, int(default_selloff)),
        "90": st.sidebar.number_input("Day 90 Sell-off Amount", 0, max_selloff, int(default_selloff)),
        "180": st.sidebar.number_input("Day 180 Sell-off Amount", 0, max_selloff, int(default_selloff))
    }
    
    # Convert absolute amounts to percentages
    selloffs_pct = {
        day: (amount / initial_xastro * 100) for day, amount in selloffs.items()
    }

    # Add rebalancing simulation checkbox
    simulate_rebalancing = st.sidebar.checkbox(
        "Simulate Market Rebalancing",
        value=False,
        help="Simulate market rebalancing 10 days after each sell-off"
    )

    # Run simulation
    df, parameter_changes = run_simulation(
        initial_xastro,
        selloffs_pct,
        pcl_amp,
        stableswap_amp,
        xastro_price,
        eclip_price,
        simulate_rebalancing
    )

    # Create tabs for different visualization groups
    tab1, tab2, tab3 = st.tabs([
        "Pool Performance", 
        "Rebalancing Analysis", 
        "Parameter Evolution"
    ])

    with tab1:
        st.header("Pool Performance Metrics")
        
        # Price Impact Plot
        fig1 = go.Figure()
        for pool_type, color in [
            ('pcl_dynamic_price', POOL_COLORS['dynamic_pcl']),
            ('pcl_fixed_price', POOL_COLORS['fixed_pcl']),
            ('stableswap_price', POOL_COLORS['stableswap'])
        ]:
            fig1.add_trace(go.Scatter(
                x=df['day'],
                y=df[pool_type],
                name=pool_type.replace('_price', '').replace('_', ' ').title(),
                line=dict(color=color)
            ))
        
        fig1.update_layout(
            title="Price Impact Comparison",
            xaxis_title="Day",
            yaxis_title="Price",
            yaxis_autorange=True
        )
        st.plotly_chart(fig1, use_container_width=True)
        add_chart_explanation(
            "Price Impact Comparison",
            "This chart shows how the price of xASTRO relative to eclipASTRO changes over time in each pool type.",
            key_points=[
                "Price = 1.0 indicates perfect peg",
                "Deviations from 1.0 indicate arbitrage opportunities",
                "Larger deviations mean larger profit potential for arbitrageurs"
            ],
            significance="Price impact directly affects trading costs and arbitrage opportunities. Lower price impact indicates a more efficient pool."
        )
        
        # Pool Composition Plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['day'],
            y=df['dynamic_pcl_x_pct'],
            name="xASTRO %",
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(31, 119, 180, 0.5)'
        ))
        fig2.add_trace(go.Scatter(
            x=df['day'],
            y=df['dynamic_pcl_x_pct'] + df['dynamic_pcl_e_pct'],
            name="eclipASTRO %",
            fill='tonexty',
            mode='none',
            fillcolor='rgba(255, 127, 14, 0.5)'
        ))
        fig2.update_layout(
            title="Pool Composition Over Time",
            xaxis_title="Day",
            yaxis_title="Composition %",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig2, use_container_width=True)
        add_chart_explanation(
            "Pool Composition",
            "Shows the relative proportion of xASTRO and eclipASTRO in the pool over time.",
            key_points=[
                "Ideal ratio is 50/50",
                "Sell-offs cause ratio imbalances",
                "Rebalancing events return the ratio toward 50/50"
            ],
            significance="Pool composition affects trading efficiency and stability. Maintaining a balanced ratio is crucial for optimal pool performance."
        )

        # TVL Impact Plot
        st.plotly_chart(create_tvl_impact_plot(df), use_container_width=True)
        add_chart_explanation(
            "Total Value Locked (TVL)",
            "Tracks the total dollar value of assets in each pool type over time.",
            key_points=[
                "Higher TVL indicates more liquidity",
                "Sharp drops indicate sell-off events",
                "Gradual changes show organic pool activity"
            ],
            significance="TVL is a key metric for pool health and liquidity depth. Higher TVL generally means better price stability and lower slippage."
        )

        # Price Impact Over Time Plot
        st.plotly_chart(create_price_impact_plot(df), use_container_width=True)
        add_chart_explanation(
            "Price Impact Trend",
            "Measures how much trades affect the pool price over time.",
            key_points=[
                "Lower price impact is better",
                "Spikes indicate large trades or imbalances",
                "Different pool types handle price impact differently"
            ],
            significance="Price impact directly affects trading costs. Lower price impact means more efficient trading and better price execution."
        )
        
        # Add spacing between charts
        st.markdown("<br/>", unsafe_allow_html=True)

    with tab2:
        st.header("Rebalancing Analysis")
        
        # Timeline of Events
        st.plotly_chart(create_event_timeline(df, parameter_changes), use_container_width=True)
        add_chart_explanation(
            "Event Timeline",
            "Visual timeline of all significant pool events including sell-offs and rebalancing actions.",
            key_points=[
                "Red lines indicate sell-off events",
                "Green lines indicate rebalancing events",
                "Hover for detailed event information"
            ],
            significance="Understanding when and how often pool-changing events occur helps predict arbitrage opportunities and assess pool stability."
        )

        # Profit Tracking
        st.plotly_chart(create_profit_tracking_plot(df), use_container_width=True)
        add_chart_explanation(
            "Profit and Fees",
            "Tracks cumulative profits from arbitrage opportunities and fees earned by liquidity providers.",
            key_points=[
                "Solid lines show arbitrage profits",
                "Dotted lines show fee revenue",
                "Steeper slopes indicate more profitable periods"
            ],
            significance="Shows the financial incentives for both arbitrageurs and liquidity providers, helping understand the pool's economic dynamics."
        )

        # Detailed Pool Metrics
        st.subheader("Pool Performance Metrics")
        
        pool_type = st.selectbox(
            "Select Pool Type for Detailed Metrics",
            options=list(POOL_COLORS.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        show_pool_metrics_summary(df, pool_type)
        
        # Rebalancing Details
        show_rebalancing_metrics(df, parameter_changes)

        # Current Rebalancing Opportunities
        st.subheader("Current Rebalancing Opportunities")
        latest_data = df.iloc[-1]
        
        for pool_type in POOL_COLORS.keys():
            with st.expander(f"{pool_type.replace('_', ' ').title()} Pool"):
                show_rebalancing_details(
                    latest_data[f'{pool_type}_x_balance'],
                    latest_data[f'{pool_type}_e_balance'],
                    xastro_price,
                    eclip_price,
                    pcl_amp if 'pcl' in pool_type else stableswap_amp,
                    pool_type
                )

    with tab3:
        st.header("Parameter Evolution")
        
        # Parameter Evolution Plot
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig4.add_trace(
            go.Scatter(
                x=df['day'], 
                y=df['pcl_dynamic_amp'],
                name="Dynamic PCL Amp",
                line=dict(color=POOL_COLORS['dynamic_pcl'])
            ),
            secondary_y=False
        )
        
        fig4.add_trace(
            go.Scatter(
                x=df['day'], 
                y=df['pcl_dynamic_out_fee'],
                name="Dynamic PCL Out Fee",
                line=dict(color=POOL_COLORS['dynamic_pcl'], dash='dash')
            ),
            secondary_y=True
        )
        
        fig4.update_layout(
            title="Dynamic PCL Parameters Over Time",
            xaxis_title="Day"
        )
        
        fig4.update_yaxes(title_text="Amplification", secondary_y=False)
        fig4.update_yaxes(title_text="Out Fee", secondary_y=True)
        
        st.plotly_chart(fig4, use_container_width=True)
        add_chart_explanation(
            "Dynamic Parameters",
            "Shows how PCL pool parameters change in response to market conditions.",
            key_points=[
                "Amplification (Amp) affects price curve steepness",
                "Out fee adjusts based on pool imbalance",
                "Parameters automatically optimize for stability"
            ],
            significance="Dynamic parameter adjustment is key to PCL's ability to maintain stability while maximizing capital efficiency."
        )

        # Parameter Changes Log
        if parameter_changes:
            st.header("Parameter Adjustment Log")
            
            # Create a more structured view of parameter changes
            for change in parameter_changes:
                if 'type' in change and change['type'] == 'parameter_change':
                    st.info(f"Day {change['day']}: {change['message']}")
                elif 'events' in change:
                    for event in change['events']:
                        color = 'success' if event['type'] == 'rebalance' else 'warning'
                        message = (f"{event['type'].title()} event in {event['pool'].replace('_', ' ').title()} pool\n"
                                 f"Impact: {abs(event['new_ratio'] - event['prev_ratio'])*100:.1f}% ratio change")
                        getattr(st, color)(f"Day {change['day']}: {message}")

            # Add explanation for parameter changes
            add_chart_explanation(
                "Parameter Changes Log",
                "Chronological record of all parameter adjustments and their triggers.",
                key_points=[
                    "Shows when and why parameters changed",
                    "Helps understand pool's adaptive behavior",
                    "Color coding indicates event significance"
                ],
                significance="Understanding parameter change patterns helps predict pool behavior and optimize trading strategies."
            )

if __name__ == "__main__":
    main()
