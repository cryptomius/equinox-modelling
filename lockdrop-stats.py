import streamlit as st
import json
import pandas as pd
import websockets
import asyncio
import time
import nest_asyncio
from base64 import b64encode
import plotly.graph_objects as go

# Enable nested event loops
nest_asyncio.apply()

# Constants
WS_ENDPOINT = "wss://neutron-rpc.publicnode.com:443/websocket"
CONTRACT_ADDRESS = "neutron1zh097hf7pz3d0pz3jnt3urhyw03kmcpxfs4as6sqz4cyfjkyyzmqpvd2n5"
XASTRO_IDENTIFIER = "factory/neutron1zlf3hutsa4qnmue53lz2tfxrutp8y2e3rj4nkghg3rupgl4mqy8s5jgxsn/xASTRO"
LOCKDROP_CONTRACT = "neutron1zh097hf7pz3d0pz3jnt3urhyw03kmcpxfs4as6sqz4cyfjkyyzmqpvd2n5"

# Lock period mapping (in seconds)
LOCK_PERIODS = {
    0: "Flexible",
    2592000: "1 Month",
    7776000: "3 Months", 
    15552000: "6 Months"
}

async def fetch_transactions(_progress_bar):
    transactions = []
    page = 1
    per_page = 100
    total_fetched = 0
    
    try:
        async with websockets.connect(WS_ENDPOINT, ping_interval=None) as websocket:
            while True:
                query = {
                    "jsonrpc": "2.0",
                    "id": page,
                    "method": "tx_search",
                    "params": {
                        "query": f"execute._contract_address='{CONTRACT_ADDRESS}'",
                        "prove": False,
                        "page": str(page),
                        "per_page": str(per_page),
                        "order_by": "asc"
                    }
                }
                
                await websocket.send(json.dumps(query))
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    
                    if "result" in data:
                        if "txs" in data["result"]:
                            new_txs = data["result"]["txs"]
                            total_count = int(data["result"].get("total_count", 0))
                            
                            if not new_txs:  # No more transactions
                                break
                                
                            filtered_txs = [
                                tx for tx in new_txs
                                if CONTRACT_ADDRESS in str(tx)
                            ]
                            
                            if filtered_txs:
                                transactions.extend(filtered_txs)
                                total_fetched += len(filtered_txs)
                                
                                # Update progress
                                progress = min(total_fetched / total_count, 1.0)
                                _progress_bar.progress(progress)
                            
                            # Break if we've fetched all transactions
                            if total_fetched >= total_count:
                                break
                            
                            page += 1
                            
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    st.error(f"Error processing page {page}: {str(e)}")
                    continue
                
    except Exception as e:
        st.error(f"WebSocket error: {str(e)}")
        return []
    
    _progress_bar.progress(1.0)
    return transactions

def process_transactions(txs):
    try:
        events_data = []
        withdrawal_count = 0
        deposit_count = 0
        
        def normalize_stake_type(stake_type):
            if not stake_type:
                return ''
            stake_type = stake_type.lower().replace(' ', '_')
            if 'single' in stake_type:
                return 'single_staking'
            if 'lp' in stake_type:
                return 'lp_staking'
            return stake_type
        
        # Running balances
        balances = {
            'single_staking': {
                'Flexible': 0,
                '1 Month': 0,
                '3 Months': 0,
                '6 Months': 0,
                'total': 0
            },
            'lp_staking': {
                'Flexible': 0,
                '1 Month': 0,
                '3 Months': 0,
                '6 Months': 0,
                'total': 0
            },
            'total': 0
        }

        def get_lock_period(duration):
            if duration == 0:
                return 'Flexible'
            elif duration == 2592000:  # 30 days
                return '1 Month'
            elif duration == 7776000:  # 90 days
                return '3 Months'
            elif duration == 15552000:  # 180 days
                return '6 Months'
            return 'Flexible'

        for tx in txs:
            try:
                tx_data = {
                    'hash': tx.get('hash'),
                    'height': tx.get('height'),
                    'timestamp': None,
                    'amount': None,
                    'duration': 0,
                    'stake_type': '',
                    'sender': None,
                    'action': None,
                    'is_withdrawal': False
                }
                
                for event in tx.get('tx_result', {}).get('events', []):
                    if event.get('type') == 'wasm':
                        attrs = {attr['key']: attr['value'] for attr in event.get('attributes', [])}
                        
                        # Debug: Print transaction details
                        st.write(f"Processing tx {tx_data['hash'][:8]}, action: {attrs.get('action')}")
                        
                        # Only process increase_lockup_position and withdraw actions
                        if attrs.get('action') == 'increase_lockup_position':
                            if not attrs.get('type') or not attrs.get('amount'):
                                continue
                                
                            tx_data['action'] = 'deposit'
                            tx_data['stake_type'] = normalize_stake_type(attrs.get('type'))
                            tx_data['duration'] = int(attrs.get('duration', 0))
                            
                            # Only process if it's a valid xASTRO deposit
                            if XASTRO_IDENTIFIER in attrs.get('asset', ''):
                                tx_data['amount'] = float(attrs['amount']) / 1_000_000
                                deposit_count += 1
                                
                                # Debug: Print deposit details
                                st.write(f"Deposit: {tx_data['amount']} xASTRO to {tx_data['stake_type']} ({get_lock_period(tx_data['duration'])})")
                        
                        elif attrs.get('action') == 'withdraw':
                            if not attrs.get('withdraw_amount'):
                                continue
                                
                            tx_data['is_withdrawal'] = True
                            tx_data['action'] = 'withdraw'
                            tx_data['amount'] = -float(attrs['withdraw_amount']) / 1_000_000
                            
                            # Try to get stake type from event
                            for type_key in ['type', 'lockup_type']:
                                if attrs.get(type_key):
                                    tx_data['stake_type'] = normalize_stake_type(attrs.get(type_key))
                                    break
                            
                            if not tx_data['stake_type']:
                                # Skip withdrawals where we can't determine the stake type
                                continue
                                
                            withdrawal_count += 1
                            
                            # Debug: Print withdrawal details
                            st.write(f"Withdrawal: {abs(tx_data['amount'])} xASTRO from {tx_data['stake_type']}")

                # Process valid transactions
                if tx_data['amount'] is not None and tx_data['stake_type']:
                    stake_type = tx_data['stake_type']
                    amount = tx_data['amount']
                    lock_period = get_lock_period(tx_data['duration'])
                    
                    if stake_type in balances:
                        # Update balances
                        balances[stake_type][lock_period] += amount
                        balances[stake_type]['total'] += amount
                        balances['total'] += amount
                        
                        # Debug: Print running balances
                        st.write(f"Updated {stake_type} {lock_period}: {balances[stake_type][lock_period]}")
                        st.write(f"Total {stake_type}: {balances[stake_type]['total']}")
                        
                        tx_data.update({
                            'running_total': balances['total'],
                            'running_single_total': balances['single_staking']['total'],
                            'running_lp_total': balances['lp_staking']['total'],
                            'lock_period': lock_period
                        })
                        events_data.append(tx_data)
                        
            except Exception as e:
                st.error(f"Error processing transaction {tx.get('hash')}: {str(e)}")
                continue

        # After processing all transactions, display final balances with percentages
        st.write("\nFinal Balances:")
        
        # Single Staking Summary
        single_total = balances['single_staking']['total']
        st.write("\nSingle Sided Staking xASTRO:")
        st.write(f"Total: {single_total:,.0f} xASTRO")
        
        expected_single = 14_925_587
        single_diff = single_total - expected_single
        st.write(f"Expected: {expected_single:,.0f} xASTRO")
        st.write(f"Difference: {single_diff:,.0f} xASTRO ({(single_diff/expected_single)*100:.2f}%)")
        
        for period in ['Flexible', '1 Month', '3 Months', '6 Months']:
            amount = balances['single_staking'][period]
            pct = (amount / single_total * 100) if single_total > 0 else 0
            st.write(f"{period}: {amount:,.0f} xASTRO ({pct:.1f}%)")
            
        # Expected values for single staking
        expected_single_values = {
            'Flexible': 2_031_885,
            '1 Month': 2_479_213,
            '3 Months': 3_729_344,
            '6 Months': 6_685_143
        }
        
        # Compare with expected values
        st.write("\nSingle Staking Differences:")
        for period, expected in expected_single_values.items():
            actual = balances['single_staking'][period]
            diff = actual - expected
            st.write(f"{period}: {diff:,.0f} xASTRO ({(diff/expected)*100:.2f}% difference)")

        # LP Staking Summary
        lp_total = balances['lp_staking']['total']
        st.write("\nLP Staking xASTRO:")
        st.write(f"Total: {lp_total:,.0f} xASTRO")
        
        expected_lp = 10_884_926
        lp_diff = lp_total - expected_lp
        st.write(f"Expected: {expected_lp:,.0f} xASTRO")
        st.write(f"Difference: {lp_diff:,.0f} xASTRO ({(lp_diff/expected_lp)*100:.2f}%)")
        
        for period in ['Flexible', '1 Month', '3 Months', '6 Months']:
            amount = balances['lp_staking'][period]
            pct = (amount / lp_total * 100) if lp_total > 0 else 0
            st.write(f"{period}: {amount:,.0f} xASTRO ({pct:.1f}%)")
            
        # Expected values for LP staking
        expected_lp_values = {
            'Flexible': 1_365_549,
            '1 Month': 5_985_110,
            '3 Months': 1_449_353,
            '6 Months': 2_084_912
        }
        
        # Compare with expected values
        st.write("\nLP Staking Differences:")
        for period, expected in expected_lp_values.items():
            actual = balances['lp_staking'][period]
            diff = actual - expected
            st.write(f"{period}: {diff:,.0f} xASTRO ({(diff/expected)*100:.2f}% difference)")

        # Total Summary
        total_xastro = single_total + lp_total
        expected_total = expected_single + expected_lp
        total_diff = total_xastro - expected_total
        
        st.write("\nTotal Summary:")
        st.write(f"Total xASTRO in Lockdrop: {total_xastro:,.0f}")
        st.write(f"Expected Total: {expected_total:,.0f}")
        st.write(f"Difference: {total_diff:,.0f} xASTRO ({(total_diff/expected_total)*100:.2f}%)")

        # Create DataFrame and return
        df = pd.DataFrame(events_data)
        
        if not df.empty:
            # Convert block height to datetime
            NEUTRON_GENESIS_TIME = 1697882751
            NEUTRON_GENESIS_HEIGHT = 1
            AVERAGE_BLOCK_TIME = 6.0
            
            df['datetime'] = pd.to_datetime(
                NEUTRON_GENESIS_TIME + 
                ((df['height'].astype(int) - NEUTRON_GENESIS_HEIGHT) * AVERAGE_BLOCK_TIME), 
                unit='s'
            )
        
        return df

    except Exception as e:
        st.error(f"Error in process_transactions: {str(e)}")
        return pd.DataFrame()

def create_timeline_charts(df):
    df = df.sort_values('datetime')
    
    # Create stacked area chart
    fig = go.Figure()
    
    # Add Single Sided trace
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['running_single_total'],
        name='Single Sided',
        stackgroup='one',  # Enable stacking
        groupnorm=None,    # Do not normalize
        line=dict(width=0.5)
    ))
    
    # Add LP Staking trace
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['running_lp_total'],
        name='LP Staking',
        stackgroup='one',  # Enable stacking
        groupnorm=None,    # Do not normalize
        line=dict(width=0.5)
    ))
    
    fig.update_layout(
        title="Total xASTRO Locked Over Time",
        xaxis_title="Date",
        yaxis_title="xASTRO Amount",
        hovermode='x unified',
        yaxis=dict(rangemode='nonnegative'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig)

def create_lock_duration_charts(df):
    # Create separate charts for single sided and LP
    for stake_type in ['single_staking', 'lp_staking']:
        filtered_df = df[df['stake_type'] == stake_type].copy()
        
        if filtered_df.empty:
            continue
            
        # Group by datetime and lock period
        pivot_df = filtered_df.pivot_table(
            index='datetime',
            columns='lock_period',
            values='amount',
            aggfunc='sum'
        ).fillna(0).cumsum()
        
        # Create stacked area chart
        fig = go.Figure()
        for duration in LOCK_PERIODS.values():
            if duration in pivot_df.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[duration],
                    name=duration,
                    stackgroup='one'
                ))
        
        st.subheader(f"{stake_type.replace('_', ' ').title()} Lock Durations")
        st.plotly_chart(fig)

def create_top_participants_table(df):
    # Calculate net positions by wallet
    wallet_positions = df.groupby('sender').agg({
        'amount': 'sum'
    }).sort_values('amount', ascending=False)
    
    # Calculate splits between single sided and LP
    single_sided = df[df['stake_type'] == 'single_staking'].groupby('sender')['amount'].sum()
    lp_staking = df[df['stake_type'] == 'lp_staking'].groupby('sender')['amount'].sum()
    
    # Combine into final table
    top_50 = pd.DataFrame({
        'Total Balance': wallet_positions['amount'],
        'Single Sided': single_sided,
        'LP Staking': lp_staking
    }).fillna(0)
    
    # Calculate percentages
    total_locked = top_50['Total Balance'].sum()
    top_50['% of Total'] = (top_50['Total Balance'] / total_locked * 100).round(2)
    
    st.subheader("Top 50 Participants")
    st.dataframe(top_50.head(50))

async def main():
    st.title("xASTRO Lockdrop Dashboard")
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Fetch transactions
    txs = await fetch_transactions(progress_bar)
    
    # Debug info for transactions
    st.write(f"Number of transactions fetched: {len(txs)}")
    
    # Process transactions
    df = process_transactions(txs)
    
    # Debug info for DataFrame
    st.write(f"DataFrame size: {df.shape}")
    if not df.empty:
        st.write("DataFrame columns:", df.columns.tolist())
        st.write("Sample of data:")
        st.write(df.head())
    
    if not df.empty:
        # Create visualizations
        create_timeline_charts(df)
        create_lock_duration_charts(df)
        create_top_participants_table(df)
    else:
        st.error("No data available - DataFrame is empty")

if __name__ == "__main__":
    asyncio.run(main())
