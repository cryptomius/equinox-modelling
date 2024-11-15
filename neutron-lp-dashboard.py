import streamlit as st
import websockets
import asyncio
import json
import nest_asyncio
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Enable nested event loops
nest_asyncio.apply()

# Constants
WS_ENDPOINT = "wss://neutron-rpc.publicnode.com:443/websocket"
CONTRACT_ADDRESS = "neutron1l9tkl663m2k3l3stzcl7mekwluj0xa8kh3sekp5qd42wh5gkevvsuzcycl"

def get_block_time(height):
    """Get block time using REST API with retries."""
    endpoints = [
        "https://rest.publicnode.com:443/neutron",
        "https://rest.lavenderfive.com:443/neutron",
        "https://api.neutron.quokkastake.io"
    ]
    
    for endpoint in endpoints:
        try:
            url = f"{endpoint}/cosmos/base/tendermint/v1beta1/blocks/{height}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data['block']['header']['time']
        except:
            continue
    
    return None

async def fetch_transactions(progress_bar):
    """Fetch contract transactions using WebSocket connection."""
    transactions = []
    page = 1
    per_page = 100
    total_fetched = 0
    
    try:
        async with websockets.connect(WS_ENDPOINT, ping_interval=None) as websocket:
            # Get total count first
            initial_query = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "tx_search",
                "params": {
                    "query": f"wasm._contract_address='{CONTRACT_ADDRESS}'",
                    "prove": False,
                    "page": "1",
                    "per_page": "1",
                    "order_by": "desc"
                }
            }
            
            await websocket.send(json.dumps(initial_query))
            initial_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            initial_data = json.loads(initial_response)
            
            if "result" not in initial_data:
                st.error(f"Invalid response format: {initial_data}")
                return []
                
            total_count = int(initial_data["result"].get("total_count", 0))
            st.write(f"Total transactions to process: {total_count}")
            
            while True:
                query = {
                    "jsonrpc": "2.0",
                    "id": page,
                    "method": "tx_search",
                    "params": {
                        "query": f"wasm._contract_address='{CONTRACT_ADDRESS}'",
                        "prove": False,
                        "page": str(page),
                        "per_page": str(per_page),
                        "order_by": "desc"
                    }
                }
                
                await websocket.send(json.dumps(query))
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    
                    if "result" in data and "txs" in data["result"]:
                        new_txs = data["result"]["txs"]
                        
                        if not new_txs:  # No more transactions
                            break
                        
                        # Process transactions in batches for timestamp fetching
                        tx_batch = []
                        for tx in new_txs:
                            tx_data = {
                                'height': tx.get('height'),
                                'hash': tx.get('hash'),
                                'tx_result': tx.get('tx_result', {})
                            }
                            tx_batch.append(tx_data)
                        
                        # Fetch timestamps in parallel
                        with ThreadPoolExecutor(max_workers=10) as executor:
                            heights = [tx['height'] for tx in tx_batch]
                            timestamps = list(executor.map(get_block_time, heights))
                        
                        # Add timestamps to transactions
                        for tx, timestamp in zip(tx_batch, timestamps):
                            if timestamp:
                                tx['timestamp'] = timestamp
                                transactions.append(tx)
                        
                        # Update progress
                        total_fetched += len(new_txs)
                        progress = min(total_fetched / total_count, 1.0)
                        progress_bar.progress(progress)
                        
                        # Break if we've fetched all transactions
                        if total_fetched >= total_count:
                            break
                        
                        page += 1
                        
                except asyncio.TimeoutError:
                    st.warning(f"Timeout on page {page}, retrying...")
                    continue
                except Exception as e:
                    st.error(f"Error processing page {page}: {str(e)}")
                    continue
                
    except Exception as e:
        st.error(f"WebSocket error: {str(e)}")
        return []
    
    st.success(f"Successfully fetched {len(transactions)} transactions")
    return transactions

def process_transactions(transactions, contract_address):
    """Process transactions to extract swap and liquidity data."""
    st.write("Processing transactions...")
    
    # Debug: Show first transaction
    if transactions:
        st.write("First transaction structure:")
        st.json(transactions[0])
        
        # Show events structure
        if 'tx_result' in transactions[0] and 'events' in transactions[0]['tx_result']:
            st.write("\nFirst transaction events:")
            for event in transactions[0]['tx_result']['events']:
                st.write("\nEvent type:", event.get('type'))
                st.write("Attributes:")
                for attr in event.get('attributes', []):
                    st.write(f"Key (raw): {attr.get('key')}")
                    st.write(f"Value (raw): {attr.get('value')}")
    
    swaps = []
    liquidity_changes = []
    
    # Will store running totals, starting from 0
    pool_states = [{
        'timestamp': None,  # Will be set to first transaction time
        'xastro_amount': 0,
        'eclipastro_amount': 0,
        'event_type': 'initial'
    }]
    
    # Sort transactions by timestamp to process in chronological order
    sorted_transactions = sorted(transactions, key=lambda x: x.get('timestamp', ''))
    
    for tx in sorted_transactions:
        try:
            timestamp = tx.get('timestamp')
            if not timestamp:
                continue
            
            # Set initial timestamp if not set
            if pool_states[0]['timestamp'] is None:
                pool_states[0]['timestamp'] = timestamp
                
            # Process events directly from tx_result
            events = tx.get('tx_result', {}).get('events', [])
            
            # Find wasm events for our contract
            for event in events:
                if event.get('type') == 'wasm':
                    # Convert attributes list to dict while preserving all values
                    attrs = {attr['key']: attr['value'] for attr in event.get('attributes', [])}
                    
                    # Check if this event is for our contract
                    if attrs.get('_contract_address') == contract_address:
                        action = attrs.get('action')
                        
                        if action == 'swap':
                            try:
                                # Get raw amounts
                                offer_amount = float(attrs.get('offer_amount', 0))
                                return_amount = float(attrs.get('return_amount', 0))
                                commission_amount = float(attrs.get('commission_amount', 0))
                                offer_asset = attrs.get('offer_asset', '')
                                ask_asset = attrs.get('ask_asset', '')
                                
                                # Convert to standard units (divide by 1e6)
                                offer_amount_std = offer_amount / 1e6
                                return_amount_std = return_amount / 1e6
                                commission_amount_std = commission_amount / 1e6
                                
                                # Record swap with standardized amounts
                                swaps.append({
                                    'timestamp': timestamp,
                                    'offer_amount': offer_amount_std,
                                    'return_amount': return_amount_std,
                                    'commission_amount': commission_amount_std,
                                    'offer_asset': offer_asset,
                                    'ask_asset': ask_asset,
                                    'tx_hash': tx['hash']
                                })
                                
                                # Update pool state after swap
                                last_state = pool_states[-1]
                                new_state = {
                                    'timestamp': timestamp,
                                    'xastro_amount': last_state['xastro_amount'],
                                    'eclipastro_amount': last_state['eclipastro_amount'],
                                    'event_type': 'swap',
                                    'tx_hash': tx['hash']
                                }
                                
                                if 'xASTRO' in offer_asset:
                                    new_state['xastro_amount'] += offer_amount_std
                                    new_state['eclipastro_amount'] -= return_amount_std
                                else:
                                    new_state['eclipastro_amount'] += offer_amount_std
                                    new_state['xastro_amount'] -= return_amount_std
                                
                                pool_states.append(new_state)
                                
                            except (ValueError, TypeError) as e:
                                st.write(f"Error processing swap amounts: {str(e)}")
                                continue
                            
                        elif action == 'provide_liquidity':
                            try:
                                # Debug output for liquidity provision
                                st.write("\nProcessing Liquidity Provision Transaction:")
                                st.write(f"Transaction Hash: {tx['hash']}")
                                
                                # Extract amounts from transfer events
                                transfer_events = [e for e in events if e.get('type') == 'transfer']
                                assets = [0, 0]  # [xASTRO, eclipASTRO]
                                
                                st.write("\nTransfer Events Found:")
                                for transfer in transfer_events:
                                    transfer_attrs = {attr['key']: attr['value'] for attr in transfer.get('attributes', [])}
                                    st.write(f"\nTransfer Attributes: {transfer_attrs}")
                                    
                                    if transfer_attrs.get('recipient') == contract_address:
                                        amount_str = transfer_attrs.get('amount', '')
                                        st.write(f"Amount String: {amount_str}")
                                        
                                        # Split combined amounts if present
                                        amounts = amount_str.split(',')
                                        for amount in amounts:
                                            try:
                                                # Extract numeric part before 'factory/'
                                                num_str = amount.split('factory/')[0].strip()
                                                num = float(num_str)
                                                
                                                # Determine token type and store amount
                                                if 'factory/neutron1zlf3hutsa4qnmue53lz2tfxrutp8y2e3rj4nkghg3rupgl4mqy8s5jgxsn' in amount:
                                                    assets[0] = num / 1_000_000  # xASTRO
                                                    st.write(f"Parsed xASTRO amount (in standard units): {assets[0]}")
                                                elif 'factory/neutron19lxlnqwzncg3dfty3umxvdxtgttnp3fefeqhmtg8xjrsz9vngnaq3yu8vn' in amount:
                                                    assets[1] = num / 1_000_000  # eclipASTRO
                                                    st.write(f"Parsed eclipASTRO amount (in standard units): {assets[1]}")
                                            except (ValueError, IndexError) as e:
                                                st.write(f"Error parsing amount '{amount}': {e}")
                                
                                st.write(f"\nFinal Parsed Amounts (in standard units) - xASTRO: {assets[0]}, eclipASTRO: {assets[1]}")
                                
                                if any(assets):
                                    liquidity_changes.append({
                                        'timestamp': timestamp,
                                        'xastro_amount': assets[0],
                                        'eclipastro_amount': assets[1],
                                        'tx_hash': tx['hash'],
                                        'type': 'provide'
                                    })
                                    
                                    # Update pool state
                                    last_state = pool_states[-1]
                                    pool_states.append({
                                        'timestamp': timestamp,
                                        'xastro_amount': last_state['xastro_amount'] + assets[0],
                                        'eclipastro_amount': last_state['eclipastro_amount'] + assets[1],
                                        'event_type': 'provide',
                                        'tx_hash': tx['hash']
                                    })
                            except (ValueError, TypeError) as e:
                                st.write(f"Error processing liquidity provision amounts: {str(e)}")
                                continue
                            
                        elif action == 'withdraw_liquidity':
                            try:
                                refund_assets = attrs.get('refund_assets', '').split(', ')
                                assets = [0, 0]  # [xASTRO, eclipASTRO]
                                
                                for asset in refund_assets:
                                    try:
                                        num_str = asset.split('factory/')[0].strip()
                                        amount = float(num_str) / 1_000_000  # Convert to standard units
                                        if 'xASTRO' in asset:
                                            assets[0] = amount
                                        elif 'eclipASTRO' in asset:
                                            assets[1] = amount
                                    except (ValueError, IndexError) as e:
                                        st.write(f"Error parsing withdrawal amount '{asset}': {e}")
                                        continue
                                
                                if any(assets):
                                    liquidity_changes.append({
                                        'timestamp': timestamp,
                                        'xastro_amount': -assets[0],
                                        'eclipastro_amount': -assets[1],
                                        'tx_hash': tx['hash'],
                                        'type': 'withdraw'
                                    })
                                    
                                    # Update pool state
                                    last_state = pool_states[-1]
                                    pool_states.append({
                                        'timestamp': timestamp,
                                        'xastro_amount': last_state['xastro_amount'] - assets[0],
                                        'eclipastro_amount': last_state['eclipastro_amount'] - assets[1],
                                        'event_type': 'withdraw',
                                        'tx_hash': tx['hash']
                                    })
                            except (ValueError, TypeError) as e:
                                st.write(f"Error processing liquidity withdrawal amounts: {str(e)}")
                                continue
                                
        except Exception as e:
            st.write(f"Error processing transaction {tx.get('hash', 'unknown')}: {str(e)}")
            continue
    
    st.write(f"Processed {len(swaps)} swaps and {len(liquidity_changes)} liquidity changes")
    
    # Create DataFrames with proper column structure
    swaps_df = pd.DataFrame(swaps)
    liquidity_df = pd.DataFrame(liquidity_changes)
    pool_states_df = pd.DataFrame(pool_states)
    
    if not swaps_df.empty:
        # Convert timestamp
        swaps_df['timestamp'] = pd.to_datetime(swaps_df['timestamp'])
        swaps_df['date'] = swaps_df['timestamp'].dt.date
        
        # Add delta columns for token movements
        swaps_df['xastro_delta'] = 0.0
        swaps_df['eclipastro_delta'] = 0.0
        
        # Calculate deltas based on swap direction
        for idx, row in swaps_df.iterrows():
            if 'xASTRO' in row['offer_asset']:
                swaps_df.at[idx, 'xastro_delta'] = row['offer_amount']
                swaps_df.at[idx, 'eclipastro_delta'] = -row['return_amount']
            else:
                swaps_df.at[idx, 'eclipastro_delta'] = row['offer_amount']
                swaps_df.at[idx, 'xastro_delta'] = -row['return_amount']
    
    if not liquidity_df.empty:
        # Convert timestamp
        liquidity_df['timestamp'] = pd.to_datetime(liquidity_df['timestamp'])
        liquidity_df['date'] = liquidity_df['timestamp'].dt.date
        
        # Filter out initial zero state
        first_nonzero = liquidity_df[
            (liquidity_df['xastro_amount'] != 0) | 
            (liquidity_df['eclipastro_amount'] != 0)
        ]['timestamp'].min()
        
        if pd.notna(first_nonzero):
            liquidity_df = liquidity_df[liquidity_df['timestamp'] >= first_nonzero]
    
    if not pool_states_df.empty:
        pool_states_df['timestamp'] = pd.to_datetime(pool_states_df['timestamp'])
        pool_states_df = pool_states_df.sort_values('timestamp')
        
        # Filter out initial zero state
        first_nonzero = pool_states_df[
            (pool_states_df['xastro_amount'] != 0) | 
            (pool_states_df['eclipastro_amount'] != 0)
        ]['timestamp'].min()
        
        if pd.notna(first_nonzero):
            pool_states_df = pool_states_df[pool_states_df['timestamp'] >= first_nonzero]
        
        pool_states_df['pool_ratio'] = pool_states_df['eclipastro_amount'] / pool_states_df['xastro_amount']
    
    return swaps_df, liquidity_df, pool_states_df

def create_dashboard():
    st.title("Neutron Astroport LP Analytics")
    
    # Contract address
    contract_address = CONTRACT_ADDRESS
    st.write(f"Analyzing contract: {contract_address}")
    
    # Add a button to refresh data
    if st.button("Refresh Data"):
        st.rerun()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Fetch transactions
    transactions = asyncio.run(fetch_transactions(progress_bar))
    
    if not transactions:
        st.error("No transactions found")
        return
        
    swaps_df, liquidity_df, pool_states_df = process_transactions(transactions, contract_address)
    
    # Display raw data for debugging
    st.subheader("Debug Information")
    
    if st.checkbox("Show example transaction"):
        example_tx = transactions[0] if transactions else "No transactions"
        st.write("Example transaction event structure:")
        st.json(example_tx)
    
    if st.checkbox("Show processed swap data"):
        st.write(swaps_df)
    
    if st.checkbox("Show processed liquidity data"):
        st.write(liquidity_df)
    
    if st.checkbox("Show pool states"):
        st.write(pool_states_df)
    
    # Create visualizations
    if not pool_states_df.empty:
        # Calculate ideal ratio (1.0) distance for color coding
        def get_ratio_impact_color(before_ratio, after_ratio):
            if before_ratio is None or after_ratio is None:
                return 'gray'  # Default color if we can't determine impact
            before_distance = abs(1 - before_ratio)
            after_distance = abs(1 - after_ratio)
            return 'green' if after_distance < before_distance else 'red'

        def get_surrounding_pool_states(timestamp, pool_states_df):
            prev_states = pool_states_df[pool_states_df['timestamp'] < timestamp]
            next_states = pool_states_df[pool_states_df['timestamp'] > timestamp]
            
            prev_state = prev_states.iloc[-1] if not prev_states.empty else None
            next_state = next_states.iloc[0] if not next_states.empty else None
            
            return prev_state, next_state

        # Swap Volume Analysis
        if not swaps_df.empty:
            st.subheader("Swap Analysis")
            
            # Calculate ratio before and after each swap
            swaps_df['prev_ratio'] = None
            swaps_df['next_ratio'] = None
            swaps_df['ratio_impact_color'] = None
            
            for idx, row in swaps_df.iterrows():
                prev_state, next_state = get_surrounding_pool_states(row['timestamp'], pool_states_df)
                
                prev_ratio = prev_state['pool_ratio'] if prev_state is not None else None
                next_ratio = next_state['pool_ratio'] if next_state is not None else None
                
                swaps_df.at[idx, 'prev_ratio'] = prev_ratio
                swaps_df.at[idx, 'next_ratio'] = next_ratio
                swaps_df.at[idx, 'ratio_impact_color'] = get_ratio_impact_color(prev_ratio, next_ratio)
            
            # Swap Volume Chart
            fig3 = go.Figure()
            
            # Add xASTRO swaps
            fig3.add_trace(
                go.Bar(
                    x=swaps_df['timestamp'],
                    y=abs(swaps_df['xastro_delta']),
                    name='xASTRO Volume',
                    marker_color=swaps_df['ratio_impact_color'],
                    customdata=np.stack((
                        swaps_df['prev_ratio'].fillna(-1),  # Use -1 for missing values
                        swaps_df['next_ratio'].fillna(-1)
                    ), axis=-1),
                    hovertemplate=(
                        'Time: %{x}<br>' +
                        'Volume: %{y:.6f}<br>' +
                        'Ratio Before: %{customdata[0]:.4f}<br>' +
                        'Ratio After: %{customdata[1]:.4f}<br>'
                    )
                )
            )
            
            fig3.update_layout(
                title='Swap Volumes (Color: Green = Improves Ratio Balance, Red = Worsens)',
                xaxis_title='Date',
                yaxis_title='Volume',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig3, use_container_width=True)

        # Pool Token Ratio Chart
        fig4 = go.Figure()
        
        fig4.add_trace(
            go.Scatter(
                x=pool_states_df['timestamp'],
                y=pool_states_df['pool_ratio'],
                name='eclipASTRO/xASTRO Ratio',
                mode='lines',
                line=dict(color='purple')
            )
        )
        
        # Add horizontal line at ratio = 1
        fig4.add_hline(
            y=1, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Ideal Ratio (1.0)",
            annotation_position="bottom right"
        )
        
        fig4.update_layout(
            title='Pool Token Ratio Over Time (eclipASTRO/xASTRO)',
            xaxis_title='Date',
            yaxis_title='Ratio',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Liquidity Events Chart
        if not liquidity_df.empty:
            # Calculate ratio impact for liquidity events
            liquidity_df['prev_ratio'] = None
            liquidity_df['next_ratio'] = None
            liquidity_df['ratio_impact_color'] = None
            
            for idx, row in liquidity_df.iterrows():
                prev_state, next_state = get_surrounding_pool_states(row['timestamp'], pool_states_df)
                
                prev_ratio = prev_state['pool_ratio'] if prev_state is not None else None
                next_ratio = next_state['pool_ratio'] if next_state is not None else None
                
                liquidity_df.at[idx, 'prev_ratio'] = prev_ratio
                liquidity_df.at[idx, 'next_ratio'] = next_ratio
                liquidity_df.at[idx, 'ratio_impact_color'] = get_ratio_impact_color(prev_ratio, next_ratio)
            
            fig5 = go.Figure()
            
            # xASTRO liquidity changes
            fig5.add_trace(
                go.Bar(
                    x=liquidity_df['timestamp'],
                    y=liquidity_df['xastro_amount'],
                    name='xASTRO',
                    marker_color=liquidity_df['ratio_impact_color'],
                    customdata=np.stack((
                        liquidity_df['prev_ratio'].fillna(-1),
                        liquidity_df['next_ratio'].fillna(-1)
                    ), axis=-1),
                    hovertemplate=(
                        'Time: %{x}<br>' +
                        'Amount: %{y:.6f}<br>' +
                        'Ratio Before: %{customdata[0]:.4f}<br>' +
                        'Ratio After: %{customdata[1]:.4f}<br>'
                    )
                )
            )
            
            # eclipASTRO liquidity changes
            fig5.add_trace(
                go.Bar(
                    x=liquidity_df['timestamp'],
                    y=liquidity_df['eclipastro_amount'],
                    name='eclipASTRO',
                    marker_color=liquidity_df['ratio_impact_color'],
                    customdata=np.stack((
                        liquidity_df['prev_ratio'].fillna(-1),
                        liquidity_df['next_ratio'].fillna(-1)
                    ), axis=-1),
                    hovertemplate=(
                        'Time: %{x}<br>' +
                        'Amount: %{y:.6f}<br>' +
                        'Ratio Before: %{customdata[0]:.4f}<br>' +
                        'Ratio After: %{customdata[1]:.4f}<br>'
                    )
                )
            )
            
            fig5.update_layout(
                title='Liquidity Events (Color: Green = Improves Ratio Balance, Red = Worsens)',
                xaxis_title='Date',
                yaxis_title='Amount Changed',
                height=500,
                showlegend=True,
                hovermode='x unified',
                barmode='group'
            )
            
            st.plotly_chart(fig5, use_container_width=True)
    
if __name__ == "__main__":
    st.set_page_config(page_title="Neutron Astroport LP Analytics", layout="wide")
    create_dashboard()