import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import websockets
import asyncio
import requests
from collections import defaultdict
import base64
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
import webbrowser
import pymongo

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

async def fetch_transactions(progress_bar, start_timestamp=None):
    """Fetch contract transactions using WebSocket connection."""
    transactions = []
    page = 1
    per_page = 25
    total_fetched = 0
    max_retries = 3
    
    try:
        # Get MongoDB connection for checking existing transactions
        client = pymongo.MongoClient(st.secrets["mongo"]["connection_string"])
        db = client["shannon-test"]
        collection = db["equinox-lp-txns"]
        
        async with websockets.connect(WS_ENDPOINT, ping_interval=None) as websocket:
            # Build query - always start from most recent
            query_str = f"wasm._contract_address='{CONTRACT_ADDRESS}'"

            # Get total count first
            initial_query = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "tx_search",
                "params": {
                    "query": query_str,
                    "prove": False,
                    "page": "1",
                    "per_page": "1",
                    "order_by": "desc"  # Ensure we're getting newest first
                }
            }
            
            await websocket.send(json.dumps(initial_query))
            initial_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            initial_data = json.loads(initial_response)
            
            if "result" not in initial_data:
                st.error(f"Invalid response format: {initial_data}")
                return []
                
            total_count = int(initial_data["result"].get("total_count", 0))
            if total_count == 0:
                return []
                
            st.write(f"Total transactions available: {total_count}")
            found_existing = False
            
            while not found_existing:
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        query = {
                            "jsonrpc": "2.0",
                            "id": page,
                            "method": "tx_search",
                            "params": {
                                "query": query_str,
                                "prove": False,
                                "page": str(page),
                                "per_page": str(per_page),
                                "order_by": "desc"  # Newest first
                            }
                        }
                        
                        await websocket.send(json.dumps(query))
                        response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                        data = json.loads(response)
                        
                        if "result" in data and "txs" in data["result"]:
                            new_txs = data["result"]["txs"]
                            
                            if not new_txs:  # No more transactions
                                success = True
                                break
                            
                            # Process transactions in smaller batches
                            batch_size = 5
                            for i in range(0, len(new_txs), batch_size):
                                batch = new_txs[i:i + batch_size]
                                tx_batch = []
                                
                                for tx in batch:
                                    # Check if transaction already exists in database
                                    existing_tx = collection.find_one({"hash": tx.get('hash')})
                                    if existing_tx:
                                        found_existing = True
                                        st.info(f"Found existing transaction at page {page}, stopping fetch")
                                        break
                                    
                                    tx_data = {
                                        'height': tx.get('height'),
                                        'hash': tx.get('hash'),
                                        'tx_result': tx.get('tx_result', {})
                                    }
                                    tx_batch.append(tx_data)
                                
                                if found_existing:
                                    break
                                
                                if tx_batch:  # Only process if we have new transactions
                                    # Fetch timestamps in parallel with limited concurrency
                                    with ThreadPoolExecutor(max_workers=5) as executor:
                                        heights = [tx['height'] for tx in tx_batch]
                                        timestamps = list(executor.map(get_block_time, heights))
                                    
                                    # Add timestamps to transactions
                                    for tx, timestamp in zip(tx_batch, timestamps):
                                        if timestamp:
                                            tx['timestamp'] = timestamp
                                            transactions.append(tx)
                                    
                                    # Update progress based on current page
                                    progress = min((page * per_page) / total_count, 1.0)
                                    progress_bar.progress(progress)
                                
                                # Add a small delay between batches
                                await asyncio.sleep(0.1)
                            
                            success = True
                            
                            if found_existing:
                                break
                            
                            page += 1
                            
                    except asyncio.TimeoutError:
                        retry_count += 1
                        st.warning(f"Timeout on page {page}, attempt {retry_count} of {max_retries}")
                        await asyncio.sleep(1)  # Wait before retrying
                        continue
                    except Exception as e:
                        retry_count += 1
                        st.error(f"Error processing page {page}, attempt {retry_count} of {max_retries}: {str(e)}")
                        await asyncio.sleep(1)  # Wait before retrying
                        continue
                
                if not success:
                    st.error(f"Failed to process page {page} after {max_retries} attempts")
                    break
                
                if found_existing:
                    break
                
    except Exception as e:
        st.error(f"WebSocket error: {str(e)}")
        return []
    finally:
        if 'client' in locals():
            client.close()
    
    if transactions:
        st.success(f"Successfully fetched {len(transactions)} new transactions")
    return transactions

def parse_refund_assets(refund_assets_str):
    """Parse refund_assets string to extract xASTRO and eclipASTRO amounts."""
    amounts = {'xastro_amount': 0, 'eclipastro_amount': 0}
    if not refund_assets_str:
        return amounts
        
    assets = refund_assets_str.split(', ')
    for asset in assets:
        try:
            # Extract amount (everything before 'factory/')
            amount_str = asset.split('factory/')[0].strip()
            amount = float(amount_str)
            
            # Determine asset type
            if 'xASTRO' in asset:
                amounts['xastro_amount'] = amount
            elif 'eclipASTRO' in asset:
                amounts['eclipastro_amount'] = amount
        except (ValueError, IndexError) as e:
            st.write(f"Error parsing refund asset '{asset}': {e}")
            continue
    
    return amounts

def process_transactions(transactions, contract_address):
    """Process transactions to extract swap and liquidity data."""
    swaps = []
    all_liquidity_events = []
    
    # Initialize pool states with zero
    pool_states = [{
        'timestamp': None,
        'xastro_amount': 0,
        'eclipastro_amount': 0,
        'event_type': 'initial',
        'pool_ratio': 1.0
    }]
    
    # Sort transactions by timestamp
    sorted_transactions = sorted(transactions, key=lambda x: x.get('timestamp', ''))
    
    # Track running totals
    running_xastro = 0
    running_eclipastro = 0
    
    for tx in sorted_transactions:
        try:
            timestamp = tx.get('timestamp')
            if not timestamp:
                continue
            
            # Set initial timestamp if not set
            if pool_states[0]['timestamp'] is None:
                pool_states[0]['timestamp'] = timestamp
            
            # Process events from stored transaction format
            for event in tx.get('events', []):
                try:
                    event_type = event.get('type')
                    
                    if event_type == 'swap':
                        try:
                            # Get raw amounts and convert to standard units
                            offer_amount = float(event.get('offer_amount', 0)) / 1e6
                            return_amount = float(event.get('return_amount', 0)) / 1e6
                            commission_amount = float(event.get('commission_amount', 0)) / 1e6
                            offer_asset = event.get('offer_asset', '')
                            ask_asset = event.get('ask_asset', '')
                            
                            swaps.append({
                                'timestamp': timestamp,
                                'offer_amount': offer_amount,
                                'return_amount': return_amount,
                                'commission_amount': commission_amount,
                                'offer_asset': offer_asset,
                                'ask_asset': ask_asset,
                                'tx_hash': tx['hash'],
                                'sender': event.get('sender', '')
                            })
                            
                            if 'xASTRO' in offer_asset:
                                running_xastro += offer_amount
                                running_eclipastro -= return_amount
                            else:
                                running_eclipastro += offer_amount
                                running_xastro -= return_amount
                            
                        except (ValueError, TypeError) as e:
                            st.write(f"Error processing swap amounts: {str(e)}")
                            continue
                            
                    elif event_type == 'provide_liquidity':
                        try:
                            # Get amounts and convert to standard units
                            xastro_amount = float(event.get('xastro_amount', 0)) / 1e6
                            eclipastro_amount = float(event.get('eclipastro_amount', 0)) / 1e6
                            
                            event_data = {
                                'timestamp': timestamp,
                                'type': 'provide',
                                'xastro_amount': xastro_amount,
                                'eclipastro_amount': eclipastro_amount,
                                'tx_hash': tx['hash'],
                                'sender': event.get('sender', '')
                            }
                            all_liquidity_events.append(event_data)
                            
                            running_xastro += xastro_amount
                            running_eclipastro += eclipastro_amount
                            
                        except Exception as e:
                            st.write(f"Error processing liquidity provision: {str(e)}")
                            continue
                            
                    elif event_type == 'withdraw_liquidity':
                        try:
                            # Parse refund_assets string to get amounts
                            amounts = parse_refund_assets(event.get('refund_assets', ''))
                            xastro_amount = amounts['xastro_amount'] / 1e6
                            eclipastro_amount = amounts['eclipastro_amount'] / 1e6
                            
                            event_data = {
                                'timestamp': timestamp,
                                'type': 'withdraw',
                                'xastro_amount': -xastro_amount,  # Store as negative for withdrawal
                                'eclipastro_amount': -eclipastro_amount,  # Store as negative for withdrawal
                                'tx_hash': tx['hash'],
                                'sender': event.get('sender', '')
                            }
                            all_liquidity_events.append(event_data)
                            
                            running_xastro -= xastro_amount
                            running_eclipastro -= eclipastro_amount
                            
                        except Exception as e:
                            st.write(f"Error processing withdrawal: {str(e)}")
                            continue
                    
                    # After each event, update pool state
                    if running_eclipastro > 0:
                        pool_ratio = running_xastro / running_eclipastro
                    else:
                        pool_ratio = 1.0
                    
                    pool_states.append({
                        'timestamp': timestamp,
                        'xastro_amount': running_xastro,
                        'eclipastro_amount': running_eclipastro,
                        'event_type': event_type,
                        'pool_ratio': pool_ratio,
                        'tx_hash': tx['hash']
                    })
                    
                except Exception as e:
                    st.write(f"Error processing event in tx {tx.get('hash', 'unknown')}: {str(e)}")
                    continue
                    
        except Exception as e:
            st.write(f"Error processing transaction {tx.get('hash', 'unknown')}: {str(e)}")
            continue
    
    # Convert to DataFrames
    swaps_df = pd.DataFrame(swaps)
    all_liquidity_df = pd.DataFrame(all_liquidity_events)
    
    # Filter out the first liquidity event
    if not all_liquidity_df.empty:
        all_liquidity_df['timestamp'] = pd.to_datetime(all_liquidity_df['timestamp'])
        
        # Find the first provide liquidity event with both assets
        first_lp_mask = (all_liquidity_df['type'] == 'provide') & \
                       (all_liquidity_df['xastro_amount'] > 0) & \
                       (all_liquidity_df['eclipastro_amount'] > 0)
        
        if first_lp_mask.any():
            first_lp_time = all_liquidity_df[first_lp_mask]['timestamp'].min()
            liquidity_df = all_liquidity_df[all_liquidity_df['timestamp'] > first_lp_time].copy()
        else:
            liquidity_df = all_liquidity_df.copy()
    else:
        liquidity_df = all_liquidity_df.copy()
    
    # Convert pool states to DataFrame
    pool_states_df = pd.DataFrame(pool_states)
    
    # Add xastro_delta and eclipastro_delta to swaps_df
    if not swaps_df.empty:
        swaps_df['xastro_delta'] = swaps_df.apply(
            lambda row: -row['offer_amount'] if 'xASTRO' in row['offer_asset'] 
            else row['return_amount'], axis=1
        )
        swaps_df['eclipastro_delta'] = swaps_df.apply(
            lambda row: -row['offer_amount'] if 'eclipASTRO' in row['offer_asset'] 
            else row['return_amount'], axis=1
        )
        swaps_df['timestamp'] = pd.to_datetime(swaps_df['timestamp'])
    
    # Convert timestamps for pool states
    if not pool_states_df.empty:
        pool_states_df['timestamp'] = pd.to_datetime(pool_states_df['timestamp'])
        
        if not liquidity_df.empty:
            min_time = liquidity_df['timestamp'].min()
            pool_states_df = pool_states_df[pool_states_df['timestamp'] >= min_time].copy()
            
        # Fix the deprecation warning by using fillna instead of replace
        if 'pool_ratio' in pool_states_df.columns:
            pool_states_df['pool_ratio'] = pool_states_df['pool_ratio'].ffill()
    
    return swaps_df, liquidity_df, pool_states_df

def get_swap_rates():
    """Fetch swap rates from MongoDB."""
    try:
        # Get MongoDB connection string from streamlit secrets
        mongo_uri = st.secrets["mongo"]["connection_string"]
        client = pymongo.MongoClient(mongo_uri)
        db = client["shannon-test"]
        collection = db["swap_rates"]
        
        # Fetch all records and convert to DataFrame
        cursor = collection.find({})
        swap_rates = list(cursor)
        
        if not swap_rates:
            return pd.DataFrame()
            
        df = pd.DataFrame(swap_rates)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create separate dataframes for each direction
        xastro_to_eclip = df[df['from_token'].str.contains('xASTRO')]
        eclip_to_xastro = df[df['from_token'].str.contains('eclipASTRO')]
        
        return xastro_to_eclip, eclip_to_xastro
        
    except Exception as e:
        st.error(f"Error fetching swap rates: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def get_astroport_price_data():
    client = pymongo.MongoClient(st.secrets["mongo"]["connection_string"])
    db = client['shannon-test']
    collection = db['astroport-price-data']
    
    # Query all documents and sort by timestamp
    cursor = collection.find({}).sort('timestamp', 1)
    
    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))
    
    # Convert timestamp from MongoDB format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def get_latest_transaction_timestamp():
    """Get the timestamp of the most recent transaction in MongoDB."""
    try:
        client = pymongo.MongoClient(st.secrets["mongo"]["connection_string"])
        db = client["shannon-test"]
        collection = db["equinox-lp-txns"]
        
        # Get the most recent transaction
        latest_tx = collection.find_one(
            sort=[("timestamp", pymongo.DESCENDING)]
        )
        
        return latest_tx["timestamp"] if latest_tx else None
        
    except Exception as e:
        st.error(f"Error fetching latest transaction timestamp: {str(e)}")
        return None

def get_stored_transactions(start_timestamp=None):
    """Retrieve transactions from MongoDB."""
    try:
        client = pymongo.MongoClient(st.secrets["mongo"]["connection_string"])
        db = client["shannon-test"]
        collection = db["equinox-lp-txns"]
        
        # Build query
        query = {}
        if start_timestamp:
            query["timestamp"] = {"$gte": start_timestamp}
            
        # Get transactions sorted by timestamp
        cursor = collection.find(query).sort("timestamp", 1)
        transactions = list(cursor)
        
        # Convert MongoDB _id to string to make transactions JSON serializable
        for tx in transactions:
            if '_id' in tx:
                tx['_id'] = str(tx['_id'])
        
        return transactions
        
    except Exception as e:
        st.error(f"Error retrieving transactions from MongoDB: {str(e)}")
        return []

def store_transactions(transactions):
    """Store transactions in MongoDB."""
    try:
        client = pymongo.MongoClient(st.secrets["mongo"]["connection_string"])
        db = client["shannon-test"]
        collection = db["equinox-lp-txns"]
        
        # Convert transactions to a format suitable for MongoDB
        mongo_transactions = []
        for tx in transactions:
            # Extract basic transaction info
            mongo_tx = {
                'height': tx.get('height'),
                'hash': tx.get('hash'),
                'timestamp': tx.get('timestamp'),
                'events': []
            }
            
            # Process events
            events = tx.get('tx_result', {}).get('events', [])
            for event in events:
                if event.get('type') == 'wasm':
                    attrs = {attr['key']: attr['value'] for attr in event.get('attributes', [])}
                    if attrs.get('_contract_address') == CONTRACT_ADDRESS:
                        action = attrs.get('action')
                        
                        if action == 'swap':
                            event_data = {
                                'type': 'swap',
                                'sender': attrs.get('sender', ''),
                                'offer_amount': attrs.get('offer_amount', ''),
                                'return_amount': attrs.get('return_amount', ''),
                                'commission_amount': attrs.get('commission_amount', ''),
                                'offer_asset': attrs.get('offer_asset', ''),
                                'ask_asset': attrs.get('ask_asset', '')
                            }
                            mongo_tx['events'].append(event_data)
                        elif action == 'provide_liquidity':
                            # Extract amounts from transfer events
                            transfer_events = [e for e in events if e.get('type') == 'transfer']
                            assets = {'xastro': 0, 'eclipastro': 0}  # Track both assets
                            
                            for transfer in transfer_events:
                                transfer_attrs = {attr['key']: attr['value'] for attr in transfer.get('attributes', [])}
                                if transfer_attrs.get('recipient') == CONTRACT_ADDRESS:
                                    amount_str = transfer_attrs.get('amount', '')
                                    amounts = amount_str.split(',')
                                    
                                    for amount in amounts:
                                        try:
                                            # Extract numeric part and asset type
                                            num_str = amount.split('factory/')[0].strip()
                                            num = float(num_str)
                                            
                                            if 'xASTRO' in amount:
                                                assets['xastro'] = num  # Store raw amount
                                            elif 'eclipASTRO' in amount:
                                                assets['eclipastro'] = num  # Store raw amount
                                                
                                        except (ValueError, IndexError) as e:
                                            st.write(f"Error parsing amount '{amount}': {e}")
                            
                            event_data = {
                                'type': 'provide_liquidity',
                                'sender': attrs.get('sender', ''),
                                'xastro_amount': assets['xastro'],  # Store raw amount
                                'eclipastro_amount': assets['eclipastro']  # Store raw amount
                            }
                            mongo_tx['events'].append(event_data)
                        elif action == 'withdraw_liquidity':
                            event_data = {
                                'type': 'withdraw_liquidity',
                                'sender': attrs.get('sender', ''),
                                'refund_assets': attrs.get('refund_assets', '')
                            }
                            mongo_tx['events'].append(event_data)
            
            if mongo_tx['events']:  # Only store transactions with relevant events
                mongo_transactions.append(mongo_tx)
        
        if mongo_transactions:
            # Use update_many with upsert to handle duplicates
            operations = [
                pymongo.UpdateOne(
                    {"hash": tx["hash"]},
                    {"$set": tx},
                    upsert=True
                )
                for tx in mongo_transactions
            ]
            result = collection.bulk_write(operations)
            st.write(f"Stored {len(mongo_transactions)} transactions in database")
            
    except Exception as e:
        st.error(f"Error storing transactions in MongoDB: {str(e)}")

def create_dashboard():
    st.title("Neutron Astroport LP Analytics")
    
    # Contract address
    contract_address = CONTRACT_ADDRESS
    
    # Add a button to refresh data
    if st.button("Refresh Data"):
        st.rerun()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Fetch latest transaction timestamp from MongoDB
    latest_timestamp = get_latest_transaction_timestamp()
    
    # Get stored transactions and fetch new ones if needed
    stored_transactions = get_stored_transactions()
    
    if latest_timestamp:
        new_transactions = asyncio.run(fetch_transactions(progress_bar, start_timestamp=latest_timestamp))
        if new_transactions:
            store_transactions(new_transactions)
            transactions = get_stored_transactions()
        else:
            transactions = stored_transactions
    else:
        transactions = asyncio.run(fetch_transactions(progress_bar))
        if transactions:
            store_transactions(transactions)
            transactions = get_stored_transactions()
    
    if not transactions:
        st.error("No transactions found")
        return

    swaps_df, liquidity_df, pool_states_df = process_transactions(transactions, contract_address)
    
    # Get Astroport price data and prepare it
    astro_price_df = get_astroport_price_data()
    
    if not pool_states_df.empty and not astro_price_df.empty:
        # Convert all timestamps to UTC and make them timezone-naive
        pool_states_df['timestamp'] = pd.to_datetime(pool_states_df['timestamp']).dt.tz_localize(None)
        astro_price_df['timestamp'] = pd.to_datetime(astro_price_df['timestamp']).dt.tz_localize(None)
        
        # Create a time-based index for interpolation
        astro_price_df.set_index('timestamp', inplace=True)
        astro_price_df = astro_price_df.resample('1min').ffill()  # Resample to minute intervals
        
        # Get xASTRO:ASTRO ratio for each pool state timestamp
        pool_states_df['xastro_astro_ratio'] = pool_states_df['timestamp'].map(
            lambda x: astro_price_df['xastro_astro_ratio'].asof(x) if x >= astro_price_df.index.min() else 1.0
        ).fillna(1.0)  # Fill any NaN values with 1.0
        
        # Multiply pool ratio by xASTRO:ASTRO ratio
        pool_states_df['adjusted_pool_ratio'] = pool_states_df['pool_ratio'] * pool_states_df['xastro_astro_ratio']
    
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
            # Ensure timestamp is pandas Timestamp
            timestamp = pd.to_datetime(timestamp)
            if timestamp.tz is not None:
                timestamp = timestamp.tz_localize(None)
            
            prev_states = pool_states_df[pool_states_df['timestamp'] < timestamp]
            next_states = pool_states_df[pool_states_df['timestamp'] > timestamp]
            
            prev_state = prev_states.iloc[-1] if not prev_states.empty else None
            next_state = next_states.iloc[0] if not next_states.empty else None
            
            return prev_state, next_state

        # Swap Volume Analysis
        if not swaps_df.empty:
            # Ensure swaps_df timestamps are consistent
            swaps_df['timestamp'] = pd.to_datetime(swaps_df['timestamp']).dt.tz_localize(None)
            
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
            
            # Swap Volume Chart with Pool Ratio
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add pool ratio line first with high zorder
            fig3.add_trace(
                go.Scatter(
                    x=pool_states_df['timestamp'],
                    y=pool_states_df['adjusted_pool_ratio'],
                    name='Adjusted Pool Ratio',
                    mode='lines',
                    line=dict(color='purple', width=1.5),
                    hovertemplate='Ratio: %{y:.4f}<extra></extra>',
                    legendrank=1,  # Make it appear first in legend
                    zorder=1000  # Ensure it renders on top
                ),
                secondary_y=False
            )

            # Add horizontal line at ratio = 1
            fig3.add_hline(
                y=1, line_dash="dash", line_color="gray",
                annotation_text="1:1 Ratio", 
                annotation_position="bottom right"
            )

            # Create empty lists to store all the data points
            sell_times = []
            sell_values = []
            sell_hashes = []
            buy_times = []
            buy_values = []
            buy_hashes = []
            
            # Collect all data points first
            for idx, row in swaps_df.iterrows():
                if 'xASTRO' in row['offer_asset']:
                    # xASTRO sell (positive)
                    sell_times.append(row['timestamp'])
                    sell_values.append(abs(row['xastro_delta']))  # Make positive
                    sell_hashes.append(row['tx_hash'])
                else:
                    # xASTRO buy (negative)
                    buy_times.append(row['timestamp'])
                    buy_values.append(-abs(row['return_amount']))  # Make negative
                    buy_hashes.append(row['tx_hash'])
            
            # Add all sells as one trace
            if sell_times:
                # Split sells into rebalancing and other sells
                rebalance_sells = swaps_df[
                    (swaps_df['offer_asset'].str.contains('xASTRO')) & 
                    (swaps_df['sender'] == 'neutron1f5qswtwykvgyerzw2uav3kl7kku4mzwq34zhrl')
                ]
                other_sells = swaps_df[
                    (swaps_df['offer_asset'].str.contains('xASTRO')) & 
                    (swaps_df['sender'] != 'neutron1f5qswtwykvgyerzw2uav3kl7kku4mzwq34zhrl')
                ]
                
                # Add rebalancing sells
                if not rebalance_sells.empty:
                    fig3.add_trace(
                        go.Bar(
                            x=rebalance_sells['timestamp'],
                            y=abs(rebalance_sells['xastro_delta']),  # Make positive
                            name='Rebalancing Sell xASTRO',
                            marker_color='lightblue',
                            width=3600000,  # 1 hour in milliseconds
                            customdata=rebalance_sells['tx_hash'].values,
                            hovertemplate=(
                                'Time: %{x}<br>' +
                                'Amount: %{y:.6f}<br>' +
                                'TX Hash: %{customdata}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        secondary_y=True
                    )
                
                # Add other sells
                if not other_sells.empty:
                    fig3.add_trace(
                        go.Bar(
                            x=other_sells['timestamp'],
                            y=abs(other_sells['xastro_delta']),  # Make positive
                            name='Sell xASTRO',
                            marker_color='green',
                            width=3600000,  # 1 hour in milliseconds
                            customdata=other_sells['tx_hash'].values,
                            hovertemplate=(
                                'Time: %{x}<br>' +
                                'Amount: %{y:.6f}<br>' +
                                'TX Hash: %{customdata}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        secondary_y=True
                    )
            
            # Add all buys as one trace
            if buy_times:
                # Split buys into rebalancing and other buys
                rebalance_buys = swaps_df[
                    (swaps_df['offer_asset'].str.contains('eclipASTRO')) & 
                    (swaps_df['sender'] == 'neutron1f5qswtwykvgyerzw2uav3kl7kku4mzwq34zhrl')
                ]
                other_buys = swaps_df[
                    (swaps_df['offer_asset'].str.contains('eclipASTRO')) & 
                    (swaps_df['sender'] != 'neutron1f5qswtwykvgyerzw2uav3kl7kku4mzwq34zhrl')
                ]
                
                # Add rebalancing buys
                if not rebalance_buys.empty:
                    fig3.add_trace(
                        go.Bar(
                            x=rebalance_buys['timestamp'],
                            y=-abs(rebalance_buys['return_amount']),  # Make negative
                            name='Rebalancing Buy xASTRO',
                            marker_color='lightblue',
                            width=3600000,  # 1 hour in milliseconds
                            customdata=rebalance_buys['tx_hash'].values,
                            hovertemplate=(
                                'Time: %{x}<br>' +
                                'Amount: %{y:.6f}<br>' +
                                'TX Hash: %{customdata}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        secondary_y=True
                    )
                
                # Add other buys
                if not other_buys.empty:
                    fig3.add_trace(
                        go.Bar(
                            x=other_buys['timestamp'],
                            y=-abs(other_buys['return_amount']),  # Make negative
                            name='Buy xASTRO',
                            marker_color='red',
                            width=3600000,  # 1 hour in milliseconds
                            customdata=other_buys['tx_hash'].values,
                            hovertemplate=(
                                'Time: %{x}<br>' +
                                'Amount: %{y:.6f}<br>' +
                                'TX Hash: %{customdata}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        secondary_y=True
                    )
            
            fig3.update_layout(
                title='xASTRO Swap Volumes and Pool Ratio',
                height=500,
                showlegend=True,
                hovermode='x unified',
                dragmode=False,
                clickmode='event',
                barmode='relative',  # Ensure proper bar stacking
                xaxis=dict(layer='below traces'),  # Force axis below all traces
                yaxis=dict(layer='below traces'),
                yaxis2=dict(layer='below traces')
            )
            
            fig3.update_xaxes(title_text='Date')
            fig3.update_yaxes(title_text='Adjusted Pool Ratio (xASTRO/eclipASTRO)', secondary_y=False)
            fig3.update_yaxes(title_text='xASTRO Volume', secondary_y=True)
            
            # Create the chart with click events enabled
            st.plotly_chart(
                fig3,
                use_container_width=True,
                config={
                    'displayModeBar': False,
                    'scrollZoom': False,
                    'staticPlot': False,
                    'doubleClick': False
                }
            )

            # Liquidity Events Chart with Pool Ratio
            if not liquidity_df.empty:
                fig5 = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Only show xASTRO liquidity changes
                xastro_adds = liquidity_df[liquidity_df['xastro_amount'] > 0]
                xastro_removes = liquidity_df[liquidity_df['xastro_amount'] < 0]
                
                # Split adds into rebalancing and other adds
                rebalance_adds = xastro_adds[xastro_adds['sender'] == 'neutron1f5qswtwykvgyerzw2uav3kl7kku4mzwq34zhrl']
                other_adds = xastro_adds[xastro_adds['sender'] != 'neutron1f5qswtwykvgyerzw2uav3kl7kku4mzwq34zhrl']
                
                # Split removes into rebalancing and other removes
                rebalance_removes = xastro_removes[xastro_removes['sender'] == 'neutron1f5qswtwykvgyerzw2uav3kl7kku4mzwq34zhrl']
                other_removes = xastro_removes[xastro_removes['sender'] != 'neutron1f5qswtwykvgyerzw2uav3kl7kku4mzwq34zhrl']
                
                # Add pool ratio line first with high zorder
                fig5.add_trace(
                    go.Scatter(
                        x=pool_states_df['timestamp'],
                        y=pool_states_df['adjusted_pool_ratio'],
                        name='Adjusted Pool Ratio',
                        mode='lines',
                        line=dict(color='purple', width=1.5),
                        hovertemplate='Ratio: %{y:.4f}<extra></extra>',
                        legendrank=1,  # Make it appear first in legend
                        zorder=1000  # Ensure it renders on top
                    ),
                    secondary_y=False
                )

                # Add horizontal line at ratio = 1
                fig5.add_hline(
                    y=1, line_dash="dash", line_color="gray",
                    annotation_text="1:1 Ratio", 
                    annotation_position="bottom right"
                )

                # Update y-axes ranges and titles
                fig5.update_yaxes(
                    title_text='Adjusted Pool Ratio (xASTRO/eclipASTRO)',
                    secondary_y=False
                )
                fig5.update_yaxes(
                    title_text='xASTRO Amount',
                    secondary_y=True,
                    range=[-500000, 500000]  # Set range from -0.5M to 0.5M
                )
                
                # Add rebalancing adds
                if not rebalance_adds.empty:
                    fig5.add_trace(
                        go.Bar(
                            x=rebalance_adds['timestamp'],
                            y=rebalance_adds['xastro_amount'],
                            name='Rebalancing Add xASTRO',
                            marker_color='lightblue',
                            width=3600000,  # 1 hour in milliseconds
                            customdata=rebalance_adds['tx_hash'].values,
                            hovertemplate=(
                                'Time: %{x}<br>' +
                                'Amount: %{y:.6f}<br>' +
                                'TX Hash: %{customdata}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        secondary_y=True
                    )
                
                # Add other adds
                if not other_adds.empty:
                    fig5.add_trace(
                        go.Bar(
                            x=other_adds['timestamp'],
                            y=other_adds['xastro_amount'],
                            name='Add xASTRO',
                            marker_color='green',
                            width=3600000,  # 1 hour in milliseconds
                            customdata=other_adds['tx_hash'].values,
                            hovertemplate=(
                                'Time: %{x}<br>' +
                                'Amount: %{y:.6f}<br>' +
                                'TX Hash: %{customdata}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        secondary_y=True
                    )
                
                # Add rebalancing removes
                if not rebalance_removes.empty:
                    fig5.add_trace(
                        go.Bar(
                            x=rebalance_removes['timestamp'],
                            y=rebalance_removes['xastro_amount'],
                            name='Rebalancing Remove xASTRO',
                            marker_color='lightblue',
                            width=3600000,  # 1 hour in milliseconds
                            customdata=rebalance_removes['tx_hash'].values,
                            hovertemplate=(
                                'Time: %{x}<br>' +
                                'Amount: %{y:.6f}<br>' +
                                'TX Hash: %{customdata}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        secondary_y=True
                    )
                
                # Add other removes
                if not other_removes.empty:
                    fig5.add_trace(
                        go.Bar(
                            x=other_removes['timestamp'],
                            y=other_removes['xastro_amount'],
                            name='Remove xASTRO',
                            marker_color='red',
                            width=3600000,  # 1 hour in milliseconds
                            customdata=other_removes['tx_hash'].values,
                            hovertemplate=(
                                'Time: %{x}<br>' +
                                'Amount: %{y:.6f}<br>' +
                                'TX Hash: %{customdata}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        secondary_y=True
                    )
                
                fig5.update_layout(
                    title='xASTRO Liquidity Events and Pool Ratio',
                    height=500,
                    showlegend=True,
                    hovermode='x unified',
                    barmode='relative',  # Ensure proper bar stacking
                    dragmode=False,
                    clickmode='event',
                    xaxis=dict(layer='below traces'),  # Force axis below all traces
                    yaxis=dict(layer='below traces'),
                    yaxis2=dict(layer='below traces')
                )
                
                fig5.update_xaxes(title_text='Date')
                fig5.update_yaxes(title_text='Adjusted Pool Ratio (xASTRO/eclipASTRO)', secondary_y=False)
                fig5.update_yaxes(title_text='xASTRO Amount', secondary_y=True)
                
                # Create the chart with click events enabled
                st.plotly_chart(
                    fig5,
                    use_container_width=True,
                    config={
                        'displayModeBar': False,
                        'scrollZoom': False,
                        'staticPlot': False,
                        'doubleClick': False
                    }
                )

                # Add custom click handler
                st.markdown("""
                <style>
                .dragcover { pointer-events: none !important; }
                </style>
                <script>
                const plotDiv2 = document.querySelector('.js-plotly-plot:nth-child(2)');
                if (plotDiv2) {
                    plotDiv2.on('plotly_click', function(data) {
                        if (data && data.points && data.points[0] && data.points[0].customdata) {
                            const url = 'https://neutron.celat.one/neutron-1/txs/' + data.points[0].customdata;
                            window.open(url, '_blank');
                        }
                    });
                    
                    // Remove dragcover when it appears
                    const observer2 = new MutationObserver(function(mutations) {
                        mutations.forEach(function(mutation) {
                            mutation.addedNodes.forEach(function(node) {
                                if (node.className === 'dragcover') {
                                    node.remove();
                                }
                            });
                        });
                    });
                    
                    observer2.observe(document.body, { childList: true, subtree: true });
                }
                </script>
                """, unsafe_allow_html=True)
        
        # Pool Composition Percentage Chart
        fig6 = go.Figure()
        
        # Calculate total and percentages
        pool_states_df['total_amount'] = pool_states_df['xastro_amount'] + pool_states_df['eclipastro_amount']
        pool_states_df['xastro_pct'] = pool_states_df['xastro_amount'] / pool_states_df['total_amount'] * 100
        pool_states_df['eclipastro_pct'] = pool_states_df['eclipastro_amount'] / pool_states_df['total_amount'] * 100
        
        # Add xASTRO area
        fig6.add_trace(
            go.Scatter(
                x=pool_states_df['timestamp'],
                y=pool_states_df['xastro_pct'],
                name='xASTRO',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(0, 255, 0, 0.5)',  # Green
                line=dict(width=0.5),
                customdata=np.stack((
                    pool_states_df['xastro_amount'],
                    pool_states_df['total_amount']
                ), axis=-1),
                hovertemplate=(
                    'Time: %{x}<br>' +
                    'xASTRO: %{customdata[0]:.6f}<br>' +
                    'Percentage: %{y:.1f}%<br>' +
                    'Total Pool: %{customdata[1]:.6f}<br>'
                )
            )
        )
        
        # Add eclipASTRO area
        fig6.add_trace(
            go.Scatter(
                x=pool_states_df['timestamp'],
                y=pool_states_df['eclipastro_pct'],
                name='eclipASTRO',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(0, 0, 255, 0.5)',  # Blue
                line=dict(width=0.5),
                customdata=np.stack((
                    pool_states_df['eclipastro_amount'],
                    pool_states_df['total_amount']
                ), axis=-1),
                hovertemplate=(
                    'Time: %{x}<br>' +
                    'eclipASTRO: %{customdata[0]:.6f}<br>' +
                    'Percentage: %{y:.1f}%<br>' +
                    'Total Pool: %{customdata[1]:.6f}<br>'
                )
            )
        )
        
        # Add a horizontal line at 50%
        fig6.add_hline(
            y=50,
            line_dash="dash",
            line_color="white",
            annotation_text="50% Balance",
            annotation_position="right",
            annotation_font_color="white"
        )
        
        fig6.update_layout(
            title='Pool Composition Over Time (Percentage)',
            xaxis_title='Date',
            yaxis_title='Percentage of Pool',
            height=500,
            showlegend=True,
            hovermode='x unified',
            yaxis=dict(
                range=[40, 60],  # Set Y-axis range to 40-60%
                ticksuffix='%'
            )
        )
        
        st.plotly_chart(fig6, use_container_width=True)
    
    # Add Swap Rates Chart
    st.subheader("Exchange Rates Analysis")
    
    # Fetch swap rates data
    xastro_to_eclip, eclip_to_xastro = get_swap_rates()
    
    if not xastro_to_eclip.empty and not eclip_to_xastro.empty:
        fig_rates = go.Figure()
        
        # Add xASTRO to eclipASTRO exchange rate
        fig_rates.add_trace(
            go.Scatter(
                x=xastro_to_eclip['timestamp'],
                y=xastro_to_eclip['exchange_rate'],
                name='xASTRO → eclipASTRO',
                line=dict(color='blue'),
                visible='legendonly'  # Hide by default but allow toggling through legend
            )
        )
        
        # Add eclipASTRO to xASTRO exchange rate
        fig_rates.add_trace(
            go.Scatter(
                x=eclip_to_xastro['timestamp'],
                y=eclip_to_xastro['exchange_rate'],
                name='eclipASTRO → xASTRO',
                line=dict(color='red')
            )
        )
        
        # Update layout
        fig_rates.update_layout(
            title='Exchange Rates Over Time',
            xaxis_title='Time',
            yaxis_title='Exchange Rate',
            hovermode='x unified',
            showlegend=True
        )
        
        # Add horizontal line at ratio = 1
        fig_rates.add_hline(
            y=1,
            line_dash="dash",
            line_color="gray",
            secondary_y=False
        )
        
        st.plotly_chart(fig_rates, use_container_width=True)
        
        # Display some statistics
        st.write("Latest Exchange Rates:")
        latest_xastro_to_eclip = xastro_to_eclip.iloc[-1]
        latest_eclip_to_xastro = eclip_to_xastro.iloc[-1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "xASTRO → eclipASTRO",
                f"{latest_xastro_to_eclip['exchange_rate']:.6f}",
                f"Impact: {latest_xastro_to_eclip['price_impact']:.2f}%"
            )
        with col2:
            st.metric(
                "eclipASTRO → xASTRO",
                f"{latest_eclip_to_xastro['exchange_rate']:.6f}",
                f"Impact: {latest_eclip_to_xastro['price_impact']:.2f}%"
            )
    else:
        st.warning("No swap rates data available")
        
    # Add Astroport Price Charts
    astroport_data = get_astroport_price_data()
    
    if not astroport_data.empty:
        # ASTRO Price Chart
        fig_astro = go.Figure()
        fig_astro.add_trace(
            go.Scatter(
                x=astroport_data['timestamp'],
                y=astroport_data['astro_price_usd'],
                mode='lines',
                name='ASTRO Price',
                line=dict(color='blue', width=1.5),
                hovertemplate='Time: %{x}<br>Price: $%{y:.4f}<extra></extra>'
            )
        )
        
        fig_astro.update_layout(
            title='ASTRO Price (USD)',
            height=400,
            showlegend=True,
            hovermode='x unified',
            yaxis_title='Price (USD)',
            xaxis_title='Date'
        )
        
        st.plotly_chart(fig_astro, use_container_width=True)
        
        # xASTRO:ASTRO Ratio Chart
        fig_ratio = go.Figure()
        fig_ratio.add_trace(
            go.Scatter(
                x=astroport_data['timestamp'],
                y=astroport_data['xastro_astro_ratio'],
                mode='lines',
                name='xASTRO:ASTRO Ratio',
                line=dict(color='green', width=1.5),
                hovertemplate='Time: %{x}<br>Ratio: %{y:.4f}<extra></extra>'
            )
        )
        
        fig_ratio.update_layout(
            title='xASTRO:ASTRO Exchange Rate',
            height=400,
            showlegend=True,
            hovermode='x unified',
            yaxis_title='Ratio',
            xaxis_title='Date'
        )
        
        st.plotly_chart(fig_ratio, use_container_width=True)
        
if __name__ == "__main__":
    st.set_page_config(page_title="Neutron Astroport LP Analytics", layout="wide")
    create_dashboard()