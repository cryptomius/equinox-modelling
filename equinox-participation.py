import streamlit as st
import websockets
import asyncio
import json
import nest_asyncio
import pandas as pd
from collections import defaultdict
import base64
import requests
import plotly.graph_objects as go

# Set page to wide mode
st.set_page_config(layout="wide")

# Enable nested event loops
nest_asyncio.apply()

# Constants
WS_ENDPOINT = "wss://neutron-rpc.publicnode.com:443/websocket"
CONTRACT_ADDRESS = "neutron1zh097hf7pz3d0pz3jnt3urhyw03kmcpxfs4as6sqz4cyfjkyyzmqpvd2n5"
LOCK_PERIODS = {
    0: "Flexible",
    2592000: "1 Month",
    7776000: "3 Months", 
    15552000: "6 Months"
}

async def fetch_sender_addresses(progress_bar):
    unique_senders = set()
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
                    "query": f"execute._contract_address='{CONTRACT_ADDRESS}'",
                    "prove": False,
                    "page": "1",
                    "per_page": "1",
                    "order_by": "asc"
                }
            }
            
            await websocket.send(json.dumps(initial_query))
            initial_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            initial_data = json.loads(initial_response)
            
            total_count = int(initial_data["result"].get("total_count", 0))
            st.write(f"Total transactions to process: {total_count}")
            
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
                    
                    if "result" in data and "txs" in data["result"]:
                        new_txs = data["result"]["txs"]
                        
                        if not new_txs:  # No more transactions
                            break
                        
                        # Process each transaction
                        for tx in new_txs:
                            events = tx.get('tx_result', {}).get('events', [])
                            
                            # Find first event with "spender" attribute
                            for event in events:
                                attrs = event.get('attributes', [])
                                for attr in attrs:
                                    if attr.get('key') == 'spender':
                                        spender_value = attr.get('value')
                                        if spender_value:
                                            unique_senders.add(spender_value)
                                        break
                        
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
        return set()
    
    st.write(f"Found {len(unique_senders)} unique participant wallets")
    return unique_senders

async def fetch_lockup_info(websocket, wallet, query_type):
    query_msg = {
        f"user_{query_type}_lockup_info": {
            "user": wallet
        }
    }
    
    query_msg_base64 = base64.b64encode(json.dumps(query_msg).encode()).decode()
    url = f"https://rest.cosmos.directory/neutron/cosmwasm/wasm/v1/contract/{CONTRACT_ADDRESS}/smart/{query_msg_base64}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "data" in data:
            return data.get("data", [])
        return []
    except Exception as e:
        st.error(f"Error fetching {query_type} lockup info for {wallet}: {str(e)}")
        return []

async def fetch_all_lockup_info(unique_senders, progress_bar):
    wallet_positions = defaultdict(lambda: {
        'single': defaultdict(float),
        'lp': defaultdict(float)
    })
    
    total_queries = len(unique_senders) * 2  # Two queries per wallet
    queries_completed = 0
    
    for wallet in unique_senders:
        # Fetch LP lockup info
        lp_positions = await fetch_lockup_info(None, wallet, "lp")
        for pos in lp_positions:
            duration = pos.get("duration", 0)
            amount = float(pos.get("xastro_amount_in_lockups", "0")) / 1_000_000
            period = LOCK_PERIODS.get(duration, "Unknown")
            wallet_positions[wallet]['lp'][period] += amount
        
        queries_completed += 1
        progress_bar.progress(queries_completed / total_queries)
        
        # Fetch Single lockup info
        single_positions = await fetch_lockup_info(None, wallet, "single")
        for pos in single_positions:
            duration = pos.get("duration", 0)
            amount = float(pos.get("xastro_amount_in_lockups", "0")) / 1_000_000
            period = LOCK_PERIODS.get(duration, "Unknown")
            wallet_positions[wallet]['single'][period] += amount
        
        queries_completed += 1
        progress_bar.progress(queries_completed / total_queries)
    
    return wallet_positions

def create_positions_dataframe(wallet_positions):
    rows = []
    for wallet, positions in wallet_positions.items():
        row = {
            'wallet': wallet,
            'single_flexible': positions['single']['Flexible'],
            'single_1month': positions['single']['1 Month'],
            'single_3months': positions['single']['3 Months'],
            'single_6months': positions['single']['6 Months'],
            'lp_flexible': positions['lp']['Flexible'],
            'lp_1month': positions['lp']['1 Month'],
            'lp_3months': positions['lp']['3 Months'],
            'lp_6months': positions['lp']['6 Months']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['total_single'] = df[[c for c in df.columns if c.startswith('single_')]].sum(axis=1)
    df['total_lp'] = df[[c for c in df.columns if c.startswith('lp_')]].sum(axis=1)
    df['total_xastro'] = df['total_single'] + df['total_lp']
    
    return df.sort_values('total_xastro', ascending=False)

async def main():
    st.title("xASTRO Positions Dashboard")
    
    # Fetch unique senders first
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Fetching participant addresses...")
    unique_senders = await fetch_sender_addresses(progress_bar)
    
    if not unique_senders:
        st.error("No participant addresses found!")
        return
        
    # Now fetch lockup info for each sender
    status_text.text("Fetching lockup positions...")
    progress_bar.progress(0)
    wallet_positions = await fetch_all_lockup_info(unique_senders, progress_bar)
    
    if not wallet_positions:
        st.error("No lockup positions found!")
        return
    
    # Create DataFrame
    status_text.text("Processing data...")
    df = create_positions_dataframe(wallet_positions)
    
    # Summary statistics at the top
    st.header("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Participants", f"{len(df):,}")
    with col2:
        st.metric("Total xASTRO Locked", f"{df['total_xastro'].sum():,.2f}")
    with col3:
        st.metric("Single Sided Total", f"{df['total_single'].sum():,.2f}")
    with col4:
        st.metric("LP Total", f"{df['total_lp'].sum():,.2f}")
    
    # Create charts
    st.header("Vault Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Single Sided Vault Chart
        single_data = {
            'Flexible': df['single_flexible'].sum(),
            '1 Month': df['single_1month'].sum(),
            '3 Months': df['single_3months'].sum(),
            '6 Months': df['single_6months'].sum()
        }
        
        fig_single = go.Figure(data=[
            go.Bar(
                x=list(single_data.keys()),
                y=list(single_data.values()),
                name='Single Sided Vault'
            )
        ])
        
        fig_single.update_layout(
            title='Single Sided Vault Distribution',
            yaxis_title='xASTRO Amount',
            showlegend=False
        )
        
        st.plotly_chart(fig_single, use_container_width=True)
    
    with col2:
        # LP Vault Chart
        lp_data = {
            'Flexible': df['lp_flexible'].sum(),
            '1 Month': df['lp_1month'].sum(),
            '3 Months': df['lp_3months'].sum(),
            '6 Months': df['lp_6months'].sum()
        }
        
        fig_lp = go.Figure(data=[
            go.Bar(
                x=list(lp_data.keys()),
                y=list(lp_data.values()),
                name='LP Vault'
            )
        ])
        
        fig_lp.update_layout(
            title='LP Vault Distribution',
            yaxis_title='xASTRO Amount',
            showlegend=False
        )
        
        st.plotly_chart(fig_lp, use_container_width=True)
    
    # Participant Positions table at the bottom
    st.header("Participant Positions")
    st.dataframe(df)
    
    # Clear the status text and progress bar
    status_text.empty()
    progress_bar.empty()

if __name__ == "__main__":
    asyncio.run(main())
