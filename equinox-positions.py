import streamlit as st
import json
import asyncio
import websockets.client
import requests
import plotly.graph_objects as go
import re
import base64
from urllib.parse import parse_qs
import time

# Constants
LOCKDROP_CONTRACT = "neutron1zh097hf7pz3d0pz3jnt3urhyw03kmcpxfs4as6sqz4cyfjkyyzmqpvd2n5"
LP_VAULT_CONTRACT = "neutron1d5p2lwh92040wfkrccdv5pamxtq7rsdzprfefd9v9vrh2c4lgheqvv6uyu"
SINGLE_VAULT_CONTRACT = "neutron1qk5nn9360pyu2tta7r4hvmuxwhxj5res79knt0sntmjcnwsycqyqy2ft9n"
VOTING_CONTRACT = "neutron19q93n64nyet24ynvw04qjqmejffkmyxakdvl08sf3n3yeyr92lrs2makhx"
LOCK_PERIODS = {
    0: "Flexible",
    2592000: "1 Month",
    7776000: "3 Months", 
    15552000: "6 Months"
}
LOCK_TIERS = {
    0: "1 Month",
    1: "3 Months",
    2: "6 Months",
    3: "9 Months",
    4: "12 Months"
}
PREFERRED_RPC = "wss://rpc-pulsarix.neutron-1.neutron.org/websocket"

async def get_rpc_endpoints():
    """Fetch RPC endpoints from Cosmos Chain Registry"""
    try:
        response = requests.get("https://raw.githubusercontent.com/cosmos/chain-registry/master/neutron/chain.json")
        data = response.json()
        
        # Convert HTTP(S) endpoints to WebSocket endpoints
        rpc_endpoints = []
        for rpc in data.get("apis", {}).get("rpc", []):
            address = rpc.get("address", "")
            if address.startswith("http"):
                ws_address = address.replace("http", "ws") + "/websocket"
                rpc_endpoints.append(ws_address)
        
        return rpc_endpoints if rpc_endpoints else ["wss://neutron-rpc.publicnode.com:443/websocket"]
    except Exception as e:
        st.error(f"Error fetching RPC endpoints: {str(e)}")
        return ["wss://neutron-rpc.publicnode.com:443/websocket"]

async def find_best_rpc():
    """Test all RPC endpoints and return them sorted by transaction count"""
    rpc_endpoints = await get_rpc_endpoints()
    endpoint_counts = []
    
    progress_bar = st.progress(0)
    
    for i, endpoint in enumerate(rpc_endpoints):
        count = await test_rpc_endpoint(endpoint)
        if count > 0:
            endpoint_counts.append((endpoint, count))
        progress_bar.progress((i + 1) / len(rpc_endpoints))
    
    progress_bar.empty()
    
    # Sort by transaction count in descending order
    endpoint_counts.sort(key=lambda x: x[1], reverse=True)
    
    if not endpoint_counts:
        st.error("No working RPC endpoints found!")
        return rpc_endpoints
    
    return [endpoint for endpoint, _ in endpoint_counts] + [ep for ep in rpc_endpoints if ep not in [e[0] for e in endpoint_counts]]

@st.cache_data(ttl=3600)
def get_cached_rpc_data():
    """Get cached RPC endpoints data"""
    if 'rpc_data' not in st.session_state:
        return None
    return st.session_state.rpc_data

def save_rpc_data(endpoints):
    """Save RPC endpoints data to cache"""
    st.session_state.rpc_data = {
        'endpoints': endpoints,
        'timestamp': int(time.time())
    }
    # Clear the cache to force update
    st.cache_data.clear()
    return endpoints

async def is_websocket_closed(websocket):
    """Check if websocket is closed"""
    try:
        await websocket.ping()
        return False
    except Exception:
        return True

async def test_rpc_endpoint(endpoint):
    """Test an RPC endpoint and return the number of transactions it reports"""
    try:
        websocket = await websockets.client.connect(endpoint, ping_interval=None)
        try:
            query = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "tx_search",
                "params": {
                    "query": f"execute._contract_address='{LOCKDROP_CONTRACT}'",
                    "prove": False,
                    "page": "1",
                    "per_page": "1",
                    "order_by": "asc"
                }
            }
            
            await websocket.send(json.dumps(query))
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            if "result" in data and "total_count" in data["result"]:
                return int(data["result"]["total_count"])
            return 0
        finally:
            await websocket.close()
    except Exception:
        return 0

async def connect_with_fallback(rpc_endpoints):
    """Try to connect to RPC endpoints with fallback"""
    for endpoint in rpc_endpoints:
        try:
            websocket = await websockets.client.connect(endpoint, ping_interval=None)
            return websocket
        except Exception:
            continue
    raise Exception("Failed to connect to any RPC endpoint")

async def query_contract(websocket, contract, query):
    """Query a smart contract using REST API"""
    try:
        # Convert query to base64
        query_base64 = base64.b64encode(json.dumps(query).encode()).decode()
        url = f"https://rest.cosmos.directory/neutron/cosmwasm/wasm/v1/contract/{contract}/smart/{query_base64}"
        
        response = requests.get(url)
        data = response.json()
        
        if "data" in data:
            return data
        return None
    except Exception as e:
        st.error(f"Error querying contract: {str(e)}")
        return None

async def fetch_lockdrop_info(websocket, wallet):
    """Fetch lockdrop participation info"""
    # Query LP vault
    lp_query = {
        "user_lp_lockup_info": {
            "user": wallet
        }
    }
    lp_data = await query_contract(websocket, LOCKDROP_CONTRACT, lp_query)
    lp_info = {}
    
    if lp_data and "data" in lp_data:
        for entry in lp_data["data"]:
            period = entry.get("duration", 0)  # Get lock period from response
            amount = float(entry.get("xastro_amount_in_lockups", "0")) / 1_000_000
            if amount > 0:
                lp_info[period] = amount
    
    # Query Single Sided vault
    single_query = {
        "user_single_lockup_info": {
            "user": wallet
        }
    }
    single_data = await query_contract(websocket, LOCKDROP_CONTRACT, single_query)
    single_info = {}
    
    if single_data and "data" in single_data:
        for entry in single_data["data"]:
            period = entry.get("duration", 0)  # Get lock period from response
            amount = float(entry.get("xastro_amount_in_lockups", "0")) / 1_000_000
            if amount > 0:
                single_info[period] = amount
    
    return lp_info, single_info

async def fetch_vault_info(websocket, wallet):
    """Fetch current vault positions"""
    # Query LP vault
    lp_query = {"staking": {"user": wallet}}
    lp_data = await query_contract(websocket, LP_VAULT_CONTRACT, lp_query)
    lp_amount = 0
    if lp_data and "data" in lp_data and "staked" in lp_data["data"]:
        lp_amount = float(lp_data["data"]["staked"]) / 1_000_000
    
    # Query Single Sided vault
    single_query = {"staking": {"user": wallet}}
    single_data = await query_contract(websocket, SINGLE_VAULT_CONTRACT, single_query)
    single_amounts = {}
    
    if single_data and "data" in single_data:
        for entry in single_data["data"]:
            duration = entry.get("duration", 0)
            staking_entries = entry.get("staking", [])
            total_amount = sum(float(stake.get("amount", "0")) for stake in staking_entries)
            if total_amount > 0:
                single_amounts[duration] = total_amount / 1_000_000
    
    return lp_amount, single_amounts

async def query_staker_info(websocket, wallet):
    """Fetch staker info from voting contract"""
    query = {
        "query_staker_info": {
            "staker": wallet
        }
    }
    return await query_contract(websocket, VOTING_CONTRACT, query)

async def query_total_essence():
    """Query total cosmic essence in existence"""
    url = "https://rest.cosmos.directory/neutron/cosmwasm/wasm/v1/contract/neutron19q93n64nyet24ynvw04qjqmejffkmyxakdvl08sf3n3yeyr92lrs2makhx/smart/eyJxdWVyeV90b3RhbF9lc3NlbmNlIjp7fX0="
    try:
        response = requests.get(url)
        data = response.json()
        return int(int(data['data']['essence']) // 1_000_000)
    except Exception as e:
        st.error(f"Error querying total cosmic essence: {str(e)}")
        return None

def process_voting_data(data):
    """Process voting power data from contract response"""
    if not data or 'data' not in data:
        return None
    
    data = data['data']
    staked = int(data.get('funds_to_unstake', 0)) / 1_000_000
    locked = int(data.get('funds_to_unlock', 0)) / 1_000_000
    
    # Process lock tier breakdown
    lock_breakdown = {}
    for info in data.get('locker_infos', []):
        tier = info.get('lock_tier')
        total_amount = sum(int(vault.get('amount', 0)) for vault in info.get('vaults', []))
        lock_breakdown[LOCK_TIERS[tier]] = total_amount / 1_000_000
    
    # Get cosmic essence
    essence_info = data.get('essence_and_rewards_info', {})
    cosmic_essence = int(essence_info.get('essence', 0)) / 1_000_000
    
    return {
        'staked': staked,
        'locked': locked,
        'lock_breakdown': lock_breakdown,
        'cosmic_essence': cosmic_essence
    }

def plot_distribution(data, title):
    """Create a bar chart for distribution visualization"""
    periods = list(LOCK_PERIODS.keys())
    period_labels = list(LOCK_PERIODS.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=period_labels,
            y=[data.get(period, 0) for period in periods],
            text=[f"{data.get(period, 0):.2f}" for period in periods],
            textposition='none',  # Hide text inside bars
            marker_color='rgb(55, 83, 109)'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Lock Period",
        yaxis_title="xASTRO Amount",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2
    )
    
    # Add grid lines and style axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def plot_lock_distribution(lock_breakdown):
    """Create a bar chart for lock period distribution"""
    periods = list(lock_breakdown.keys())
    amounts = list(lock_breakdown.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=periods,
            y=amounts,
            marker_color='rgb(55, 83, 109)'
        )
    ])
    
    fig.update_layout(
        title="ECLIP Lock Period Distribution",
        xaxis_title="Lock Period",
        yaxis_title="ECLIP Amount",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

async def main():
    st.title("Equinox Position Tracker")
    
    # Get wallet address from URL query parameter if present
    default_wallet = st.query_params.get('wallet', '')
    
    # Input field for wallet address
    wallet = st.text_input("Enter Neutron wallet address:", value=default_wallet)
    
    # Initialize session state for RPC endpoints if not already done
    if 'healthy_rpc_endpoints' not in st.session_state:
        # Try preferred endpoint first
        try:
            websocket = await websockets.client.connect(PREFERRED_RPC, ping_interval=None)
            await websocket.close()
            st.session_state.healthy_rpc_endpoints = [PREFERRED_RPC]
        except Exception:
            # Fallback to endpoint discovery
            with st.spinner("Finding healthy RPC endpoints..."):
                endpoints = await find_best_rpc()
                if not endpoints:
                    return
                st.session_state.healthy_rpc_endpoints = endpoints
    
    if wallet:
        # Validate wallet address
        if not re.match(r'^neutron1[a-zA-Z0-9]{38,39}$', wallet):
            st.error("Invalid Neutron wallet address")
            return
        
        try:
            websocket = await connect_with_fallback(st.session_state.healthy_rpc_endpoints)
            
            try:
                # Fetch lockdrop participation
                st.header("Lockdrop Participation")
                
                lp_info, single_info = await fetch_lockdrop_info(websocket, wallet)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plot_distribution(lp_info, "LP Vault Distribution"), use_container_width=True)
                
                with col2:
                    st.plotly_chart(plot_distribution(single_info, "Single Sided Vault Distribution"), use_container_width=True)
                
                # Fetch current vault positions
                st.header("Post-lockdrop Participation")
                
                lp_amount, single_amounts = await fetch_vault_info(websocket, wallet)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("LP Vault")
                    st.metric("Staked xASTRO", f"{lp_amount:.6f}")
                
                with col4:
                    st.subheader("Single Sided Vault")
                    st.plotly_chart(plot_distribution(single_amounts, "Current Staking Distribution"), use_container_width=True)
                
                # Fetch voting power info
                st.header("Voting Power")
                
                voting_data = await query_staker_info(websocket, wallet)
                voting_info = process_voting_data(voting_data)
                
                if voting_info:
                    col5, col6 = st.columns(2)
                    
                    with col5:
                        st.subheader("Staked ECLIP")
                        st.metric("Staked", f"{int(voting_info['staked']):,}")
                    
                    with col6:
                        st.subheader("Locked ECLIP")
                        st.metric("Locked", f"{int(voting_info['locked']):,}")
                    
                    st.subheader("Lock Tier Breakdown")
                    st.plotly_chart(plot_lock_distribution(voting_info['lock_breakdown']), use_container_width=True)
                    
                    st.subheader("Cosmic Essence")
                    st.metric("Cosmic Essence", f"{int(voting_info['cosmic_essence']):,}")
                    
                    # Calculate and display voting power percentage
                    total_essence = await query_total_essence()
                    if total_essence:
                        voting_power_pct = (voting_info['cosmic_essence'] / total_essence) * 100
                        st.metric("Voting Power", f"{voting_power_pct:.2f}%")
            
            finally:
                await websocket.close()
        
        except Exception as e:
            st.error(f"Error connecting to RPC endpoint: {str(e)}")
            # Clear the RPC endpoints to force a retest on next run
            if 'healthy_rpc_endpoints' in st.session_state:
                del st.session_state.healthy_rpc_endpoints

if __name__ == "__main__":
    asyncio.run(main())
