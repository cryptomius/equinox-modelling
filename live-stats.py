import streamlit as st
import requests
import json
import base64
from typing import Dict, List

# Constants
CONTRACT_ADDRESS = "neutron1zh097hf7pz3d0pz3jnt3urhyw03kmcpxfs4as6sqz4cyfjkyyzmqpvd2n5"
BASE_URL = "https://rest-kralum.neutron-1.neutron.org"
DURATION_LABELS = {
    0: "Flexible",
    2592000: "1 Month",
    7776000: "3 Months",
    15552000: "6 Months"
}

def query_contract(query: Dict) -> Dict:
    """Query the smart contract with base64 encoded query."""
    query_string = json.dumps(query)
    query_base64 = base64.b64encode(query_string.encode()).decode()
    endpoint = f"{BASE_URL}/cosmwasm/wasm/v1/contract/{CONTRACT_ADDRESS}/smart/{query_base64}"
    response = requests.get(endpoint)
    if response.status_code != 200:
        raise Exception(f"API returned status code {response.status_code}: {response.text}")
    return response.json()

def format_number(number: float) -> str:
    """Format number with thousands separator and no decimals."""
    return f"{int(number):,}"

def calculate_percentages(values: List[float]) -> List[float]:
    """Calculate percentages from a list of values."""
    total = sum(values)
    return [value / total * 100 if total > 0 else 0 for value in values]

def display_metric_with_percent(label: str, value: float, percentage: float):
    """Display a metric with a percentage below it without delta arrows."""
    st.metric(label, format_number(value))
    st.markdown(f"<div style='text-align: center; color: #737373;'>{percentage:.0f}%</div>", unsafe_allow_html=True)

def display_lockup_section(data: Dict, title: str, total_committed: float):
    """Display a lockup section with its breakdown."""
    st.subheader(title)
    
    # Extract and process values
    lockups = data['data'].get('single_lockups' if 'single_lockups' in data['data'] else 'lp_lockups', [])
    values = [float(lockup['xastro_amount_in_lockups']) / 1_000_000 for lockup in lockups]
    percentages = calculate_percentages(values)
    
    # Display total and percentage of total committed
    section_total = sum(values)
    total_percentage = (section_total / total_committed * 100) if total_committed > 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total xASTRO", format_number(section_total))
    with col2:
        st.metric("% of Total Committed", f"{total_percentage:.0f}%")
    
    # Create columns for each duration
    cols = st.columns(4)
    
    # Display each duration's data
    for i, (value, percentage) in enumerate(zip(values, percentages)):
        with cols[i]:
            display_metric_with_percent(DURATION_LABELS[lockups[i]['duration']], value, percentage)

def display_incentives_section():
    """Display incentives section with Single Sided and LP Vault values."""
    st.header("Incentives")
    
    # Query incentives data
    single_sided = query_contract({"incentives": {"stake_type": "single_staking"}})
    lp_vault = query_contract({"incentives": {"stake_type": "lp_staking"}})
    
    # Calculate totals
    total_beclip = (float(single_sided['data']['beclip']) + float(lp_vault['data']['beclip'])) / 1_000_000
    total_eclip = (float(single_sided['data']['eclip']) + float(lp_vault['data']['eclip'])) / 1_000_000
    
    # Display totals first
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total bECLIP", format_number(total_beclip))
    with col2:
        st.metric("Total ECLIP", format_number(total_eclip))
    
    st.markdown("---")  # Add a visual separator
    
    # Create two columns for Single Sided and LP Vault
    col1, col2 = st.columns(2)
    
    # Display Single Sided incentives
    with col1:
        st.subheader("Single Sided")
        ss_beclip = float(single_sided['data']['beclip']) / 1_000_000
        ss_eclip = float(single_sided['data']['eclip']) / 1_000_000
        st.metric("bECLIP", format_number(ss_beclip))
        st.metric("ECLIP", format_number(ss_eclip))
    
    # Display LP Vault incentives
    with col2:
        st.subheader("LP Vault")
        lp_beclip = float(lp_vault['data']['beclip']) / 1_000_000
        lp_eclip = float(lp_vault['data']['eclip']) / 1_000_000
        st.metric("bECLIP", format_number(lp_beclip))
        st.metric("ECLIP", format_number(lp_eclip))

def main():
    st.set_page_config(page_title="Equinox Lockdrop", layout="wide")
    st.title("Equinox Lockdrop")
    
    try:
        # Query both datasets first
        single_sided_data = query_contract({"single_lockup_info": {}})
        lp_data = query_contract({"lp_lockup_info": {}})
        
        # Calculate total committed xASTRO first
        single_sided_total = sum(float(lockup['xastro_amount_in_lockups']) / 1_000_000 
                               for lockup in single_sided_data['data']['single_lockups'])
        lp_total = sum(float(lockup['xastro_amount_in_lockups']) / 1_000_000 
                      for lockup in lp_data['data']['lp_lockups'])
        
        total_committed = single_sided_total + lp_total
        
        # Display total committed at the top
        st.header("Total Committed xASTRO")
        st.metric("Total", format_number(total_committed))
        
        st.markdown("---")  # Add a visual separator
        
        # Display detailed sections
        with st.expander("Single Sided Raw Data"):
            st.json(single_sided_data)
        display_lockup_section(single_sided_data, "Single Sided Staking xASTRO", total_committed)
        
        st.markdown("---")  # Add a visual separator
        
        with st.expander("LP Staking Raw Data"):
            st.json(lp_data)
        display_lockup_section(lp_data, "LP Staking xASTRO", total_committed)
        
        st.markdown("---")  # Add a visual separator
        
        # Display Incentives section at the bottom
        display_incentives_section()
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()