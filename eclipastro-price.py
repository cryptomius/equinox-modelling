import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, timezone

# Page config
st.set_page_config(
    page_title="EclipASTRO peg",
    page_icon="📈",
    layout="wide"
)

# Auto-refresh every 5 minutes
st.markdown(
    """
    <meta http-equiv="refresh" content="300">
    """,
    unsafe_allow_html=True
)

# MongoDB connection
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["connection_string"])

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_price_data():
    client = init_connection()
    db = client["shannon-test"]
    
    # Define the minimum start timestamp and 90-day window
    min_start_time = datetime.fromisoformat("2024-11-19T06:15:34.499+00:00")
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=90)
    
    # Use the later of the two dates
    query_start_time = max(min_start_time, thirty_days_ago)
    
    data = list(db["astroport-price-data"].find(
        {
            "timestamp": {"$gte": query_start_time},
            "eclipastro_price_usd": {"$exists": True}
        },
        {
            "timestamp": 1,
            "eclipastro_price_usd": 1,
            "astro_price_usd": 1,
            "_id": 0
        }
    ).sort("timestamp", 1))
    
    return pd.DataFrame(data)

# App title
st.title("EclipASTRO peg")

try:
    # Load the data
    df = get_price_data()
    
    if not df.empty:
        # Get most recent prices
        latest_prices = df.iloc[-1]
        eclipastro_price = latest_prices['eclipastro_price_usd']
        astro_price = latest_prices['astro_price_usd']
        
        # Calculate depeg percentage
        depeg_percentage = ((eclipastro_price / astro_price) - 1) * 100
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "eclipASTRO Price",
                f"${eclipastro_price:.6f}"
            )
        
        with col2:
            st.metric(
                "ASTRO Price",
                f"${astro_price:.6f}"
            )
        
        with col3:
            st.metric(
                "Depeg",
                f"{depeg_percentage:+.2f}%",
                help="Percentage difference between eclipASTRO and ASTRO prices. Positive means eclipASTRO is trading above ASTRO."
            )
        
        # Melt the dataframe to create a format suitable for multiple lines
        df_melted = pd.melt(
            df,
            id_vars=['timestamp'],
            value_vars=['eclipastro_price_usd', 'astro_price_usd'],
            var_name='token',
            value_name='price'
        )
        
        # Create the time series chart
        fig = px.line(
            df_melted,
            x="timestamp",
            y="price",
            color="token",
            title="Token Prices (USD)",
            template="plotly_dark",
            labels={
                "price": "Price (USD)",
                "timestamp": "Date",
                "token": "Token"
            }
        )
        
        # Customize the chart
        fig.update_layout(
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update legend labels with correct capitalization
        name_mapping = {
            'eclipastro_price_usd': 'eclipASTRO',
            'astro_price_usd': 'ASTRO'
        }
        fig.for_each_trace(lambda t: t.update(
            name=name_mapping[t.name],
            hovertemplate='$%{y:.6f}'
        ))
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate depeg percentage over time
        df['depeg_percentage'] = ((df['eclipastro_price_usd'] / df['astro_price_usd']) - 1) * 100
        
        # Calculate 10-period SMA of depeg percentage
        df['depeg_percentage_sma'] = df['depeg_percentage'].rolling(window=10).mean()
        
        # Create the depeg percentage area chart
        fig_depeg = px.area(
            df,
            x="timestamp",
            y="depeg_percentage_sma",
            title="Depeg Percentage Over Time (10-period SMA)",
            template="plotly_dark",
            labels={
                "depeg_percentage_sma": "Depeg % (10-period SMA)",
                "timestamp": "Date"
            }
        )
        
        # Customize the depeg chart
        fig_depeg.update_layout(
            hovermode="x unified",
            showlegend=False
        )
        
        # Update hover template
        fig_depeg.update_traces(
            hovertemplate='%{y:.2f}%'
        )
        
        # Add a horizontal line at 0%
        fig_depeg.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="white", 
            opacity=0.5
        )
        
        # Display the depeg chart
        st.plotly_chart(fig_depeg, use_container_width=True)
        
        # Add footnote about auto-update
        st.caption("📊 Chart automatically updates every 5 minutes")
        
    else:
        st.warning("No price data available for the selected time period.")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.error("Please check your MongoDB connection string in Streamlit secrets.")
