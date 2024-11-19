import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, timezone

# Page config
st.set_page_config(
    page_title="eclipASTRO Price History",
    page_icon="📈",
    layout="wide"
)

# MongoDB connection
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["connection_string"])

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_price_data():
    client = init_connection()
    db = client["shannon-test"]
    
    # Query the last 30 days of data
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    
    data = list(db["astroport-price-data"].find(
        {
            "timestamp": {"$gte": thirty_days_ago},
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
st.title("Token Price History")

try:
    # Load the data
    df = get_price_data()
    
    if not df.empty:
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
    
    else:
        st.warning("No price data available for the selected time period.")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.error("Please check your MongoDB connection string in Streamlit secrets.")
