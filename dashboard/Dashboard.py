import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime



# Page configuration
st.set_page_config(
    page_title="Crime Maps Dashboard 2022",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .tab-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .filter-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f4;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Crimes_-_2022_20250925.csv")
        
    # Convert date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Hour'] = df['Date'].dt.hour
        df['Month'] = df['Date'].dt.month
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Month_Name'] = df['Date'].dt.month_name()
    
    # Clean coordinates
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df = df.dropna(subset=['Latitude', 'Longitude'])
        df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
        df = df[(df['Latitude'].between(41.6, 42.1)) & (df['Longitude'].between(-87.9,-87.5))]
    
    # Handle boolean columns
    for bool_col in ['Arrest', 'Domestic']:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].fillna(False).astype(bool)
    
    # Create time periods
    if 'Hour' in df.columns:
        df['Time_Period'] = pd.cut(df['Hour'], 
                                    bins=[0, 6, 12, 18, 24], 
                                    labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
                                    include_lowest=True)
    
    # Handle missing values
    df['Primary Type'] = df['Primary Type'].fillna('UNKNOWN')
    df['Community Area'] = df['Community Area'].fillna('Unknown Area')
    df.dropna(inplace=True)
    
    return df
    

def main():
    # Header
    st.markdown('<h1 class="main-header">🗺️ Crime Maps Analytics Dashboard 2022</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data")
        return
    
    # Debug information
    with st.expander("📊 Data Information", expanded=False):
        st.write(f"**Total rows:** {len(df):,}")
        st.write(f"**Available columns:** {', '.join(df.columns.tolist())}")
        st.write(f"**Crime types:** {df['Primary Type'].nunique()} unique types")
    
    # Filter Section
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    st.markdown("### 🔍 **Map Filters**")
    
    # First row of filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        crime_types = ['All'] + sorted(df['Primary Type'].unique().tolist())
        selected_crime_type = st.selectbox("🔴 Crime Type", crime_types)
    
    with col2:
        time_periods = ['All'] + sorted(df['Time_Period'].dropna().astype(str).unique().tolist())
        selected_time_period = st.selectbox("⏰ Time Period", time_periods)
    
    with col3:
        arrest_filter = st.selectbox("🚔 Arrest Status", ['All', 'Arrested', 'Not Arrested'])
    
    with col4:
        domestic_filter = st.selectbox("🏠 Domestic Violence", ['All', 'Domestic', 'Non-Domestic'])
    
    # Second row of filters
    col5, col6 = st.columns(2)
    
    with col5:
        if 'Date' in df.columns:
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            date_range = st.date_input("📅 Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            date_range = None
    
    with col6:
        months = ['All'] + sorted(df['Month_Name'].unique().tolist())
        selected_month = st.selectbox("📆 Month", months)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_crime_type != 'All':
        filtered_df = filtered_df[filtered_df['Primary Type'] == selected_crime_type]
    
    if selected_time_period != 'All':
        filtered_df = filtered_df[filtered_df['Time_Period'].astype(str) == selected_time_period]
    
    if arrest_filter == 'Arrested':
        filtered_df = filtered_df[filtered_df['Arrest'] == True]
    elif arrest_filter == 'Not Arrested':
        filtered_df = filtered_df[filtered_df['Arrest'] == False]
    
    if domestic_filter == 'Domestic':
        filtered_df = filtered_df[filtered_df['Domestic'] == True]
    elif domestic_filter == 'Non-Domestic':
        filtered_df = filtered_df[filtered_df['Domestic'] == False]
    
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['Month_Name'] == selected_month]
    
    # Key metrics cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{len(filtered_df):,}</h3><p>Total Crimes</p></div>', unsafe_allow_html=True)
    
    with col2:
        arrest_rate = (filtered_df['Arrest'] == True).mean() * 100 if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-card"><h3>{arrest_rate:.1f}%</h3><p>Arrest Rate</p></div>', unsafe_allow_html=True)
    
    with col3:
        domestic_rate = (filtered_df['Domestic'] == True).mean() * 100 if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-card"><h3>{domestic_rate:.1f}%</h3><p>Family-Related</p></div>', unsafe_allow_html=True)
    
    with col4:
        unique_areas = filtered_df['Community Area'].nunique() if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-card"><h3>{unique_areas}</h3><p>Areas</p></div>', unsafe_allow_html=True)
    
    with col5:
        peak_hour = int(filtered_df['Hour'].mode().iloc[0]) if len(filtered_df) > 0 else 12
        st.markdown(f'<div class="metric-card"><h3>{peak_hour}:00</h3><p>Peak Hour</p></div>', unsafe_allow_html=True)
    
    with col6:
            if len(filtered_df) > 0:
                filtered_df["Date"] = pd.to_datetime(filtered_df["Date"], errors="coerce")
                filtered_df["Month"] = filtered_df["Date"].dt.month_name()

                month_counts = filtered_df["Month"].value_counts()

                top_month = month_counts.idxmax()
                top_month_count = month_counts.max()
            else:
                top_month, top_month_count = "N/A", 0

            st.markdown(
                f'<div class="metric-card"><h3>{top_month_count:,}</h3><p>Crimes in {top_month}</p></div>',
                unsafe_allow_html=True
            )
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌍 **Crime Distribution**", 
        "🎯 **Area Analysis**", 
        "🌊 **Crime Flow**", 
        "🗺️ **Risk Zones**",
        "📈 **Statistical Maps**",
        "🔥 **Top Crime Types**"
    ])
    
    with tab1:
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters")
        else:
            # Main distribution map
            fig = px.scatter_mapbox(
                filtered_df.sample(min(1000, len(filtered_df))),
                lat="Latitude",
                lon="Longitude",
                color="Primary Type",
                hover_data=["Primary Type", "Community Area"],
                zoom=10,
                height=650,
                title=f"Crime Distribution Map ({len(filtered_df):,} incidents)",
                mapbox_style="open-street-map"
            )
            fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
            
            
    
    with tab2:
        
        if len(filtered_df) == 0:
            st.warning("No data available")
        else:
            # Bubble map by community area

            area_data = filtered_df.groupby('Community Area').agg({
                'Latitude': 'mean',
                'Longitude': 'mean',
                'Primary Type': 'count'
            }).reset_index()
            area_data.columns = ['Area', 'Lat', 'Lon', 'Count']
            
            fig = px.scatter_mapbox(
                area_data,
                lat="Lat",
                lon="Lon",
                size="Count",
                color="Count",
                hover_name="Area",
                size_max=30,
                zoom=10,
                height=650,
                title="Crime count by Area",
                mapbox_style="open-street-map"
            )
            fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        
    with tab3:

        if len(filtered_df) == 0:
            st.warning("No data available")
        else:
            # Hourly flow analysis
            hourly_data = filtered_df.groupby(['Hour', 'Community Area']).size().reset_index(name='Count')
        
        
            fig = px.scatter_mapbox(
                filtered_df,
                lat="Latitude", lon="Longitude",
                color="Hour",
                size_max=10,
                zoom=10, 
                height=400,
                title="Crime Distribution by Hour",
                mapbox_style="open-street-map",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        
            st.subheader("Day vs Night Crime Distribution")
            
            # Separate day and night crimes
            day_crimes = filtered_df[filtered_df['Hour'].between(6, 18)]
            night_crimes = filtered_df[filtered_df['Hour'].between(18, 24)]
            col2, col3 = st.columns(2)
            with col2:
                morning_crimes = filtered_df[filtered_df['Hour'].between(6, 12)]
                evening_crimes = filtered_df[filtered_df['Hour'].between(18, 24)]
                
                if len(morning_crimes) > 10:
                    morning_hotspots = morning_crimes.groupby('Community Area').agg({
                        'Latitude': 'mean',
                        'Longitude': 'mean',
                        'Primary Type': 'count'
                    }).reset_index()
                    morning_hotspots.columns = ['Area', 'Lat', 'Lon', 'Count']
                    
                    fig_morning = px.scatter_mapbox(
                        morning_hotspots,
                        lat="Lat", lon="Lon",
                        size="Count",
                        color="Count",
                        hover_name="Area",
                        size_max=25,
                        zoom=10,
                        height=350,
                        title=f"Morning Crime Hotspots (6AM-12PM)",
                        mapbox_style="open-street-map",
                        color_continuous_scale="Oranges"
                    )
                    fig_morning.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                    st.plotly_chart(fig_morning, use_container_width=True)
            
            with col3:
                if len(evening_crimes) > 10:
                    evening_hotspots = evening_crimes.groupby('Community Area').agg({
                        'Latitude': 'mean',
                        'Longitude': 'mean',
                        'Primary Type': 'count'
                    }).reset_index()
                    evening_hotspots.columns = ['Area', 'Lat', 'Lon', 'Count']
                    
                    fig_evening = px.scatter_mapbox(
                        evening_hotspots,
                        lat="Lat", lon="Lon",
                        size="Count",
                        color="Count",
                        hover_name="Area",
                        size_max=25,
                        zoom=10,
                        height=350,
                        title=f"Evening Crime Hotspots (18-24)",
                        mapbox_style="open-street-map",
                        color_continuous_scale="Purples"
                    )
                    fig_evening.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                    st.plotly_chart(fig_evening, use_container_width=True)
        with tab4:
            st.markdown('<p class="tab-header">Area Risk Zones</p>', unsafe_allow_html=True)
            
            if len(filtered_df) == 0:
                st.warning("No data available")
            else:
                # Risk classification
                area_risk = filtered_df.groupby('Community Area').size().reset_index(name='Crime_Count')
                area_coords = filtered_df.groupby('Community Area').agg({
                    'Latitude': 'mean', 'Longitude': 'mean'
                }).reset_index()
                
                risk_data = area_risk.merge(area_coords, on='Community Area')
                risk_data['Risk_Level'] = pd.cut(risk_data['Crime_Count'], 
                                            bins=3, labels=['Low', 'Medium', 'High'])
                
                fig = px.scatter_mapbox(
                    risk_data,
                    lat="Latitude", lon="Longitude",
                    color="Risk_Level",
                    size="Crime_Count",
                    hover_name="Community Area",
                    size_max=25, zoom=10, height=650,
                    title="Community Risk Zones",
                    mapbox_style="open-street-map"
                )
                fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown('<p class="tab-header">Statistical Crime Maps</p>', unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data available")
        else:
            # Statistical analysis maps
            col1, col2 = st.columns(2)
            
            with col1:
                # Arrest rate by area
                arrest_stats = filtered_df.groupby('Community Area').agg({
                    'Arrest': ['count', 'sum'],
                    'Latitude': 'mean',
                    'Longitude': 'mean'
                }).reset_index()
                arrest_stats.columns = ['Area', 'Total', 'Arrests', 'Lat', 'Lon']
                arrest_stats['Arrest_Rate'] = (arrest_stats['Arrests'] / arrest_stats['Total'] * 100).round(1)
                
                fig = px.scatter_mapbox(
                    arrest_stats[arrest_stats['Total'] >= 5],  # Areas with at least 5 crimes
                    lat="Lat", lon="Lon",
                    color="Arrest_Rate",
                    size="Total",
                    hover_name="Area",
                    hover_data=['Arrest_Rate', 'Total'],
                    color_continuous_scale="RdYlGn",
                    size_max=30, zoom=10, height=400,
                    title="Arrest Rate by Area (%)",
                    mapbox_style="open-street-map"
                )
                fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Domestic violence concentration
                domestic_stats = filtered_df.groupby('Community Area').agg({
                    'Domestic': ['count', 'sum'],
                    'Latitude': 'mean',
                    'Longitude': 'mean'
                }).reset_index()
                domestic_stats.columns = ['Area', 'Total', 'Domestic_Cases', 'Lat', 'Lon']
                domestic_stats['Domestic_Rate'] = (domestic_stats['Domestic_Cases'] / domestic_stats['Total'] * 100).round(1)
                
                fig = px.scatter_mapbox(
                    domestic_stats[domestic_stats['Total'] >= 5],
                    lat="Lat", lon="Lon",
                    color="Domestic_Rate",
                    size="Domestic_Cases",
                    hover_name="Area",
                    hover_data=['Domestic_Rate', 'Domestic_Cases'],
                    color_continuous_scale="Oranges",
                    size_max=25, zoom=10, height=400,
                    title="Domestic Violence Rate by Area (%)",
                    mapbox_style="open-street-map"
                )
                fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            
            
    with tab6:
    
        
        if len(filtered_df) == 0:
            st.warning("No data available")
        else:
            
            crime_stats = filtered_df['Primary Type'].value_counts().head(10).reset_index()
            crime_stats.columns = ['Crime Type', 'Count']
            crime_stats['Percentage'] = (crime_stats['Count'] / len(filtered_df) * 100).round(2)

            st.subheader("Top Crime Types Statistics")
            st.dataframe(crime_stats, use_container_width=True, height=300)

            numeric_cols = filtered_df.select_dtypes(include=['int64','float64']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if "id" not in col.lower()]
            
            option = numeric_cols[0] if numeric_cols else 'Year'



           
            top_crimes = filtered_df['Primary Type'].value_counts().head(3).index.tolist()

            if len(top_crimes) >= 2:

                for crime_type in top_crimes:
                    crime_data = filtered_df[filtered_df['Primary Type'] == crime_type]

                    if len(crime_data) >= 10:
                        sampled_data = crime_data.sample(min(1000, len(crime_data)))

                        st.markdown(f"### {crime_type} ({len(crime_data)} cases)")

                        # Bubble Map with Location Description
                        fig_bubble = px.scatter_mapbox(
                            sampled_data,
                            lat="Latitude",
                            lon="Longitude",
                            size=option,
                            color="Location Description",
                            size_max=15,
                            zoom=9,
                            height=500,
                            mapbox_style="open-street-map",
                            title=f"{crime_type} - Crime Locations",
                            hover_data=[option, "Location Description"]
                        )
                        st.plotly_chart(fig_bubble, use_container_width=True)





                        top_location = crime_data['Location Description'].value_counts().head(1)
                        top_location_name = top_location.index[0] if len(top_location) > 0 else " NuN"
                        top_location_count = top_location.iloc[0] if len(top_location) > 0 else 0
                        
                        col1, col2, col3,  = st.columns(3)
                        with col1:
                            st.metric("Total Cases", len(crime_data))
                        with col2:
                            st.metric("Top Location", top_location_name)
                        with col3:
                            st.metric("Cases at Top Location", top_location_count)
                      
            # Density Map
            if len(top_crimes) >= 2:
                
                # Create comparison data
                comparison_data = []
                for crime_type in top_crimes[:3]:
                    crime_data = filtered_df[filtered_df['Primary Type'] == crime_type]
                    if len(crime_data) > 0:
                        comparison_data.append({
                            'Crime Type': crime_type,
                            'Count': len(crime_data),
                            f'Avg {option}': crime_data[option].mean(),
                            f'Max {option}': crime_data[option].max(),
                            'Latitude Range': crime_data['Latitude'].max() - crime_data['Latitude'].min(),
                            'Longitude Range': crime_data['Longitude'].max() - crime_data['Longitude'].min()
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
               
                
                # Create animated monthly heatmap
                monthly_data = filtered_df.groupby(['Month_Name', 'Community Area']).agg({
                    'Latitude': 'mean',
                    'Longitude': 'mean',
                    'Primary Type': 'count'
                }).reset_index()
                monthly_data.columns = ['Month', 'Area', 'Lat', 'Lon', 'Count']
                
                if len(monthly_data) > 0:
                    fig = px.scatter_mapbox(
                        monthly_data,
                        lat="Lat", lon="Lon",
                        size="Count",
                        color="Count",
                        animation_frame="Month",
                        hover_name="Area",
                        size_max=20,
                        zoom=10, height=500,
                        title="Monthly Crime Density Animation",
                        mapbox_style="open-street-map"
                    )
                    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()