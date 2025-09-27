import streamlit as st
import pandas as pd
import geopandas as gpd
from scipy.stats import gaussian_kde
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, time

# Page configuration
st.set_page_config(
    page_title="Crime Maps Dashboard 2022",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#  CSS for better styling
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

# Load  data
@st.cache_data
def load_data():
    """Load and preprocess the crime data"""
    try:
        #  file path
        df = pd.read_csv("../data/Crimes_raw.csv")
        
        # column names - 
        column_mapping = {
            'Date': 'Date',
            'Primary Type': 'Primary Type',
            'Description': 'Description',
            'Location Description': 'Location Description',
            'Arrest': 'Arrest',
            'Domestic': 'Domestic',
            'Community Area': 'Community Area',
            'Ward': 'Ward',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude',
            'X Coordinate': 'X Coordinate',
            'Y Coordinate': 'Y Coordinate',
            'Location': 'Location',
            'Case Number': 'Case Number',
            'IUCR': 'IUCR',
            'Beat': 'Beat',
            'District': 'District',
            'FBI Code': 'FBI Code',
            'Year': 'Year',
            'Updated On': 'Updated On',
            
        }
        
        # find columns with similar names
        for expected_col, standard_col in column_mapping.items():
            if expected_col not in df.columns:
                # Look for similar column names
                similar_cols = [col for col in df.columns if expected_col.lower().replace(' ', '') in col.lower().replace(' ', '')]
                if similar_cols:
                    df = df.rename(columns={similar_cols[0]: standard_col})
        
        #  date column to datetime
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            except:
                # Try different date formats
                for date_format in ['%m/%d/%Y %I:%M:%S %p', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y']:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
                        break
                    except:
                        continue
        
        #  time-based columns 
        if 'Date' in df.columns and not df['Date'].isna().all():
            df['Hour'] = df['Date'].dt.hour
            df['Month'] = df['Date'].dt.month.astype(int)
            df['Day_of_Week'] = df['Date'].dt.day_name()
            df['Month_Name'] = df['Date'].dt.month_name()
            df['Week'] = df['Date'].dt.isocalendar().week
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Fall'

            df['Season'] = df['Month'].apply(get_season)
        else:
            df['Hour'] = 12  # Default hour
            df['Month'] = 6  # Default month
            df['Day_of_Week'] = 'Monday'
            df['Month_Name'] = 'June'
            df['Week'] = 1
            df['Quarter'] = 2
        
        # Clean 
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df = df.dropna(subset=['Latitude', 'Longitude'])
            df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
            df = df[(df['Latitude'].between(41.6, 42.1)) & (df['Longitude'].between(-87.9,-87.5))]
        else:
            st.error("Latitude and Longitude columns are required but not found in the data")
            return None
        
        # Handle boolean columns
        for bool_col in ['Arrest', 'Domestic']:
            if bool_col in df.columns:
                if df[bool_col].dtype == 'object':
                    df[bool_col] = df[bool_col].map({'true': True, 'True': True, 'TRUE': True, 
                                                   'false': False, 'False': False, 'FALSE': False})
                df[bool_col] = df[bool_col].fillna(False).astype(bool)
            else:
                df[bool_col] = False 
        #  time periods for better analysis
        if 'Hour' in df.columns:
            try:
                df['Time_Period'] = pd.cut(df['Hour'], 
                                          bins=[0, 6, 12, 18, 23], 
                                          labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-23)'],
                                          include_lowest=True)
            except:
                df['Time_Period'] = 'Morning (6-12)'  
        
        #  missing Primary Type
        if 'Primary Type' not in df.columns:
            df['Primary Type'] = 'UNKNOWN'
        else:
            df['Primary Type'] = df['Primary Type'].fillna('UNKNOWN')
        
        #  missing Community Area
        if 'Community Area' not in df.columns:
            df['Community Area'] = 'Unknown Area'
        else:
            df['Community Area'] = df['Community Area'].fillna('Unknown Area')
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        return df
        
    except FileNotFoundError:
        st.error("File not found. Please check the file path: Crimes_-_2022_20250925.csv")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def safe_sample(df, n):
    """Safely sample dataframe"""
    if df is None or len(df) == 0:
        return df
    return df.sample(min(n, len(df))) if len(df) > n else df

def create_safe_map(data, lat_col, lon_col, **kwargs):
    """Create map with error handling"""
    try:
        if data is None or len(data) == 0:
            return None
        
        # Ensure we have valid coordinates
        clean_data = data.dropna(subset=[lat_col, lon_col])
        clean_data = clean_data[(clean_data[lat_col] != 0) & (clean_data[lon_col] != 0)]
        
        if len(clean_data) == 0:
            return None
            
        # Limit data size for performance
        if len(clean_data) > 5000:
            clean_data = clean_data.sample(5000)
        
        # Create map based on map type
        map_type = kwargs.get('map_type', 'scatter')
        
        if map_type == 'density':
            fig = px.density_mapbox(clean_data, lat=lat_col, lon=lon_col, **{k: v for k, v in kwargs.items() if k != 'map_type'})
        else:
            fig = px.scatter_mapbox(clean_data, lat=lat_col, lon=lon_col, **{k: v for k, v in kwargs.items() if k != 'map_type'})
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":50,"l":0,"b":0},
            height=kwargs.get('height', 500)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🗺️ Crime Maps Analytics Dashboard 2022</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your file path and format.")
        st.info("Expected columns: Date, Primary Type, Latitude, Longitude, Arrest, Domestic, Community Area")
        return
    
    # Debug information
    with st.expander("📊 Data Information", expanded=False):
        st.write(f"**Total rows:** {len(df):,}")
        st.write(f"**Available columns:** {', '.join(df.columns.tolist())}")
        st.write(f"**Date range:** {df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns and not df['Date'].isna().all() else "Date information not available")
        st.write(f"**Coordinate coverage:** {len(df.dropna(subset=['Latitude', 'Longitude'])):,} records with valid coordinates")
        st.write(f"**Crime types:** {df['Primary Type'].nunique()} unique types")
    
    # Filter Section
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    st.markdown("### 🔍 **Map Filters**")
    
    # First row of filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        crime_types = ['All'] + sorted([str(x) for x in df['Primary Type'].unique() if pd.notna(x)])
        selected_crime_type = st.selectbox("🔴 Crime Type", crime_types, key="crime_type")
    
    with col2:
        if 'Time_Period' in df.columns:
            time_periods = ['All'] + sorted([str(x) for x in df['Time_Period'].dropna().unique()])
            selected_time_period = st.selectbox("⏰ Time Period", time_periods, key="time_period")
        else:
            selected_time_period = 'All'
            st.selectbox("⏰ Time Period", ['All'], key="time_period", disabled=True)
    
    with col3:
        arrest_filter = st.selectbox("🚔 Arrest Status", ['All', 'Arrested', 'Not Arrested'], key="arrest")
    
    with col4:
        domestic_filter = st.selectbox("🏠 Domestic Violence", ['All', 'Domestic', 'Non-Domestic'], key="domestic")
    
    # Second row of filters
    col5, col6 = st.columns(2)
    
    with col5:
        if 'Date' in df.columns and not df['Date'].isna().all():
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            date_range = st.date_input("📅 Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="date_range")
        else:
            st.text_input("📅 Date Range", "Date data not available", disabled=True, key="date_range")
            date_range = None
    
    with col6:
        if 'Month_Name' in df.columns:
            months = ['All'] + sorted([str(x) for x in df['Month_Name'].unique() if pd.notna(x)], 
                                    key=lambda x: datetime.strptime(x, '%B').month if x != 'All' else 0)
            selected_month = st.selectbox("📆 Month", months, key="month")
        else:
            selected_month = 'All'
            st.selectbox("📆 Month", ['All'], key="month", disabled=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filtered_df = df.copy()
    
    try:
        if selected_crime_type != 'All':
            filtered_df = filtered_df[filtered_df['Primary Type'] == selected_crime_type]
        
        if selected_time_period != 'All' and 'Time_Period' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Time_Period'] == selected_time_period]
        
        if arrest_filter == 'Arrested':
            filtered_df = filtered_df[filtered_df['Arrest'] == True]
        elif arrest_filter == 'Not Arrested':
            filtered_df = filtered_df[filtered_df['Arrest'] == False]
        
        if domestic_filter == 'Domestic':
            filtered_df = filtered_df[filtered_df['Domestic'] == True]
        elif domestic_filter == 'Non-Domestic':
            filtered_df = filtered_df[filtered_df['Domestic'] == False]
        
        if date_range and len(date_range) == 2 and 'Date' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['Date'].dt.date >= date_range[0]) & 
                (filtered_df['Date'].dt.date <= date_range[1])
            ]
        
        if selected_month != 'All' and 'Month_Name' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Month_Name'] == selected_month]
    
    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")
        filtered_df = df.copy()
    
    # Key metrics cards
    col1, col2, col3, col4, col5 ,col6 = st.columns(6)
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{len(filtered_df):,}</h3><p>Total Crimes</p></div>', unsafe_allow_html=True)
    
    with col2:
        arrest_rate = (filtered_df['Arrest'] == True).mean() * 100 if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-card"><h3>{arrest_rate:.1f}%</h3><p>Arrest Rate</p></div>', unsafe_allow_html=True)
    
    with col3:
        domestic_rate = (filtered_df['Domestic'] == True).mean() * 100 if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-card"><h3>{domestic_rate:.1f}%</h3><p>Family-Related Crimes</p></div>', unsafe_allow_html=True)
    
    with col4:
        unique_areas = filtered_df['Community Area'].nunique() if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-card"><h3>{unique_areas}</h3><p>Community Areas</p></div>', unsafe_allow_html=True)
    
    with col5:
        peak_hour = int(filtered_df['Hour'].mode().iloc[0]) if len(filtered_df) > 0 and 'Hour' in filtered_df.columns else 12
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
    tab1, tab2, tab3, tab4,tab6 = st.tabs([
        "🌍 **Crime Distribution**", 
        "🔥 **Density Heatmap**", 
        "⏰ **Temporal Patterns**", 
        "📊 **Crime Categories**",
        "🎯 **High-Risk  Zones**"
    ])

    #---------------------------------------------------------------------------------
    with tab1:
        st.markdown('<p class="tab-header">Geographic Distribution of Crimes</p>', unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters")
        else:
            # Main distribution map
            fig = create_safe_map(
                filtered_df, 
                'Latitude', 
                'Longitude',
                color='Primary Type',
                title=f"Crime Distribution Map ({len(filtered_df):,} incidents)",
                height=650,
                hover_data=['Primary Type', 'Community Area']
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to create map with current data")
            
            # Secondary analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Crime Areas")
                try:
                    area_crimes = filtered_df.groupby('Community Area').size().reset_index(name='Crime_Count')
                    area_crimes = area_crimes.sort_values('Crime_Count', ascending=False).head(10)
                    
                    if len(area_crimes) > 0:
                        st.dataframe(area_crimes, hide_index=True, use_container_width=True)
                    else:
                        st.info("No area data available")
                except Exception as e:
                    st.error(f"Error analyzing areas: {str(e)}")
            
            with col2:
                st.subheader("Crime Type Distribution")
                district_data = df.groupby('District').agg({
                    'Latitude': 'mean',
                    'Longitude': 'mean',
                    'Case Number': 'count'
                }).reset_index()
                district_data.columns = ['District', 'Latitude', 'Longitude', 'Crime_Count']

                # Create bubble map
                fig = px.scatter_mapbox(
                    district_data,
                    lat='Latitude',
                    lon='Longitude',
                    size='Crime_Count',
                    color='Crime_Count',
                    hover_name='District',
                    hover_data=['Crime_Count'],
                    color_continuous_scale='Reds',
                    size_max=50,
                    zoom=9,
                    title='Crime Density Bubble Map by District'
                )

                # Update map style
                fig.update_layout(mapbox_style="open-street-map")

               

                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)



    #---------------------------------------------------------------------------------
    with tab2:
        st.markdown('<p class="tab-header">Crime Cluster Analysis</p>', unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters")
        elif len(filtered_df) < 50:
            st.warning("Insufficient data for clustering analysis. Need at least 50 data points.")
        else:
            try:
                # Prepare coordinates 
                valid_data = filtered_df.dropna(subset=['Latitude', 'Longitude'])
                valid_data = valid_data[(valid_data['Latitude'] != 0) & (valid_data['Longitude'] != 0)]
                
                if len(valid_data) < 50:
                    st.warning(f"Only {len(valid_data)} valid coordinate points available. Need at least 50 for clustering.")
                else:
                    # Use all data - no sampling
                    sample_data = valid_data.copy()
                    
                    coords = sample_data[['Latitude', 'Longitude']].values

                    # Perform K-means clustering
                    n_clusters = min(8, len(sample_data) // 10)  # Adjust clusters based on data size
                    
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(coords)

                    # Add cluster labels to data
                    sample_data_copy = sample_data.copy()
                    sample_data_copy['Cluster'] = clusters

                    # Create cluster map
                    fig = go.Figure()

                    # Color palette for clusters
                    colors = px.colors.qualitative.Set3[:n_clusters]

                    # Add scatter points colored by cluster
                    for i in range(n_clusters):
                        cluster_data = sample_data_copy[sample_data_copy['Cluster'] == i]
                        if len(cluster_data) > 0:
                            fig.add_trace(go.Scattermapbox(
                                lat=cluster_data['Latitude'],
                                lon=cluster_data['Longitude'],
                                mode='markers',
                                marker=dict(
                                    size=6,
                                    color=colors[i] if i < len(colors) else f'rgb({50+i*30}, {100+i*20}, {150+i*25})',
                                    opacity=0.7
                                ),
                                name=f'Cluster {i+1}',
                                text=[f"Crime: {crime}<br>Community: {area}" 
                                    for crime, area in zip(cluster_data['Primary Type'], 
                                                        cluster_data.get('Community Area', 'Unknown'))],
                                hovertemplate='<b>%{fullData.name}</b><br>' +
                                            '%{text}<br>' +
                                            'Lat: %{lat:.4f}<br>' +
                                            'Lon: %{lon:.4f}<br>' +
                                            '<extra></extra>'
                            ))

                    # Add cluster centers
                    centers = kmeans.cluster_centers_
                    fig.add_trace(go.Scattermapbox(
                        lat=centers[:, 0],
                        lon=centers[:, 1],
                        mode='markers',
                        marker=dict(
                            size=20,
                            color='red',
                            symbol='star',
                            opacity=1.0
                        ),
                        name='Cluster Centers',
                        hovertemplate='<b>Cluster Center</b><br>' +
                                    'Lat: %{lat:.4f}<br>' +
                                    'Lon: %{lon:.4f}<br>' +
                                    '<extra></extra>'
                    ))

                    # Calculate map center and zoom
                    center_lat = sample_data['Latitude'].mean()
                    center_lon = sample_data['Longitude'].mean()
                    
                    # Determine zoom level based on data spread
                    lat_range = sample_data['Latitude'].max() - sample_data['Latitude'].min()
                    lon_range = sample_data['Longitude'].max() - sample_data['Longitude'].min()
                    zoom_level = max(8, min(12, 10 - max(lat_range, lon_range) * 10))

                    fig.update_layout(
                        mapbox_style="open-street-map",
                        mapbox=dict(
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=zoom_level
                        ),
                        title=f'Crime Cluster Analysis Map ({n_clusters} Clusters)',
                        title_font_size=20,
                        title_x=0.5,
                        font_size=12,
                        height=650,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster Analysis Summary
                    st.subheader("📊 Cluster Analysis Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Cluster statistics
                        cluster_stats = []
                        for i in range(n_clusters):
                            cluster_data = sample_data_copy[sample_data_copy['Cluster'] == i]
                            if len(cluster_data) > 0:
                                stats = {
                                    'Cluster': f'Cluster {i+1}',
                                    'Size': len(cluster_data),
                                    'Percentage': f"{(len(cluster_data)/len(sample_data_copy)*100):.1f}%",
                                    'Top Crime': cluster_data['Primary Type'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                                    'Avg Lat': f"{cluster_data['Latitude'].mean():.4f}",
                                    'Avg Lon': f"{cluster_data['Longitude'].mean():.4f}"
                                }
                                cluster_stats.append(stats)
                        
                        cluster_df = pd.DataFrame(cluster_stats)
                        st.dataframe(cluster_df, hide_index=True, use_container_width=True)
                    
                    with col2:
                        # Crime type distribution by cluster
                        st.write("**Crime Distribution by Cluster:**")
                        
                        # Create a pivot table for crime types by cluster
                        crime_cluster_pivot = sample_data_copy.groupby(['Cluster', 'Primary Type']).size().unstack(fill_value=0)
                        
                        if not crime_cluster_pivot.empty:
                            # Show top 5 crime types
                            top_crimes = sample_data_copy['Primary Type'].value_counts().head(5).index
                            available_crimes = [crime for crime in top_crimes if crime in crime_cluster_pivot.columns]
                            
                            if available_crimes:
                                cluster_crime_subset = crime_cluster_pivot[available_crimes]
                                
                                fig_bar = px.bar(
                                    cluster_crime_subset.reset_index(),
                                    x='Cluster',
                                    y=available_crimes,
                                    title="Top Crime Types by Cluster",
                                    labels={'value': 'Number of Crimes', 'variable': 'Crime Type'}
                                )
                                fig_bar.update_layout(height=300)
                                st.plotly_chart(fig_bar, use_container_width=True)
                            else:
                                st.info("Crime type data not available for visualization")
                    
                    # Additional insights
                    st.subheader("🔍 Key Insights")
                    
                    insights_col1, insights_col2, insights_col3 = st.columns(3)
                    
                    with insights_col1:
                        largest_cluster = sample_data_copy['Cluster'].value_counts().index[0]
                        largest_size = sample_data_copy['Cluster'].value_counts().iloc[0]
                        st.metric("Largest Cluster", f"Cluster {largest_cluster + 1}", f"{largest_size} crimes")
                    
                    with insights_col2:
                        if 'Arrest' in sample_data_copy.columns:
                            arrest_by_cluster = sample_data_copy.groupby('Cluster')['Arrest'].mean()
                            best_arrest_cluster = arrest_by_cluster.idxmax()
                            best_arrest_rate = arrest_by_cluster.max() * 100
                            st.metric("Highest Arrest Rate", f"Cluster {best_arrest_cluster + 1}", f"{best_arrest_rate:.1f}%")
                        else:
                            st.metric("Arrest Data", "Not Available", "")
                    
                    with insights_col3:
                        # Calculate cluster density (crimes per unit area approximation)
                        cluster_densities = []
                        for i in range(n_clusters):
                            cluster_data = sample_data_copy[sample_data_copy['Cluster'] == i]
                            if len(cluster_data) > 1:
                                lat_range = cluster_data['Latitude'].max() - cluster_data['Latitude'].min()
                                lon_range = cluster_data['Longitude'].max() - cluster_data['Longitude'].min()
                                area_approx = max(lat_range * lon_range, 0.0001)  # Avoid division by zero
                                density = len(cluster_data) / area_approx
                                cluster_densities.append(density)
                        
                        if cluster_densities:
                            densest_cluster = np.argmax(cluster_densities)
                            st.metric("Densest Cluster", f"Cluster {densest_cluster + 1}", "Highest crime density")
                        else:
                            st.metric("Density Analysis", "Not Available", "")
                            
            except ImportError:
                st.error("sklearn library is required for clustering analysis. Please install it: pip install scikit-learn")
            except Exception as e:
                st.error(f"Error in clustering analysis: {str(e)}")
                st.info("Try reducing the data size or check data quality.")

    #---------------------------------------------------------------------------------
    with tab3:
        st.markdown('<p class="tab-header">Temporal Crime Pattern Maps</p>', unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters")
        else:
            # Time period distribution
            if 'Time_Period' in filtered_df.columns:
                fig = create_safe_map(
                    filtered_df,
                    'Latitude',
                    'Longitude',
                    color='Time_Period',
                    title="Crime Distribution by Time of Day",
                    height=500
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Hourly analysis
            col1, col2 = st.columns(2)
            
            with col1:
                if len(filtered_df) == 0:
                    st.warning("No data matches the selected filters")
                else:
                    fig = px.scatter_mapbox(
                        filtered_df,
                        lat="Latitude",
                        lon="Longitude",
                        color="Month_Name",  
                        hover_data=["Primary Type", "Description", "Date"],
                        animation_frame="Month",   
                        zoom=9,
                        height=600,
                        title="Crime Distribution by Month (2022)"
                    )

                    # map setting
                    fig.update_layout(mapbox_style="open-street-map")
                    fig.update_layout(title_x=0.5, title_font_size=20)

                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
               # Sample data for performance

                # Create hexagonal binning map
                fig = ff.create_hexbin_mapbox(
                    data_frame=df,
                    lat="Latitude",
                    lon="Longitude",
                    nx_hexagon=20,
                    opacity=0.6,
                    labels={"color": "Crime Count"},
                    color_continuous_scale="Reds",
                    agg_func=np.sum
                )

                # Update map style
                fig.update_layout(mapbox_style="open-street-map")

                # Customize layout
                fig.update_layout(
                    title='Hexagonal Binning Crime Density Map',
                    title_font_size=20,
                    title_x=0.5,
                    font_size=12,
                    width=900,
                    height=600,
                    mapbox=dict(
                        center=dict(lat=filtered_df['Latitude'].mean(), lon=filtered_df['Longitude'].mean()),
                        zoom=9
                    )
                )
                st.plotly_chart(fig, use_container_width=True)  

    #---------------------------------------------------------------------------------
    with tab4:
        st.markdown('<p class="tab-header">Crime Category Distribution Maps</p>', unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters")
        else:
            # Main crime type map
            top_crimes = filtered_df['Primary Type'].value_counts().head(8).index.tolist()
            if len(top_crimes) > 0:
                top_crime_data = filtered_df[filtered_df['Primary Type'].isin(top_crimes)]
                
                fig = create_safe_map(
                    top_crime_data,
                    'Latitude',
                    'Longitude',
                    color='Primary Type',
                    title="Geographic Distribution of Top Crime Types",
                    height=600
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # Divide layout into columns
            col1, col2 = st.columns(2)

            with col1:

                # Create grid for contour
                lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
                lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()

                # Create meshgrid
                lat_grid = np.linspace(lat_min, lat_max, 50)
                lon_grid = np.linspace(lon_min, lon_max, 50)
                lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)

                # Calculate density using KDE
                positions = np.vstack([df['Longitude'], df['Latitude']])
                kernel = gaussian_kde(positions)
                density = kernel(np.vstack([lon_mesh.ravel(), lat_mesh.ravel()]))
                density = density.reshape(lon_mesh.shape)

                # Create contour map
                fig = go.Figure(data=go.Contour(
                    z=density,
                    x=lon_grid,
                    y=lat_grid,
                    colorscale='Reds',
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=12, color='white')
                    )
                ))

                # Customize layout
                fig.update_layout(
                    title='Crime Density Contour Map',
                    title_font_size=20,
                    title_x=0.5,
                    font_size=12,
                    width=900,
                    height=600,
                    xaxis_title='Longitude',
                    yaxis_title='Latitude'
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter_mapbox(
                    df,
                    lat="Latitude",
                    lon="Longitude",
                    color="Season",  
                    hover_data=["Primary Type", "Description", "Date"],
                    zoom=10,
                    height=600,
                    title="Seasonal Crime Distribution"
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)

    #---------------------------------------------------------------------------------
    with tab6:
        st.markdown('<p class="tab-header">Crime High-Risk Areas Analysis</p>', unsafe_allow_html=True)
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters")
        else:
            # Hotspot identification
            try:
                hotspot_data = filtered_df.groupby(['Latitude', 'Longitude']).size().reset_index(name='Crime_Count')
                
                if len(hotspot_data) > 0:
                    # Get top 5% locations
                    threshold = hotspot_data['Crime_Count'].quantile(0.95)
                    hotspots = hotspot_data[hotspot_data['Crime_Count'] >= threshold]
                    
                    if len(hotspots) > 0:
                        fig = create_safe_map(
                            hotspots,
                            'Latitude',
                            'Longitude',
                            size='Crime_Count',
                            color='Crime_Count',
                            color_continuous_scale='Reds',
                            title=f"High-Risk Areas (Top 5% - {len(hotspots)} locations)",
                            height=600
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Hotspot statistics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("High-Risk Areas Statistics")
                            st.metric("Number of High-Risk Areas", len(hotspots))
                            st.metric("Max Crimes at Single Location", hotspots['Crime_Count'].max())
                        
                        with col2:
                            st.subheader("Top 10 High-Risk Areas")
                            top_hotspots = hotspots.sort_values('Crime_Count', ascending=False).head(10)
                            st.dataframe(
                                top_hotspots[['Latitude', 'Longitude', 'Crime_Count']].round(6),
                                hide_index=True,
                                use_container_width=True
                            )
                    else:
                        st.info("No significant High-Risk Areas detected with current filters")
                else:
                    st.info("Insufficient data for High-Risk Areas analysis")
            except Exception as e:
                st.error(f"Error in High-Risk Areas analysis: {str(e)}")

if __name__ == "__main__":
    main()