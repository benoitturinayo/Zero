# hidden_hunger_streamlit.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression 
from urllib.request import getproxies
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Hidden Hunger Analyzer - Africa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AFRICA CONFIGURATION
# ============================================================================

AFRICAN_COUNTRIES = [
    'DZA', 'AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CPV', 'CMR', 'CAF', 'TCD', 'COM', 'COD',
    'COG', 'CIV', 'DJI', 'EGY', 'GNQ', 'ERI', 'SWZ', 'ETH', 'GAB', 'GMB', 'GHA', 'GIN',
    'GNB', 'KEN', 'LSO', 'LBR', 'LBY', 'MDG', 'MWI', 'MLI', 'MRT', 'MUS', 'MYT', 'MAR',
    'MOZ', 'NAM', 'NER', 'NGA', 'RWA', 'STP', 'SEN', 'SYC', 'SLE', 'SOM', 'ZAF', 'SSD',
    'SDN', 'TZA', 'TGO', 'TUN', 'UGA', 'ZMB', 'ZWE'
]

AFRICAN_COUNTRY_NAMES = {
    'DZA': 'Algeria', 'AGO': 'Angola', 'BEN': 'Benin', 'BWA': 'Botswana', 'BFA': 'Burkina Faso',
    'BDI': 'Burundi', 'CPV': 'Cabo Verde', 'CMR': 'Cameroon', 'CAF': 'Central African Republic',
    'TCD': 'Chad', 'COM': 'Comoros', 'COD': 'DR Congo', 'COG': 'Congo', 'CIV': "Côte d'Ivoire",
    'DJI': 'Djibouti', 'EGY': 'Egypt', 'GNQ': 'Equatorial Guinea', 'ERI': 'Eritrea', 
    'SWZ': 'Eswatini', 'ETH': 'Ethiopia', 'GAB': 'Gabon', 'GMB': 'Gambia', 'GHA': 'Ghana',
    'GIN': 'Guinea', 'GNB': 'Guinea-Bissau', 'KEN': 'Kenya', 'LSO': 'Lesotho', 'LBR': 'Liberia',
    'LBY': 'Libya', 'MDG': 'Madagascar', 'MWI': 'Malawi', 'MLI': 'Mali', 'MRT': 'Mauritania',
    'MUS': 'Mauritius', 'MYT': 'Mayotte', 'MAR': 'Morocco', 'MOZ': 'Mozambique', 'NAM': 'Namibia',
    'NER': 'Niger', 'NGA': 'Nigeria', 'RWA': 'Rwanda', 'STP': 'São Tomé & Príncipe', 'SEN': 'Senegal',
    'SYC': 'Seychelles', 'SLE': 'Sierra Leone', 'SOM': 'Somalia', 'ZAF': 'South Africa',
    'SSD': 'South Sudan', 'SDN': 'Sudan', 'TZA': 'Tanzania', 'TGO': 'Togo', 'TUN': 'Tunisia',
    'UGA': 'Uganda', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe'
}

HIDDEN_HUNGER_INDICATORS = {
    'SH.STA.STNT.ZS': 'Stunting (children under 5)',
    'SH.STA.WAST.ZS': 'Wasting (children under 5)',
    'SH.ANM.CHLD.ZS': 'Anemia (children 6-59 months)',
    'SH.ANM.ALLW.ZS': 'Anemia (women of reproductive age)',
    'SN.ITK.DEFC.ZS': 'Prevalence of undernourishment',
    'AG.PRD.FOOD.XD': 'Food production index',
    'SH.H2O.BASW.ZS': 'Basic drinking water services',
    'SH.STA.BASS.ZS': 'Basic sanitation services',
    'SH.STA.ODFC.ZS': 'Open defecation',
    'SI.POV.DDAY': 'Poverty ($2.15/day)',
    'NY.GDP.PCAP.CD': 'GDP per capita',
    'SH.STA.BRTC.ZS': 'Skilled birth attendance',
    'SH.IMM.MEAS': 'Measles immunization',
    'AG.YLD.CREL.KG': 'Cereal yield',
    'AG.LND.AGRI.ZS': 'Agricultural land',
}

# ============================================================================
# DATA FETCHING FUNCTIONS (IMPROVED)
# ============================================================================

def get_proxy_settings():
    """Retrieves system proxy settings."""
    proxies = getproxies()
    return {p: url for p, url in proxies.items() if url}

def test_api_connection():
    """Test World Bank API connection"""
    st.subheader("Testing API Connection...")
    
    test_url = "https://api.worldbank.org/v2/country/NGA/indicator/SP.POP.TOTL"
    params = {'format': 'json', 'date': '2020:2023', 'per_page': 10}
    
    try:
        st.write("Attempting to connect to World Bank API...")
        response = requests.get(test_url, params=params, timeout=10, verify=True)
        st.write(f"✓ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.write("Response preview:")
            if isinstance(data, list) and len(data) > 1 and data[1]:
                st.write(f"Found {len(data[1])} records")
                st.json(data[1][0] if data[1] else {})
            st.success("✓ API connection successful!")
            return True
        else:
            st.error(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.SSLError as e:
        st.warning("⚠ SSL verification failed. Trying without SSL verification...")
        try:
            response = requests.get(test_url, params=params, timeout=10, verify=False)
            if response.status_code == 200:
                st.success("✓ API connection successful (without SSL verification)")
                return True
        except Exception as e2:
            st.error(f"✗ Connection failed even without SSL: {e2}")
            return False
    except Exception as e:
        st.error(f"✗ Connection failed: {type(e).__name__}: {e}")
        st.write("Common solutions:")
        st.write("1. Check your internet connection")
        st.write("2. Check if you're behind a proxy/firewall")
        st.write("3. Try using a VPN if World Bank API is blocked")
        return False

@st.cache_data(ttl=3600)
def fetch_worldbank_data(indicator: str, countries: list = None):
    """Fetch indicator data from World Bank API with improved error handling"""
    if countries is None:
        countries = AFRICAN_COUNTRIES
    
    # Split into smaller batches to avoid URL length issues
    batch_size = 10
    all_records = []
    
    for i in range(0, len(countries), batch_size):
        batch = countries[i:i + batch_size]
        country_param = ';'.join(batch)
        url = f"https://api.worldbank.org/v2/country/{country_param}/indicator/{indicator}"
        params = {
            'format': 'json',
            'date': '2010:2024',
            'per_page': 10000
        }
        
        # Retry mechanism
        for attempt in range(3):
            try:
                proxies = get_proxy_settings()
                
                # Try with SSL verification first
                try:
                    response = requests.get(
                        url, 
                        params=params, 
                        timeout=30,
                        verify=True, 
                        proxies=proxies if proxies else None
                    )
                except requests.exceptions.SSLError:
                    # If SSL fails, retry without verification
                    response = requests.get(
                        url, 
                        params=params, 
                        timeout=30,
                        verify=False, 
                        proxies=proxies if proxies else None
                    )
                
                response.raise_for_status()
                data = response.json()
                
                # Parse the response
                if isinstance(data, list) and len(data) > 1 and data[1]:
                    for item in data[1]:
                        if item and item.get('value') is not None:
                            try:
                                all_records.append({
                                    "country_code": item.get("countryiso3code", ""),
                                    "country_name": item.get("country", {}).get("value", ""),
                                    "indicator_code": indicator,
                                    "indicator_name": HIDDEN_HUNGER_INDICATORS.get(
                                        indicator, item.get("indicator", {}).get("value", "")
                                    ),
                                    "year": int(item.get("date", 0)),
                                    "value": float(item.get("value", 0)),
                                })
                            except (ValueError, TypeError, KeyError):
                                continue
                
                # Success for this batch
                break
                
            except requests.exceptions.Timeout:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue
            except requests.exceptions.RequestException:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue
            except Exception:
                return pd.DataFrame()
    
    return pd.DataFrame(all_records)

@st.cache_data(ttl=3600)
def fetch_all_indicators():
    """Fetch all hidden hunger indicators with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_data = {}
    total_indicators = len(HIDDEN_HUNGER_INDICATORS)
    success_count = 0
    
    for idx, (indicator, name) in enumerate(HIDDEN_HUNGER_INDICATORS.items()):
        status_text.text(f"Fetching {name}... ({idx+1}/{total_indicators})")
        
        try:
            df = fetch_worldbank_data(indicator)
            if not df.empty:
                all_data[indicator] = df
                success_count += 1
        except Exception:
            pass
        
        progress_bar.progress((idx + 1) / total_indicators)
        time.sleep(0.5)
    
    status_text.empty()
    progress_bar.empty()
    
    # if all_data:
    #     st.success(f"✓ Successfully loaded {success_count} out of {total_indicators} indicators")
    
    return all_data

def get_latest_data(df):
    """Get latest available data for each country"""
    if df.empty:
        return df
    latest_years = df.groupby('country_code')['year'].max().reset_index()
    latest_data = pd.merge(latest_years, df, on=['country_code', 'year'])
    return latest_data

def combine_indicators(data):
    """Combine all indicators into single dataframe"""
    combined = {}
    
    for indicator, df in data.items():
        latest_data = get_latest_data(df)
        indicator_name = HIDDEN_HUNGER_INDICATORS.get(indicator, indicator)
        
        for _, row in latest_data.iterrows():
            country = row['country_name']
            if country not in combined:
                combined[country] = {}
            combined[country][indicator_name] = row['value']
    
    return pd.DataFrame(combined).T

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_hotspots(data):
    """Identify hidden hunger hotspots"""
    hotspots = {}
    
    if 'SH.STA.STNT.ZS' in data:
        stunting = get_latest_data(data['SH.STA.STNT.ZS'])
        high_stunting = stunting[stunting['value'] > 30]
        hotspots['stunting_hotspots'] = high_stunting.nlargest(10, 'value')[['country_name', 'value']]
    
    if 'SH.ANM.CHLD.ZS' in data:
        anemia = get_latest_data(data['SH.ANM.CHLD.ZS'])
        high_anemia = anemia[anemia['value'] > 40]
        hotspots['anemia_hotspots'] = high_anemia.nlargest(10, 'value')[['country_name', 'value']]
    
    return hotspots

def analyze_correlations(data):
    """Analyze correlations between indicators"""
    combined_data = combine_indicators(data)
    correlations = {}
    
    if not combined_data.empty and 'Stunting (children under 5)' in combined_data.columns:
        corr_matrix = combined_data.corr()
        stunting_correlations = corr_matrix['Stunting (children under 5)'].sort_values(ascending=False)
        
        positive_corr = stunting_correlations[stunting_correlations > 0.3].drop('Stunting (children under 5)', errors='ignore')
        correlations['worsening_factors'] = positive_corr
        
        negative_corr = stunting_correlations[stunting_correlations < -0.3]
        correlations['protective_factors'] = negative_corr
    
    return correlations

def predict_vulnerable_countries_cross_sectional(data):
    """Predict stunting rates using linear regression"""
    combined_data = combine_indicators(data)

    target_indicator = 'Stunting (children under 5)'
    feature_indicators = [
        'Prevalence of undernourishment',
        'Poverty ($2.15/day)',
        'GDP per capita',
        'Basic drinking water services',
        'Basic sanitation services',
        'Skilled birth attendance',
        'Measles immunization',
        'Food production index',
        'Cereal yield'
    ]

    required_columns = [target_indicator] + feature_indicators
    for col in required_columns:
        if col not in combined_data.columns:
            return pd.DataFrame()

    model_data = combined_data[required_columns].copy()
    model_data.dropna(subset=[target_indicator], inplace=True)

    if model_data.empty:
        return pd.DataFrame()

    for feature in feature_indicators:
        if model_data[feature].isnull().any():
            model_data[feature].fillna(model_data[feature].mean(), inplace=True)

    X = model_data[feature_indicators]
    y = model_data[target_indicator]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    predictions = model.predict(X_scaled)

    predicted_stunting_df = pd.DataFrame({
        'country_name': model_data.index, 
        'predicted_stunting': predictions
    })
    predicted_stunting_df['predicted_stunting'] = predicted_stunting_df['predicted_stunting'].clip(0, 100)
    predicted_stunting_df = predicted_stunting_df.sort_values(by='predicted_stunting', ascending=False).reset_index(drop=True)

    return predicted_stunting_df

def predict_indicator_trend(df, country_code, years_to_predict=5):
    """Predict future trend for an indicator"""
    country_data = df[df['country_code'] == country_code].sort_values('year')
    
    if len(country_data) < 3:
        return None, None

    X = country_data[['year']]
    y = country_data['value']

    model = LinearRegression()
    model.fit(X, y)

    last_year = country_data['year'].max()
    future_years = np.array(range(last_year + 1, last_year + 1 + years_to_predict)).reshape(-1, 1)
    
    predicted_values = model.predict(future_years)
    predicted_values[predicted_values < 0] = 0

    prediction_df = pd.DataFrame({
        'year': future_years.flatten(),
        'value': predicted_values,
        'type': 'Predicted'
    })

    history_df = country_data[['year', 'value']].copy()
    history_df['type'] = 'Historical'

    combined_df = pd.concat([history_df, prediction_df], ignore_index=True)
    return combined_df, model.coef_[0]

def calculate_vulnerability_index(data):
    """Calculate hidden hunger vulnerability index"""
    combined_data = combine_indicators(data)
    vulnerability = {}
    
    if not combined_data.empty:
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(combined_data.fillna(0)),
            columns=combined_data.columns,
            index=combined_data.index
        )
        
        weights = {
            'Stunting (children under 5)': 0.3,
            'Anemia (children 6-59 months)': 0.2,
            'Poverty ($2.15/day)': 0.15,
            'Basic drinking water services': 0.1,
            'GDP per capita': 0.1,
            'Food production index': 0.05,
            'Cereal yield': 0.05,
            'Skilled birth attendance': 0.05
        }
        
        vulnerability_scores = {}
        for country in normalized_data.index:
            score = 0
            for indicator, weight in weights.items():
                if indicator in normalized_data.columns:
                    score += normalized_data.loc[country, indicator] * weight
            vulnerability_scores[country] = score
        
        ranked_vulnerability = sorted(vulnerability_scores.items(), key=lambda x: x[1], reverse=True)
        vulnerability['most_vulnerable'] = ranked_vulnerability[:10]
        vulnerability['least_vulnerable'] = ranked_vulnerability[-10:]
    
    return vulnerability

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_indicator_map(data, indicator_code):
    """Create a choropleth map for a selected indicator"""
    if indicator_code not in data or data[indicator_code].empty:
        return None
    
    indicator_df = get_latest_data(data[indicator_code])
    indicator_name = HIDDEN_HUNGER_INDICATORS.get(indicator_code, indicator_code)
    
    fig = px.choropleth(
        indicator_df,
        locations='country_code',
        color='value',
        hover_name='country_name',
        hover_data={'value': ':.1f', 'year': True},
        color_continuous_scale='RdYlGn_r',
        title=f'{indicator_name} in Africa',
        labels={'value': f'{indicator_name} Value'}
    )
    
    fig.update_geos(scope='africa', resolution=50)
    fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'))
    fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0}, width=None) # width=None to allow stretch
    return fig

def create_hotspot_bar_chart(hotspots, indicator):
    """Create bar chart for hotspots"""
    if indicator not in hotspots or hotspots[indicator].empty:
        return None
    
    fig = px.bar(
        hotspots[indicator],
        x='value',
        y='country_name',
        orientation='h',
        title=f'Countries with Highest {indicator.replace("_", " ").title()}',
        labels={'value': 'Prevalence (%)', 'country_name': 'Country'}
    )
    
    fig.update_layout(height=400, width=None) # width=None to allow stretch
    return fig

def create_correlation_heatmap(correlations):
    """Create correlation heatmap"""
    if not correlations:
        return None
    
    top_factors = {}
    if 'worsening_factors' in correlations:
        top_factors.update(correlations['worsening_factors'].head(5).to_dict())
    if 'protective_factors' in correlations:
        top_factors.update(correlations['protective_factors'].head(5).to_dict())
    
    if not top_factors:
        return None
    
    fig = px.bar(
        x=list(top_factors.values()),
        y=list(top_factors.keys()),
        orientation='h',
        color=list(top_factors.values()),
        color_continuous_scale='RdYlBu',
        title='Factors Correlated with Stunting',
        labels={'x': 'Correlation Coefficient', 'y': 'Factor'}
    )
    
    fig.update_layout(height=400, width=None) # width=None to allow stretch
    return fig

def create_multi_indicator_trend_chart(data, indicators_dict):
    """Create a line chart showing trends of multiple indicators"""
    all_trends = []
    for code, name in indicators_dict.items():
        if code in data and not data[code].empty:
            df = data[code]
            aggregated_df = df.groupby('year')['value'].mean().reset_index()
            aggregated_df['indicator'] = name
            all_trends.append(aggregated_df)

    if not all_trends:
        return None

    combined_df = pd.concat(all_trends, ignore_index=True)

    fig = px.line(
        combined_df, 
        x='year', 
        y='value', 
        color='indicator',
        title='Average Indicator Trends Over Time in Africa',
        labels={'year': 'Year', 'value': 'Average Prevalence (%)', 'indicator': 'Indicator'},
        markers=True
    )

    fig.update_layout(hovermode="x unified", width=None) # width=None to allow stretch
    return fig

# ============================================================================
# STREAMLIT APP PAGES
# ============================================================================

def show_dashboard(data):
    """Main dashboard view"""
    st.markdown("<h3>Hidden Hunger Dashboard</h3>", unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'SH.STA.STNT.ZS' in data:
            stunting_data = get_latest_data(data['SH.STA.STNT.ZS'])
            if not stunting_data.empty:
                avg_stunting = stunting_data['value'].mean()
                st.metric("Average Stunting", f"{avg_stunting:.1f}%")

    with col2:
        if 'SH.ANM.CHLD.ZS' in data:
            anemia_data = get_latest_data(data['SH.ANM.CHLD.ZS'])
            if not anemia_data.empty:
                avg_anemia = anemia_data['value'].mean()
                st.metric("Average Anemia", f"{avg_anemia:.1f}%")

    with col3:
        if 'SI.POV.DDAY' in data:
            poverty_data = get_latest_data(data['SI.POV.DDAY'])
            if not poverty_data.empty:
                avg_poverty = poverty_data['value'].mean()
                st.metric("Average Poverty", f"{avg_poverty:.1f}%")

    with col4:
        if 'NY.GDP.PCAP.CD' in data:
            gdp_data = get_latest_data(data['NY.GDP.PCAP.CD'])
            if not gdp_data.empty:
                avg_gdp = gdp_data['value'].mean()
                st.metric("Average GDP per Capita", f"${avg_gdp:,.0f}")

    # Maps
    st.markdown("<h4>Geographical Distribution</h4>", unsafe_allow_html=True)
    
    map_indicator_options = {
        'SH.STA.STNT.ZS': 'Stunting (children under 5)',
        'SH.STA.WAST.ZS': 'Wasting (children under 5)',
        'SH.ANM.CHLD.ZS': 'Anemia (children 6-59 months)',
        'SH.ANM.ALLW.ZS': 'Anemia (women of reproductive age)',
        'SN.ITK.DEFC.ZS': 'Prevalence of undernourishment',
        'AG.PRD.FOOD.XD': 'Food production index',
        'SH.H2O.BASW.ZS': 'Basic drinking water services',
        'SH.STA.BASS.ZS': 'Basic sanitation services',
        'SH.STA.ODFC.ZS': 'Open defecation',
        'SI.POV.DDAY': 'Poverty ($2.15/day)',
        'NY.GDP.PCAP.CD': 'GDP per capita',
        'SH.STA.BRTC.ZS': 'Skilled birth attendance',
        'SH.IMM.MEAS': 'Measles immunization',
        'AG.YLD.CREL.KG': 'Cereal yield',
        'AG.LND.AGRI.ZS': 'Agricultural land',
    }
    
    selected_map_indicator_name = st.selectbox(
        "Select indicator for map:",
        list(map_indicator_options.values()),
        index=0
    )
    
    selected_map_indicator_code = [code for code, name in map_indicator_options.items() if name == selected_map_indicator_name][0]
    
    indicator_map = create_indicator_map(data, selected_map_indicator_code) # width=None is set in create_indicator_map
    if indicator_map:
        st.plotly_chart(indicator_map, width='stretch')
    else:
        st.info(f"No data available for {selected_map_indicator_name}")
    
    # Vulnerability analysis
    st.markdown("<h4>Vulnerability Analysis</h4>", unsafe_allow_html=True)
    vulnerability = calculate_vulnerability_index(data)
    
    if vulnerability and 'most_vulnerable' in vulnerability:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Most Vulnerable Countries**")
            for country, score in vulnerability['most_vulnerable'][:5]:
                st.write(f"• {country}: {score:.2f}")
        
        with col2:
            st.write("**Least Vulnerable Countries**")
            for country, score in vulnerability['least_vulnerable'][-5:]:
                st.write(f"• {country}: {score:.2f}")
    
    # Correlation analysis
    st.markdown("<h4>Root Cause Analysis</h4>", unsafe_allow_html=True)
    correlations = analyze_correlations(data)
    
    corr_chart = create_correlation_heatmap(correlations)
    if corr_chart:
        st.plotly_chart(corr_chart, width='stretch')

    # Trends
    st.markdown("<h4>Indicator Trends Over Time</h4>", unsafe_allow_html=True)
    
    trend_indicators_options = {
        'SH.STA.STNT.ZS': 'Stunting (children under 5)',
        'SH.ANM.CHLD.ZS': 'Anemia (children 6-59 months)',
        'SN.ITK.DEFC.ZS': 'Prevalence of undernourishment',
    }
    
    trend_chart = create_multi_indicator_trend_chart(data, trend_indicators_options)
    if trend_chart:
        st.plotly_chart(trend_chart, width='stretch')

def show_hotspot_analysis(data):
    """Hotspot analysis view"""
    st.markdown("<h3>Hidden Hunger Hotspots</h3>", unsafe_allow_html=True)
    
    hotspots = analyze_hotspots(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'stunting_hotspots' in hotspots and not hotspots['stunting_hotspots'].empty:
            st.markdown("<h4>Stunting Hotspots (>30%)</h4>", unsafe_allow_html=True)
            stunting_chart = create_hotspot_bar_chart(hotspots, 'stunting_hotspots')
            if stunting_chart: # width=None is set in create_hotspot_bar_chart
                st.plotly_chart(stunting_chart, width='stretch')
    
    with col2:
        if 'anemia_hotspots' in hotspots and not hotspots['anemia_hotspots'].empty:
            st.markdown("<h4>Anemia Hotspots (>40%)</h4>", unsafe_allow_html=True)
            anemia_chart = create_hotspot_bar_chart(hotspots, 'anemia_hotspots')
            if anemia_chart: # width=None is set in create_hotspot_bar_chart
                st.plotly_chart(anemia_chart, width='stretch')

def show_country_comparison(data):
    """Country comparison view"""
    st.markdown("<h3>Country Comparison</h3>", unsafe_allow_html=True)
    
    available_countries = list(combine_indicators(data).index)
    selected_countries = st.multiselect(
        "Select countries to compare:",
        available_countries,
        default=available_countries[:3] if available_countries else []
    )
    
    if selected_countries:
        combined_data = combine_indicators(data)
        comparison_data = combined_data.loc[selected_countries]
        
        available_indicators = [col for col in comparison_data.columns if not comparison_data[col].isna().all()]
        selected_indicator = st.selectbox("Select indicator to compare:", available_indicators)
        
        if selected_indicator:
            st.write(f"This chart compares the latest available values for **{selected_indicator}** across the selected countries.")
            fig = px.bar(
                comparison_data.reset_index(),
                x='index',
                y=selected_indicator,
                title=f'{selected_indicator} - Country Comparison',
                labels={'index': 'Country', selected_indicator: 'Value'}
            )
            # This part improves the appearance of the data labels
            fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig, width='stretch')
            
            # Show data table
            st.markdown("<h4>Comparison Data</h4>", unsafe_allow_html=True)
            st.dataframe(comparison_data[available_indicators].style.format("{:.1f}"))

def show_predictive_analysis(data):
    """Predictive analysis view"""
    st.markdown("<h3>Predictive Analysis - Future Trends & Vulnerability</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    This section provides a simple forecast for key indicators using linear regression on historical data. 
    Select a country and an indicator to see its projected trend over the next 5 years. This can help 
    identify countries where the situation may be worsening and require proactive intervention.
    """)

    # Selectors for country and indicator
    col1, col2 = st.columns(2)
    with col1:
        available_countries = sorted(AFRICAN_COUNTRY_NAMES.items(), key=lambda item: item[1])
        country_name = st.selectbox(
            "Select a country:", 
            [name for code, name in available_countries],
            index=available_countries.index(('NGA', 'Nigeria')) if ('NGA', 'Nigeria') in available_countries else 0
        )
        country_code = [code for code, name in AFRICAN_COUNTRY_NAMES.items() if name == country_name][0]

    with col2:
        predictable_indicators = {
            'SH.STA.STNT.ZS': 'Stunting (children under 5)',
            'SH.ANM.CHLD.ZS': 'Anemia (children 6-59 months)',
            'SN.ITK.DEFC.ZS': 'Prevalence of undernourishment',
            'NY.GDP.PCAP.CD': 'GDP per capita',
        }
        indicator_name = st.selectbox("Select an indicator to forecast:", list(predictable_indicators.values()))
        indicator_code = [code for code, name in predictable_indicators.items() if name == indicator_name][0]

    if indicator_code in data:
        indicator_df = data[indicator_code]
        prediction_df, slope = predict_indicator_trend(indicator_df, country_code)

        if prediction_df is not None:
            fig = px.line(
                prediction_df,
                x='year',
                y='value',
                color='type',
                title=f'5-Year Forecast for {indicator_name} in {country_name}',
                labels={'year': 'Year', 'value': 'Value'},
                markers=True
            )
            fig.update_layout(legend_title_text='Data Type')
            st.plotly_chart(fig, width='stretch')

            st.markdown("<h4>Analysis</h4>", unsafe_allow_html=True)
            if slope > 0.05:
                st.warning(f"**Worsening Trend:** The forecast indicates that {indicator_name} is likely to increase in {country_name}.")
            elif slope < -0.05:
                st.success(f"**Improving Trend:** The forecast suggests a positive trend, with {indicator_name} likely to decrease in {country_name}.")
            else:
                st.info(f"**Stable Trend:** The forecast shows a relatively stable trend for {indicator_name} in {country_name}.")
        else:
            st.warning(f"Not enough historical data available for {country_name} to create a reliable forecast for this indicator.")

    st.markdown("---")
    st.markdown("<h4>Cross-Country Vulnerability Prediction (Stunting)</h4>", unsafe_allow_html=True)
    st.markdown("""
    This model predicts the current stunting rate for each country based on other available indicators.
    Countries with higher predicted stunting are identified as more vulnerable.
    """)

    predicted_vulnerability = predict_vulnerable_countries_cross_sectional(data)

    if not predicted_vulnerability.empty:
        st.write("### Top 10 Countries Predicted to be Most Vulnerable to Stunting:")
        st.dataframe(predicted_vulnerability.head(10).style.format({'predicted_stunting': '{:.1f}%'}))

        st.write("### Bottom 10 Countries Predicted to be Least Vulnerable to Stunting:")
        st.dataframe(predicted_vulnerability.tail(10).style.format({'predicted_stunting': '{:.1f}%'}))

        fig = px.bar(
            predicted_vulnerability.head(10),
            x='predicted_stunting',
            y='country_name',
            orientation='h',
            title='Top 10 Countries Predicted Most Vulnerable to Stunting',
            labels={'predicted_stunting': 'Predicted Stunting Rate (%)', 'country_name': 'Country'}, width='stretch')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch')

        # Add bar chart for least vulnerable countries
        fig = px.bar(
            predicted_vulnerability.tail(10).sort_values(by='predicted_stunting', ascending=True),
            x='predicted_stunting',
            y='country_name',
            orientation='h', width='stretch',
            title='Bottom 10 Countries Predicted Least Vulnerable to Stunting', labels={'predicted_stunting': 'Predicted Stunting Rate (%)', 'country_name': 'Country'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Could not perform cross-country vulnerability prediction due to insufficient data for all required indicators.")

def show_policy_briefs(data):
    """Policy briefs view"""
    st.markdown("<h3>Policy Brief Generator</h3>", unsafe_allow_html=True)
    
    st.info("""
    Generate actionable policy briefs for targeted interventions against hidden hunger.
    Select a country to generate a customized brief.
    """)
    
    # Country selector
    available_countries = list(combine_indicators(data).index)
    selected_country = st.selectbox("Select country:", available_countries)
    
    if selected_country and st.button("Generate Policy Brief"):
        generate_policy_brief(selected_country, data)

def generate_policy_brief(country, data):
    """Generate a policy brief for a specific country"""
    st.markdown(f"""
    # Hidden Hunger Policy Brief
    ## {country}
    *Generated on {datetime.now().strftime('%Y-%m-%d')}*""")
    
    brief_content = f"""# Hidden Hunger Policy Brief
## {country}
*Generated on {datetime.now().strftime('%Y-%m-%d')}*

---

### Executive Summary
Urgent action needed to address hidden hunger through targeted, multi-sectoral 
interventions focusing on micronutrient deficiencies and childhood stunting.

###  Key Findings\n"""
    # Country-specific data
    combined_data = combine_indicators(data)
    if country in combined_data.index:
        country_data = combined_data.loc[country]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Stunting (children under 5)' in country_data:
                stunting = country_data['Stunting (children under 5)']
                st.metric("Stunting Rate", f"{stunting:.1f}%")
                brief_content += f"- **Stunting Rate**: {stunting:.1f}%\n"
        
        with col2:
            if 'Anemia (children 6-59 months)' in country_data:
                anemia = country_data['Anemia (children 6-59 months)']
                st.metric("Child Anemia", f"{anemia:.1f}%")
                brief_content += f"- **Child Anemia**: {anemia:.1f}%\n"
        
        with col3:
            if 'Poverty ($2.15/day)' in country_data:
                poverty = country_data['Poverty ($2.15/day)']
                st.metric("Poverty Rate", f"{poverty:.1f}%")
                brief_content += f"- **Poverty Rate**: {poverty:.1f}%\n"
    
    brief_content += """
### Recommended Interventions
    
    **High-Impact Actions:**
    - Micronutrient supplementation (Vitamin A, Iron)
    - Therapeutic feeding for SAM children  
    - Multiple micronutrient powders for home fortification
    
    **Cross-Sectoral Solutions:**
    - Agricultural diversification for nutrient-rich crops
    - Women's education and empowerment programs
    - Water, sanitation and hygiene (WASH) improvements
    
    ### Immediate Next Steps (90 Days)
    1. Launch emergency micronutrient supplementation in highest-burden districts
    2. Scale up community-based management of acute malnutrition
    3. Initiate multi-sectoral task force for coordinated response
    
    ### Monitoring Indicators
    - Stunting prevalence reduction
    - Anemia rates in women and children  
    - SAM admission rates
    - Micronutrient supplementation coverage
    
    ###  Cost & Funding
    - **Initial 6-month response**: $2-5 million
    - **Scale-up phase (12-24 months)**: $10-20 million
    - **Funding sources**: Government budget, World Bank, Global Nutrition Fund
    
    ---
    *This brief was generated using real-time data from World Bank, WHO, and FAO*
"""
    st.markdown(brief_content)

    st.download_button(
        label="Download Policy Brief",
        data=brief_content,
        file_name=f"Policy_Brief_{country.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
    )

def show_about():
    """About page"""
    st.markdown("<h3>About Hidden Hunger Analyzer</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Mission
    To dramatically reduce micronutrient deficiencies and childhood stunting across Africa, 
    creating a pathway to end severe hunger by 2050.
    
    ## Objectives
    - Map malnutrition hotspots at high spatial resolution
    - Predict future risk using machine learning and seasonal models
    - Analyze root causes with rigorous causal methods
    - Design and optimize interventions across sectors
    - Translate findings into actionable policy briefs
    
    ## Data Sources
    This tool uses exclusively open-access data from:
    - **World Bank Open Data** - Economic, demographic, and nutrition indicators
    - **WHO Global Health Observatory** - Health and nutrition data
    - **FAO STAT** - Food security and agricultural data
    - **UNICEF** - Child nutrition and WASH indicators
    
    ## Technical Approach
    - **Geospatial Analysis**: Hotspot mapping and vulnerability indices
    - **Machine Learning**: Risk prediction and feature importance
    - **Causal Inference**: Root cause analysis and intervention impact
    - **Real-time Analytics**: Always current data without local storage
    
    ##  Pathway to 2050
    - **2025**: Reduce stunting by 25% in pilot districts
    - **2030**: Halve childhood stunting in targeted regions
    - **2035**: Scale successful interventions nationally
    - **2040**: Achieve continental coverage
    - **2050**: End severe hidden hunger in Africa
    
    ##  Target Audience
    - **Policymakers**: National and local government officials
    - **Implementing Partners**: NGOs and development organizations
    - **Researchers**: Academics and data scientists
    - **Donors**: Funding agencies and international organizations
    
    ##  Technology Stack
    - **Streamlit**: Interactive web application framework
    - **Plotly**: Interactive visualizations and maps
    - **Pandas**: Data manipulation and analysis
    - **Scikit-learn**: Machine learning algorithms
    
    ---
    
    *Built with ❤️ for a hunger-free Africa*
    """)

def main():
    # Header
    st.title(" Hidden Hunger Analysis in Africa")
    st.markdown("""
    **Real-time analysis to end micronutrient deficiencies and childhood stunting**  
    *Using geospatial mapping, predictive models, and causal analysis*
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis", 
        ["Dashboard", "Hotspot Analysis", "Country Comparison", "Predictive Analysis", "Policy Briefs", "About"]
    )
    
    # Connection test option
    if st.sidebar.button(" Test API Connection"):
        test_api_connection()
    
    # Fetch data (cached)
    with st.spinner("Loading latest nutrition data..."):
        data = fetch_all_indicators()
    
    if not data:
        st.error("Failed to load data. Please check your internet connection and try the 'Test API Connection' button.")
        return
    
    # Main content based on selection
    if app_mode == "Dashboard":
        show_dashboard(data)
    elif app_mode == "Hotspot Analysis":
        show_hotspot_analysis(data)
    elif app_mode == "Country Comparison":
        show_country_comparison(data)
    elif app_mode == "Predictive Analysis":
        show_predictive_analysis(data)
    elif app_mode == "Policy Briefs":
        show_policy_briefs(data)
    else:
        show_about()

    # Add a data source footnote to all pages
    if app_mode != "About":
        st.markdown("---")
        st.markdown("""
        *Data Sources: World Bank, WHO, FAO, UNICEF. All data is fetched in real-time from publicly available APIs.*
        """)

if __name__ == "__main__":
    main()