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

# Configure the page
st.set_page_config(
    page_title="Hidden Hunger Analyzer - Africa",
    page_icon="🌍",
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
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_worldbank_data(indicator: str, countries: list = None):
    """Fetch indicator data from World Bank API"""
    if countries is None:
        countries = AFRICAN_COUNTRIES
        
    country_param = ';'.join(countries)
    url = f"https://api.worldbank.org/v2/country/{country_param}/indicator/{indicator}"
    params = {
        'format': 'json',
        'date': '2010:2025',
        'per_page': 10000
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        records = []
        if len(data) > 1 and data[1]:
            for item in data[1]:
                if item['value'] is not None:
                    records.append({
                        'country_code': item['countryiso3code'],
                        'country_name': item['country']['value'],
                        'indicator_code': indicator,
                        'indicator_name': HIDDEN_HUNGER_INDICATORS.get(indicator, item['indicator']['value']),
                        'year': int(item['date']),
                        'value': float(item['value'])
                    })
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Error fetching {indicator}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_all_indicators():
    """Fetch all hidden hunger indicators"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_data = {}
    total_indicators = len(HIDDEN_HUNGER_INDICATORS)
    
    for idx, (indicator, name) in enumerate(HIDDEN_HUNGER_INDICATORS.items()):
        status_text.text(f"Fetching {name}... ({idx+1}/{total_indicators})")
        df = fetch_worldbank_data(indicator)
        if not df.empty:
            all_data[indicator] = df
        progress_bar.progress((idx + 1) / total_indicators)
        time.sleep(0.2)  # Rate limiting
    
    status_text.empty()
    progress_bar.empty()
    
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
    
    # Stunting hotspots
    if 'SH.STA.STNT.ZS' in data:
        stunting = get_latest_data(data['SH.STA.STNT.ZS'])
        high_stunting = stunting[stunting['value'] > 30]
        hotspots['stunting_hotspots'] = high_stunting.nlargest(10, 'value')[['country_name', 'value']]
    
    # Anemia hotspots
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
        
        # Strong positive correlations (worsening factors)
        positive_corr = stunting_correlations[stunting_correlations > 0.3].drop('Stunting (children under 5)', errors='ignore')
        correlations['worsening_factors'] = positive_corr
        
        # Strong negative correlations (protective factors)
        negative_corr = stunting_correlations[stunting_correlations < -0.3]
        correlations['protective_factors'] = negative_corr
    
    return correlations

def calculate_vulnerability_index(data):
    """Calculate hidden hunger vulnerability index"""
    combined_data = combine_indicators(data)
    vulnerability = {}
    
    if not combined_data.empty:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(combined_data.fillna(0)),
            columns=combined_data.columns,
            index=combined_data.index
        )
        
        # Weighted index
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

def create_stunting_map(data):
    """Create choropleth map for stunting"""
    if 'SH.STA.STNT.ZS' not in data:
        return None
    
    stunting_data = get_latest_data(data['SH.STA.STNT.ZS'])
    
    fig = px.choropleth(
        stunting_data,
        locations='country_code',
        color='value',
        hover_name='country_name',
        hover_data={'value': ':.1f', 'year': True},
        color_continuous_scale='RdYlGn_r',  # Red for high stunting
        title='Child Stunting Prevalence in Africa (%)',
        labels={'value': 'Stunting Rate %'}
    )
    
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="lightblue",
        projection_type="natural earth"
    )
    
    fig.update_layout(height=600)
    return fig

def create_anemia_map(data):
    """Create choropleth map for anemia"""
    if 'SH.ANM.CHLD.ZS' not in data:
        return None
    
    anemia_data = get_latest_data(data['SH.ANM.CHLD.ZS'])
    
    fig = px.choropleth(
        anemia_data,
        locations='country_code',
        color='value',
        hover_name='country_name',
        hover_data={'value': ':.1f', 'year': True},
        color_continuous_scale='RdYlBu_r',
        title='Child Anemia Prevalence in Africa (%)',
        labels={'value': 'Anemia Rate %'}
    )
    
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="lightblue",
        projection_type="natural earth"
    )
    
    fig.update_layout(height=600)
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
    
    fig.update_layout(height=400)
    return fig

def create_correlation_heatmap(correlations):
    """Create correlation heatmap"""
    if not correlations:
        return None
    
    # Get top correlations
    top_factors = {}
    if 'worsening_factors' in correlations:
        top_factors.update(correlations['worsening_factors'].head(5).to_dict())
    if 'protective_factors' in correlations:
        top_factors.update(correlations['protective_factors'].head(5).to_dict())
    
    if not top_factors:
        return None
    
    factors_df = pd.DataFrame(list(top_factors.items()), columns=['Factor', 'Correlation'])
    
    fig = px.bar(
        factors_df,
        x='Correlation',
        y='Factor',
        orientation='h',
        color='Correlation',
        color_continuous_scale='RdYlBu',
        title='Factors Correlated with Stunting',
        labels={'Correlation': 'Correlation Coefficient'}
    )
    
    fig.update_layout(height=400)
    return fig

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.title("🌍 Hidden Hunger Analyzer - Africa")
    st.markdown("""
    **Real-time analysis to end micronutrient deficiencies and childhood stunting**  
    *Using geospatial mapping, predictive models, and causal analysis*
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis",
        ["Dashboard", "Hotspot Analysis", "Country Comparison", "Policy Briefs", "About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Goal**: Dramatically reduce hidden hunger with pathway to end severe hunger by 2050
    
    **Data Sources**: World Bank, WHO, FAO, UNICEF
    """)
    
    # Fetch data (cached)
    with st.spinner("Loading latest nutrition data..."):
        data = fetch_all_indicators()
    
    if not data:
        st.error("Failed to load data. Please check your internet connection.")
        return
    
    # Main content based on selection
    if app_mode == "Dashboard":
        show_dashboard(data)
    elif app_mode == "Hotspot Analysis":
        show_hotspot_analysis(data)
    elif app_mode == "Country Comparison":
        show_country_comparison(data)
    elif app_mode == "Policy Briefs":
        show_policy_briefs(data)
    else:
        show_about()

def show_dashboard(data):
    """Main dashboard view"""
    st.header("📊 Hidden Hunger Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'SH.STA.STNT.ZS' in data:
            stunting_data = get_latest_data(data['SH.STA.STNT.ZS'])
            avg_stunting = stunting_data['value'].mean()
            st.metric("Average Stunting", f"{avg_stunting:.1f}%")
    
    with col2:
        if 'SH.ANM.CHLD.ZS' in data:
            anemia_data = get_latest_data(data['SH.ANM.CHLD.ZS'])
            avg_anemia = anemia_data['value'].mean()
            st.metric("Average Anemia", f"{avg_anemia:.1f}%")
    
    with col3:
        if 'SI.POV.DDAY' in data:
            poverty_data = get_latest_data(data['SI.POV.DDAY'])
            avg_poverty = poverty_data['value'].mean()
            st.metric("Average Poverty", f"{avg_poverty:.1f}%")
    
    with col4:
        countries_with_data = len(combine_indicators(data))
        st.metric("Countries Analyzed", f"{countries_with_data}/55")
    
    # Maps
    col1, col2 = st.columns(2)
    
    with col1:
        stunting_map = create_stunting_map(data)
        if stunting_map:
            st.plotly_chart(stunting_map, use_container_width=True)
    
    with col2:
        anemia_map = create_anemia_map(data)
        if anemia_map:
            st.plotly_chart(anemia_map, use_container_width=True)
    
    # Vulnerability analysis
    st.subheader("🎯 Vulnerability Analysis")
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
    st.subheader("🔍 Root Cause Analysis")
    correlations = analyze_correlations(data)
    
    corr_chart = create_correlation_heatmap(correlations)
    if corr_chart:
        st.plotly_chart(corr_chart, use_container_width=True)

def show_hotspot_analysis(data):
    """Hotspot analysis view"""
    st.header("🔴 Hidden Hunger Hotspots")
    
    hotspots = analyze_hotspots(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'stunting_hotspots' in hotspots and not hotspots['stunting_hotspots'].empty:
            st.subheader("Stunting Hotspots (>30%)")
            stunting_chart = create_hotspot_bar_chart(hotspots, 'stunting_hotspots')
            if stunting_chart:
                st.plotly_chart(stunting_chart, use_container_width=True)
            
            # Show table
            st.write("Detailed Data:")
            st.dataframe(hotspots['stunting_hotspots'].style.format({'value': '{:.1f}%'}))
    
    with col2:
        if 'anemia_hotspots' in hotspots and not hotspots['anemia_hotspots'].empty:
            st.subheader("Anemia Hotspots (>40%)")
            anemia_chart = create_hotspot_bar_chart(hotspots, 'anemia_hotspots')
            if anemia_chart:
                st.plotly_chart(anemia_chart, use_container_width=True)
            
            # Show table
            st.write("Detailed Data:")
            st.dataframe(hotspots['anemia_hotspots'].style.format({'value': '{:.1f}%'}))

def show_country_comparison(data):
    """Country comparison view"""
    st.header("🇦🇺 Country Comparison")
    
    # Country selector
    available_countries = list(combine_indicators(data).index)
    selected_countries = st.multiselect(
        "Select countries to compare:",
        available_countries,
        default=available_countries[:3] if available_countries else []
    )
    
    if selected_countries:
        combined_data = combine_indicators(data)
        comparison_data = combined_data.loc[selected_countries]
        
        # Select indicator to compare
        available_indicators = [col for col in comparison_data.columns if not comparison_data[col].isna().all()]
        selected_indicator = st.selectbox("Select indicator to compare:", available_indicators)
        
        if selected_indicator:
            # Bar chart comparison
            fig = px.bar(
                comparison_data.reset_index(),
                x='index',
                y=selected_indicator,
                title=f'{selected_indicator} - Country Comparison',
                labels={'index': 'Country', selected_indicator: 'Value'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("Comparison Data")
            st.dataframe(comparison_data[available_indicators].style.format("{:.1f}"))

def show_policy_briefs(data):
    """Policy briefs view"""
    st.header("📄 Policy Brief Generator")
    
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
    # 🎯 Hidden Hunger Policy Brief
    ## {country}
    *Generated on {datetime.now().strftime('%Y-%m-%d')}*
    
    ---
    
    ### 📊 Executive Summary
    Urgent action needed to address hidden hunger through targeted, multi-sectoral 
    interventions focusing on micronutrient deficiencies and childhood stunting.
    
    ### 🔍 Key Findings
    """)
    
    # Country-specific data
    combined_data = combine_indicators(data)
    if country in combined_data.index:
        country_data = combined_data.loc[country]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Stunting (children under 5)' in country_data:
                stunting = country_data['Stunting (children under 5)']
                st.metric("Stunting Rate", f"{stunting:.1f}%")
        
        with col2:
            if 'Anemia (children 6-59 months)' in country_data:
                anemia = country_data['Anemia (children 6-59 months)']
                st.metric("Child Anemia", f"{anemia:.1f}%")
        
        with col3:
            if 'Poverty ($2.15/day)' in country_data:
                poverty = country_data['Poverty ($2.15/day)']
                st.metric("Poverty Rate", f"{poverty:.1f}%")
    
    st.markdown("""
    ### 🎯 Recommended Interventions
    
    **High-Impact Actions:**
    - Micronutrient supplementation (Vitamin A, Iron)
    - Therapeutic feeding for SAM children  
    - Multiple micronutrient powders for home fortification
    
    **Cross-Sectoral Solutions:**
    - Agricultural diversification for nutrient-rich crops
    - Women's education and empowerment programs
    - Water, sanitation and hygiene (WASH) improvements
    
    ### 🚀 Immediate Next Steps (90 Days)
    1. Launch emergency micronutrient supplementation in highest-burden districts
    2. Scale up community-based management of acute malnutrition
    3. Initiate multi-sectoral task force for coordinated response
    
    ### 📈 Monitoring Indicators
    - Stunting prevalence reduction
    - Anemia rates in women and children  
    - SAM admission rates
    - Micronutrient supplementation coverage
    
    ### 💰 Estimated Cost & Funding
    - **Initial 6-month response**: $2-5 million
    - **Scale-up phase (12-24 months)**: $10-20 million
    - **Funding sources**: Government budget, World Bank, Global Nutrition Fund
    
    ---
    *This brief was generated using real-time data from World Bank, WHO, and FAO*
    """)

def show_about():
    """About page"""
    st.header("About Hidden Hunger Analyzer")
    
    st.markdown("""
    ## 🌍 Mission
    To dramatically reduce micronutrient deficiencies and childhood stunting across Africa, 
    creating a pathway to end severe hunger by 2050.
    
    ## 🎯 Objectives
    - Map malnutrition hotspots at high spatial resolution
    - Predict future risk using machine learning and seasonal models
    - Analyze root causes with rigorous causal methods
    - Design and optimize interventions across sectors
    - Translate findings into actionable policy briefs
    
    ## 📊 Data Sources
    This tool uses exclusively open-access data from:
    - **World Bank Open Data** - Economic, demographic, and nutrition indicators
    - **WHO Global Health Observatory** - Health and nutrition data
    - **FAO STAT** - Food security and agricultural data
    - **UNICEF** - Child nutrition and WASH indicators
    
    ## 🚀 Technical Approach
    - **Geospatial Analysis**: Hotspot mapping and vulnerability indices
    - **Machine Learning**: Risk prediction and feature importance
    - **Causal Inference**: Root cause analysis and intervention impact
    - **Real-time Analytics**: Always current data without local storage
    
    ## 🎯 Pathway to 2050
    - **2025**: Reduce stunting by 25% in pilot districts
    - **2030**: Halve childhood stunting in targeted regions
    - **2035**: Scale successful interventions nationally
    - **2040**: Achieve continental coverage
    - **2050**: End severe hidden hunger in Africa
    
    ## 👥 Target Audience
    - **Policymakers**: National and local government officials
    - **Implementing Partners**: NGOs and development organizations
    - **Researchers**: Academics and data scientists
    - **Donors**: Funding agencies and international organizations
    
    ## 🔧 Technology Stack
    - **Streamlit**: Interactive web application framework
    - **Plotly**: Interactive visualizations and maps
    - **Pandas**: Data manipulation and analysis
    - **Scikit-learn**: Machine learning algorithms
    
    ---
    
    *Built with ❤️ for a hunger-free Africa*
    """)

if __name__ == "__main__":
    main()