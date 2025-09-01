import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import time

# Set page config
st.set_page_config(page_title="CATL Performance Guarantee Valuation", layout="wide")

# Title and description
st.title("🔋 CATL Performance Guarantee Valuation Tool")
st.markdown("### Monte Carlo Simulation for Battery Energy Storage System")

# Sidebar for input parameters
st.sidebar.header("📊 Simulation Parameters")

# Create tabs for different parameter categories
tab1, tab2, tab3, tab4, tab5 = st.tabs(["System Specs", "Market Conditions", "Degradation", "Costs", "Risk Factors"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System Configuration")
        initial_capacity = st.number_input("Initial Capacity (MWh)", value=218.94, min_value=100.0, max_value=500.0)
        power_rating = st.number_input("Power Rating (MW)", value=113.31, min_value=50.0, max_value=250.0)
        roundtrip_efficiency = st.slider("Round-trip Efficiency (%)", min_value=85.0, max_value=98.0, value=96.5)
        
    with col2:
        st.subheader("Operating Profile")
        base_cycles_per_year = st.slider("Base Cycles per Year", min_value=100, max_value=500, value=365)
        project_lifetime = st.slider("Project Lifetime (years)", min_value=10, max_value=20, value=15)
        discount_rate = st.slider("Discount Rate (%)", min_value=3.0, max_value=15.0, value=7.0, step=0.5)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Energy Prices")
        avg_price_spread = st.number_input("Average Daily Price Spread ($/MWh)", value=50.0, min_value=10.0, max_value=200.0)
        price_volatility = st.slider("Price Spread Volatility (%)", min_value=10, max_value=100, value=30)
        price_growth_rate = st.slider("Annual Price Growth (%)", min_value=-5.0, max_value=10.0, value=2.0)
        
    with col2:
        st.subheader("Revenue Streams")
        capacity_payment = st.number_input("Capacity Payment ($/MW/year)", value=50000, min_value=0, max_value=200000)
        ancillary_revenue = st.number_input("Ancillary Services ($/MWh)", value=5.0, min_value=0.0, max_value=20.0)

with tab3:
    st.subheader("Degradation Parameters")
    col1, col2 = st.columns(2)
    with col1:
        degradation_scenario = st.selectbox(
            "Degradation Scenario",
            ["Guaranteed", "Optimistic (20% better)", "Pessimistic (20% worse)", "Custom"]
        )
        if degradation_scenario == "Custom":
            annual_degradation = st.slider("Annual Degradation Rate (%)", min_value=1.0, max_value=5.0, value=2.5)
        else:
            annual_degradation = 2.5  # Base case from contract
            
    with col2:
        degradation_uncertainty = st.slider("Degradation Uncertainty (%)", min_value=0, max_value=50, value=20)
        augmentation_threshold = st.slider("Augmentation Trigger (% of initial)", min_value=60, max_value=90, value=70)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Warranty Costs")
        initial_capex = st.number_input("Initial CAPEX ($M)", value=100.0, min_value=50.0, max_value=500.0)
        base_warranty_years = st.number_input("Base Warranty Period (years)", value=3, min_value=1, max_value=5)
        perf_guarantee_years = st.number_input("Performance Guarantee Period (years)", value=3, min_value=1, max_value=5)
        
    with col2:
        st.subheader("Extended Warranty Pricing")
        extended_warranty_y4_15 = st.slider("Years 4-15 (% of CAPEX/year)", min_value=0.5, max_value=3.0, value=1.5)
        extended_warranty_y16_20 = st.slider("Years 16-20 (% of CAPEX/year)", min_value=1.0, max_value=4.0, value=2.0)
        
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Failure Probabilities")
        module_failure_rate = st.slider("Annual Module Failure Rate (%)", min_value=0.1, max_value=5.0, value=1.0)
        serial_defect_prob = st.slider("Serial Defect Probability (%)", min_value=0.1, max_value=10.0, value=2.0)
        
    with col2:
        st.subheader("Downtime & Repairs")
        avg_repair_time = st.slider("Average Repair Time (days)", min_value=1, max_value=30, value=7)
        availability_target = st.slider("Availability Target (%)", min_value=90, max_value=99, value=97)

# Simulation settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Simulation Settings")
num_simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
include_guarantee = st.sidebar.checkbox("Include Performance Guarantee", value=True)
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")

# Monte Carlo Simulation Class
class BESSMonteCarloSimulation:
    def __init__(self, params):
        self.params = params
        self.results = {}
        
    def calculate_degradation_path(self, scenario_type="Guaranteed"):
        """Calculate capacity degradation over project lifetime"""
        years = np.arange(0, self.params['project_lifetime'] + 1)
        
        if scenario_type == "Guaranteed":
            # Use contract values for first 3 years
            guaranteed_values = [100, 94.40, 91.11, 88.97]
            if len(years) <= 3:
                return guaranteed_values[:len(years)]
            
            # Extrapolate for remaining years
            capacity_retention = np.zeros(len(years))
            capacity_retention[:4] = guaranteed_values
            
            for i in range(4, len(years)):
                annual_deg = self.params['annual_degradation'] * (1 + np.random.normal(0, self.params['degradation_uncertainty']/100))
                capacity_retention[i] = capacity_retention[i-1] * (1 - annual_deg/100)
        else:
            # Custom degradation path
            capacity_retention = np.zeros(len(years))
            capacity_retention[0] = 100
            
            for i in range(1, len(years)):
                annual_deg = self.params['annual_degradation'] * (1 + np.random.normal(0, self.params['degradation_uncertainty']/100))
                capacity_retention[i] = capacity_retention[i-1] * (1 - annual_deg/100)
                
        return capacity_retention
    
    def calculate_revenue(self, year, capacity_retention, cycles):
        """Calculate annual revenue based on capacity and market conditions"""
        effective_capacity = self.params['initial_capacity'] * (capacity_retention/100)
        
        # Energy arbitrage revenue
        price_spread = self.params['avg_price_spread'] * (1 + self.params['price_growth_rate']/100)**year
        price_spread *= (1 + np.random.normal(0, self.params['price_volatility']/100))
        
        energy_revenue = (
            effective_capacity * 
            cycles * 
            price_spread * 
            (self.params['roundtrip_efficiency']/100)
        )
        
        # Capacity payments
        capacity_revenue = self.params['power_rating'] * self.params['capacity_payment']
        
        # Ancillary services
        ancillary_revenue = effective_capacity * cycles * self.params['ancillary_revenue']
        
        return energy_revenue + capacity_revenue + ancillary_revenue
    
    def calculate_opex(self, year, has_warranty):
        """Calculate operating expenses"""
        base_opex = self.params['initial_capex'] * 1e6 * 0.015  # 1.5% of CAPEX
        
        if not has_warranty and year >= self.params['base_warranty_years']:
            # Additional maintenance without warranty
            base_opex *= 1.5
            
        return base_opex
    
    def calculate_warranty_cost(self, year):
        """Calculate warranty cost for given year"""
        if year < self.params['base_warranty_years']:
            return 0
        elif year < 15:
            return self.params['initial_capex'] * 1e6 * (self.params['extended_warranty_y4_15']/100)
        else:
            return self.params['initial_capex'] * 1e6 * (self.params['extended_warranty_y16_20']/100)
    
    def simulate_single_scenario(self, with_guarantee=True):
        """Run single simulation scenario"""
        cash_flows = []
        capacity_path = self.calculate_degradation_path(self.params['degradation_scenario'])
        
        for year in range(self.params['project_lifetime']):
            # Determine actual cycles (with some randomness)
            actual_cycles = self.params['base_cycles_per_year'] * np.random.uniform(0.9, 1.1)
            
            # Check for failures
            has_failure = np.random.random() < (self.params['module_failure_rate']/100)
            if has_failure:
                downtime_factor = 1 - (self.params['avg_repair_time']/365)
                actual_cycles *= downtime_factor
            
            # Calculate revenue
            revenue = self.calculate_revenue(year, capacity_path[year], actual_cycles)
            
            # Calculate costs
            opex = self.calculate_opex(year, with_guarantee)
            warranty_cost = self.calculate_warranty_cost(year) if with_guarantee else 0
            
            # Net cash flow
            net_cf = revenue - opex - warranty_cost
            cash_flows.append(net_cf)
            
        return np.array(cash_flows)
    
    def calculate_npv(self, cash_flows):
        """Calculate NPV of cash flows"""
        discount_factors = [(1 + self.params['discount_rate']/100)**(-i) for i in range(len(cash_flows))]
        return np.sum(cash_flows * discount_factors)
    
    def run_simulation(self, num_simulations=1000):
        """Run full Monte Carlo simulation"""
        npv_with_guarantee = []
        npv_without_guarantee = []
        
        for _ in range(num_simulations):
            # With guarantee
            cf_with = self.simulate_single_scenario(with_guarantee=True)
            npv_with_guarantee.append(self.calculate_npv(cf_with))
            
            # Without guarantee
            cf_without = self.simulate_single_scenario(with_guarantee=False)
            npv_without_guarantee.append(self.calculate_npv(cf_without))
        
        self.results = {
            'npv_with_guarantee': np.array(npv_with_guarantee),
            'npv_without_guarantee': np.array(npv_without_guarantee),
            'guarantee_value': np.array(npv_with_guarantee) - np.array(npv_without_guarantee)
        }
        
        return self.results

# Run simulation when button clicked
if run_simulation:
    with st.spinner('Running Monte Carlo simulation...'):
        # Prepare parameters
        sim_params = {
            'initial_capacity': initial_capacity,
            'power_rating': power_rating,
            'roundtrip_efficiency': roundtrip_efficiency,
            'base_cycles_per_year': base_cycles_per_year,
            'project_lifetime': project_lifetime,
            'discount_rate': discount_rate,
            'avg_price_spread': avg_price_spread,
            'price_volatility': price_volatility,
            'price_growth_rate': price_growth_rate,
            'capacity_payment': capacity_payment,
            'ancillary_revenue': ancillary_revenue,
            'degradation_scenario': degradation_scenario,
            'annual_degradation': annual_degradation,
            'degradation_uncertainty': degradation_uncertainty,
            'initial_capex': initial_capex,
            'base_warranty_years': base_warranty_years,
            'perf_guarantee_years': perf_guarantee_years,
            'extended_warranty_y4_15': extended_warranty_y4_15,
            'extended_warranty_y16_20': extended_warranty_y16_20,
            'module_failure_rate': module_failure_rate,
            'serial_defect_prob': serial_defect_prob,
            'avg_repair_time': avg_repair_time,
            'augmentation_threshold': augmentation_threshold
        }
        
        # Run simulation
        sim = BESSMonteCarloSimulation(sim_params)
        results = sim.run_simulation(num_simulations)
        
    # Display results
    st.markdown("---")
    st.header("📈 Simulation Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_value = np.mean(results['guarantee_value']) / 1e6
        st.metric("Mean Guarantee Value", f"${mean_value:.2f}M")
    
    with col2:
        prob_positive = np.sum(results['guarantee_value'] > 0) / len(results['guarantee_value']) * 100
        st.metric("Probability of Positive Value", f"{prob_positive:.1f}%")
    
    with col3:
        var_95 = np.percentile(results['guarantee_value'], 5) / 1e6
        st.metric("Value at Risk (95%)", f"${var_95:.2f}M")
    
    with col4:
        max_value = np.max(results['guarantee_value']) / 1e6
        st.metric("Maximum Value", f"${max_value:.2f}M")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # NPV Distribution
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=results['npv_with_guarantee']/1e6, name='With Guarantee', opacity=0.7))
        fig1.add_trace(go.Histogram(x=results['npv_without_guarantee']/1e6, name='Without Guarantee', opacity=0.7))
        fig1.update_layout(
            title='NPV Distribution Comparison',
            xaxis_title='NPV ($M)',
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Guarantee Value Distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=results['guarantee_value']/1e6, name='Guarantee Value'))
        fig2.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig2.update_layout(
            title='Performance Guarantee Value Distribution',
            xaxis_title='Value ($M)',
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    
    summary_data = {
        'Metric': ['Mean NPV', 'Std Dev', '5th Percentile', '25th Percentile', 
                   'Median', '75th Percentile', '95th Percentile'],
        'With Guarantee ($M)': [
            np.mean(results['npv_with_guarantee'])/1e6,
            np.std(results['npv_with_guarantee'])/1e6,
            np.percentile(results['npv_with_guarantee'], 5)/1e6,
            np.percentile(results['npv_with_guarantee'], 25)/1e6,
            np.median(results['npv_with_guarantee'])/1e6,
            np.percentile(results['npv_with_guarantee'], 75)/1e6,
            np.percentile(results['npv_with_guarantee'], 95)/1e6
        ],
        'Without Guarantee ($M)': [
            np.mean(results['npv_without_guarantee'])/1e6,
            np.std(results['npv_without_guarantee'])/1e6,
            np.percentile(results['npv_without_guarantee'], 5)/1e6,
            np.percentile(results['npv_without_guarantee'], 25)/1e6,
            np.median(results['npv_without_guarantee'])/1e6,
            np.percentile(results['npv_without_guarantee'], 75)/1e6,
            np.percentile(results['npv_without_guarantee'], 95)/1e6
        ],
        'Difference ($M)': [
            np.mean(results['guarantee_value'])/1e6,
            np.std(results['guarantee_value'])/1e6,
            np.percentile(results['guarantee_value'], 5)/1e6,
            np.percentile(results['guarantee_value'], 25)/1e6,
            np.median(results['guarantee_value'])/1e6,
            np.percentile(results['guarantee_value'], 75)/1e6,
            np.percentile(results['guarantee_value'], 95)/1e6
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(2)
    st.dataframe(summary_df, use_container_width=True)
    
    # Sensitivity Analysis
    st.markdown("---")
    st.subheader("🎯 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if mean_value > 0:
            st.success(f"✅ The Performance Guarantee creates an expected value of ${mean_value:.2f}M")
        else:
            st.warning(f"⚠️ The Performance Guarantee destroys ${abs(mean_value):.2f}M in expected value")
    
    with col2:
        if prob_positive > 70:
            st.success(f"✅ High confidence ({prob_positive:.1f}%) that guarantee adds value")
        elif prob_positive > 50:
            st.info(f"ℹ️ Moderate confidence ({prob_positive:.1f}%) that guarantee adds value")
        else:
            st.warning(f"⚠️ Low confidence ({prob_positive:.1f}%) that guarantee adds value")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if mean_value > 0 and prob_positive > 60:
        st.markdown("""
        **Recommendation: ACCEPT the Performance Guarantee**
        - The guarantee provides positive expected value
        - Risk mitigation benefits outweigh the costs
        - Consider negotiating for better terms if possible
        """)
    elif mean_value > 0 and prob_positive <= 60:
        st.markdown("""
        **Recommendation: NEGOTIATE the Performance Guarantee**
        - The guarantee shows marginal value
        - High uncertainty in outcomes
        - Focus on reducing warranty costs or improving terms
        """)
    else:
        st.markdown("""
        **Recommendation: DECLINE the Performance Guarantee**
        - The guarantee does not provide sufficient value
        - Self-insurance may be more cost-effective
        - Consider alternative risk mitigation strategies
        """)

# Add information panel
with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    ### Using the CATL Performance Guarantee Valuation Tool
    
    1. **Adjust Parameters**: Use the tabs above to modify system specifications, market conditions, and risk factors
    2. **Set Simulation Parameters**: Choose the number of simulations in the sidebar (more = more accurate but slower)
    3. **Run Simulation**: Click the "Run Simulation" button to generate results
    4. **Interpret Results**: 
       - Positive guarantee value means the warranty creates value
       - Check the probability of positive value for confidence level
       - Review the distribution charts to understand risk profile
    5. **Make Decision**: Use the recommendations and insights to inform your decision
    
    ### Key Metrics Explained:
    - **Mean Guarantee Value**: Average financial benefit of having the guarantee
    - **Probability of Positive Value**: Likelihood that the guarantee will be beneficial
    - **Value at Risk (95%)**: Worst-case scenario (95% confidence level)
    - **NPV Distribution**: Shows the range of possible financial outcomes
    """)