import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import time

# Set page config with minimalist theme
st.set_page_config(
    page_title="CATL Performance Guarantee Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimalist design
st.markdown("""
<style>
    /* Clean, minimal typography */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    /* Subtle borders and spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 0px;
        background-color: transparent;
        border: none;
        color: #666;
        font-weight: 400;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #000;
        color: #000;
        font-weight: 500;
    }
    
    /* Clean metric styling */
    [data-testid="metric-container"] {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        padding: 16px;
        border-radius: 4px;
        box-shadow: none;
    }
    
    /* Subtle button styling */
    .stButton > button {
        background-color: #000;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 24px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #333;
    }
    
    /* Clean input fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 8px;
    }
    
    /* Remove excessive decorations */
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 32px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header with clean typography
st.markdown("# Performance Guarantee Valuation")
st.markdown("##### Battery Energy Storage System - Monte Carlo Analysis")

# Instructions card at the top
with st.container():
    st.markdown("---")
    st.markdown("""
    ### How to Use This Tool
    
    **Step 1:** Configure your system parameters across the five tabs below  
    **Step 2:** Adjust simulation settings in the sidebar (number of iterations)  
    **Step 3:** Click 'Run Simulation' to generate probabilistic analysis  
    **Step 4:** Review three-scenario comparison: No Extension vs Performance Only vs Full Warranty  
    
    *The tool will compare NPV across all three warranty scenarios to help optimize your decision.*
    """)
    st.markdown("---")

# Sidebar - simplified and clear
with st.sidebar:
    st.markdown("### Simulation Controls")
    
    num_simulations = st.slider(
        "Simulation Iterations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="More iterations increase accuracy but take longer to compute"
    )
    
    st.markdown("---")
    
    run_simulation = st.button(
        "Run Simulation",
        type="primary",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("""
    <small style='color: #666;'>
    Computation time: ~{:.1f} seconds
    </small>
    """.format(num_simulations/1000 * 3), unsafe_allow_html=True)  # x3 for three scenarios

# Main parameter tabs - cleaner organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "System Configuration",
    "Market Parameters", 
    "Degradation Model",
    "Cost Structure",
    "Risk Factors"
])

# Initialize default values
capacity_payment_rate = 50000
rte_for_capacity = 86.0
base_cycles_per_year = 365

with tab1:
    st.markdown("#### Battery System Specifications")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capacity = st.number_input(
            "Initial Capacity (MWh)",
            value=218.94,
            min_value=100.0,
            max_value=500.0,
            format="%.2f",
            help="Nameplate energy capacity"
        )
        
    with col2:
        power_rating = st.number_input(
            "Power Rating (MW)",
            value=100.0,
            min_value=50.0,
            max_value=250.0,
            format="%.2f",
            help="Maximum charge/discharge power (fixed, does not degrade)"
        )
        
    with col3:
        roundtrip_efficiency = st.slider(
            "Round-trip Efficiency (%)",
            min_value=85.0,
            max_value=98.0,
            value=96.5,
            format="%.1f",
            help="DC-AC efficiency"
        )
    
    st.markdown("#### Operating Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Cycles per day selector with more options
        cycle_options = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, "Custom"]
        cycles_selection = st.selectbox(
            "Daily Cycling Pattern",
            options=cycle_options,
            index=2,  # Default to 1.0 cycles per day
            format_func=lambda x: f"{x} cycle{'s' if x != 1 else ''}/day" if x != "Custom" else "Custom",
            help="Number of complete charge-discharge cycles per day. Higher cycling can increase revenue but accelerates degradation."
        )
        
        # Handle custom input
        if cycles_selection == "Custom":
            cycles_per_day = st.number_input(
                "Enter custom cycles/day",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                format="%.1f"
            )
        else:
            cycles_per_day = cycles_selection
        
        # Calculate and display annual cycles
        base_cycles_per_year = cycles_per_day * 365
        st.caption(f"→ {base_cycles_per_year:.0f} cycles/year")
        
        # Add note about typical assumptions
        if abs(cycles_per_day - 1.0) > 0.1:
            st.caption("⚠️ Typical assumption: 365 cycles/year (1.0/day)")
        
    with col2:
        project_lifetime = st.slider(
            "Project Lifetime (years)",
            min_value=10,
            max_value=20,
            value=15
        )
        
    with col3:
        discount_rate = st.slider(
            "Discount Rate (%)",
            min_value=3.0,
            max_value=15.0,
            value=7.0,
            step=0.5,
            format="%.1f"
        )

with tab2:
    st.markdown("#### Energy Market Assumptions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_price_spread = st.number_input(
            "Average Daily Spread ($/MWh)",
            value=50.0,
            min_value=10.0,
            max_value=200.0,
            format="%.1f",
            help="Peak-to-trough price differential"
        )
        
    with col2:
        price_volatility = st.slider(
            "Price Volatility (%)",
            min_value=10,
            max_value=100,
            value=30,
            help="Standard deviation of price spreads"
        )
        
    with col3:
        price_growth_rate = st.slider(
            "Annual Price Growth (%)",
            min_value=-5.0,
            max_value=10.0,
            value=2.0,
            format="%.1f"
        )
    
    st.markdown("#### Additional Revenue Streams")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Capacity Payment**")
        st.caption("Based on available MWh/4 at point of connection")
        
        # Display the calculation formula
        st.latex(r"\text{Payment} = \frac{\text{MWh} \times \text{SOH} \times \text{RTE}}{4} \times \text{\$/MWh/year}")
        
        capacity_payment_rate = st.number_input(
            "Capacity Payment Rate ($/MWh/year)",
            value=50000,
            min_value=0,
            max_value=200000,
            step=5000,
            format="%d",
            help="Payment per MWh of available capacity divided by 4"
        )
        
    with col2:
        ancillary_revenue = st.number_input(
            "Ancillary Services ($/MWh)",
            value=5.0,
            min_value=0.0,
            max_value=20.0,
            format="%.1f"
        )
        
        # RTE for capacity payment calculation
        rte_for_capacity = st.number_input(
            "RTE for Capacity Payment (%)",
            value=86.0,
            min_value=80.0,
            max_value=95.0,
            format="%.1f",
            help="Round-trip efficiency at point of connection (typically 86%)"
        )

with tab3:
    st.markdown("#### Degradation Modeling")
    col1, col2 = st.columns(2)
    
    with col1:
        degradation_scenario = st.selectbox(
            "Degradation Scenario",
            ["Guaranteed (Contract)", "Optimistic (-20%)", "Pessimistic (+20%)", "Custom"],
            help="Based on CATL warranty tables"
        )
        
        if degradation_scenario == "Custom":
            annual_degradation = st.slider(
                "Annual Degradation Rate (%)",
                min_value=1.0,
                max_value=5.0,
                value=2.5,
                format="%.1f"
            )
        else:
            annual_degradation = 2.5
            st.info(f"Using contract baseline: {annual_degradation}% per year")
            
    with col2:
        degradation_uncertainty = st.slider(
            "Degradation Uncertainty (%)",
            min_value=0,
            max_value=50,
            value=20,
            help="Stochastic variation in degradation"
        )
        
        augmentation_threshold = st.slider(
            "Augmentation Threshold (%)",
            min_value=60,
            max_value=90,
            value=70,
            help="Capacity level triggering augmentation"
        )

with tab4:
    st.markdown("#### Capital and Warranty Costs")
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capex = st.number_input(
            "Initial CAPEX ($M)",
            value=55.0,
            min_value=20.0,
            max_value=500.0,
            format="%.1f",
            help="Total capital expenditure for battery system"
        )
        
        base_warranty_years = st.number_input(
            "Base Warranty Period (years)",
            value=3,
            min_value=1,
            max_value=5,
            help="Initial warranty coverage period (included in CAPEX)"
        )
        
    with col2:
        perf_guarantee_years = st.number_input(
            "Initial Performance Guarantee (years)",
            value=3,
            min_value=1,
            max_value=5,
            help="Initial performance guarantee period (included in CAPEX)"
        )
    
    st.markdown("#### Extended Coverage Options")
    st.info("Configure pricing for two extension scenarios: Performance Only vs Full Warranty")
    
    # Performance Guarantee Only Option
    st.markdown("##### Option 1: Performance Guarantee Only")
    col1, col2 = st.columns(2)
    
    with col1:
        perf_only_y1_7 = st.slider(
            "Years 1-7 Total Cost (% CAPEX)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Total cost for years 1-7 (paid over the period)"
        )
        
    with col2:
        perf_only_y8_12 = st.slider(
            "Years 8-12 Total Cost (% CAPEX)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Total cost for years 8-12 (paid over 5 years)"
        )
    
    # Full Warranty + Performance Guarantee Option
    st.markdown("##### Option 2: Full Warranty + Performance Guarantee")
    col1, col2 = st.columns(2)
    
    with col1:
        extended_warranty_y4_15 = st.slider(
            "Years 4-15 (% CAPEX/year)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            format="%.1f",
            help="Annual payment for extended warranty"
        )
        
    with col2:
        extended_warranty_y16_20 = st.slider(
            "Years 16-20 (% CAPEX/year)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Annual payment for final years"
        )

with tab5:
    st.markdown("#### Failure Probability Modeling")
    col1, col2 = st.columns(2)
    
    with col1:
        module_failure_rate = st.slider(
            "Annual Module Failure Rate (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            format="%.1f"
        )
        
        serial_defect_prob = st.slider(
            "Serial Defect Probability (%)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help=">5% modules affected triggers serial defect classification"
        )
        
    with col2:
        avg_repair_time = st.slider(
            "Average Repair Time (days)",
            min_value=1,
            max_value=30,
            value=7
        )
        
        availability_target = st.slider(
            "Availability Target (%)",
            min_value=90,
            max_value=99,
            value=97
        )

# Monte Carlo Simulation Class - Modified for three scenarios
class BESSMonteCarloSimulation:
    def __init__(self, params):
        self.params = params
        self.results = {}
        
    def calculate_degradation_path(self, scenario_type="Guaranteed"):
        """Calculate capacity degradation over project lifetime"""
        years = np.arange(0, self.params['project_lifetime'] + 1)
        
        # Adjust degradation based on cycling intensity
        cycles_per_day = self.params.get('cycles_per_day', 1.0)
        cycle_stress_factor = 1.0
        if cycles_per_day > 1.0:
            # Higher cycling accelerates degradation
            cycle_stress_factor = 1.0 + (cycles_per_day - 1.0) * 0.15
        elif cycles_per_day < 1.0:
            # Lower cycling reduces degradation
            cycle_stress_factor = 0.9 + cycles_per_day * 0.1
        
        if scenario_type == "Guaranteed":
            # Use typical warranty values for first 3 years, adjusted for cycling
            guaranteed_values = [100, 94.40, 91.11, 88.97]
            
            # Adjust guaranteed values based on cycling pattern if different from contract assumption
            if abs(cycles_per_day - 1.0) > 0.1:
                for i in range(1, len(guaranteed_values)):
                    degradation = 100 - guaranteed_values[i]
                    adjusted_degradation = degradation * cycle_stress_factor
                    guaranteed_values[i] = 100 - adjusted_degradation
            
            if len(years) <= 3:
                return guaranteed_values[:len(years)]
            
            capacity_retention = np.zeros(len(years))
            capacity_retention[:4] = guaranteed_values
            
            for i in range(4, len(years)):
                annual_deg = self.params['annual_degradation'] * cycle_stress_factor
                annual_deg *= (1 + np.random.normal(0, self.params['degradation_uncertainty']/100))
                capacity_retention[i] = capacity_retention[i-1] * (1 - annual_deg/100)
        else:
            capacity_retention = np.zeros(len(years))
            capacity_retention[0] = 100
            
            for i in range(1, len(years)):
                annual_deg = self.params['annual_degradation'] * cycle_stress_factor
                annual_deg *= (1 + np.random.normal(0, self.params['degradation_uncertainty']/100))
                capacity_retention[i] = capacity_retention[i-1] * (1 - annual_deg/100)
                
        return capacity_retention
    
    def calculate_revenue(self, year, capacity_retention, cycles):
        """Calculate annual revenue based on capacity and market conditions"""
        effective_capacity = self.params['initial_capacity'] * (capacity_retention/100)
        
        # Base price spread with annual growth
        base_spread = self.params['avg_price_spread'] * (1 + self.params['price_growth_rate']/100)**year
        
        # Adjust price spread based on cycling pattern
        cycles_per_day = self.params.get('cycles_per_day', 1.0)
        if cycles_per_day <= 1.0:
            spread_efficiency = 1.0
        elif cycles_per_day <= 1.5:
            spread_efficiency = 0.90
        elif cycles_per_day <= 2.0:
            spread_efficiency = 0.85
        else:
            spread_efficiency = 0.85 - (cycles_per_day - 2.0) * 0.05
            spread_efficiency = max(spread_efficiency, 0.70)
        
        # Apply volatility and efficiency adjustment
        price_spread = base_spread * spread_efficiency
        price_spread *= (1 + np.random.normal(0, self.params['price_volatility']/100))
        
        energy_revenue = (
            effective_capacity * 
            cycles * 
            price_spread * 
            (self.params['roundtrip_efficiency']/100)
        )
        
        # Capacity payment based on available MWh at point of connection
        soh = capacity_retention / 100  # State of Health as fraction
        rte_for_capacity = self.params.get('rte_for_capacity', 86.0) / 100
        available_capacity = self.params['initial_capacity'] * soh * rte_for_capacity
        capacity_revenue = (available_capacity / 4) * self.params['capacity_payment_rate']
        
        ancillary_revenue = effective_capacity * cycles * self.params['ancillary_revenue']
        
        return energy_revenue + capacity_revenue + ancillary_revenue
    
    def calculate_opex(self, year, warranty_type):
        """Calculate operating expenses based on warranty coverage"""
        base_opex = self.params['initial_capex'] * 1e6 * 0.015
        
        # Increase opex if no warranty coverage
        if warranty_type == "no_extension" and year >= self.params['base_warranty_years']:
            base_opex *= 1.5  # 50% increase in O&M without warranty
        elif warranty_type == "performance_only" and year >= self.params['base_warranty_years']:
            base_opex *= 1.25  # 25% increase with performance guarantee only
            
        return base_opex
    
    def calculate_warranty_cost(self, year, warranty_type):
        """Calculate warranty cost for given year and warranty type"""
        if year < self.params['base_warranty_years']:
            return 0  # Base warranty included in CAPEX
        
        if warranty_type == "no_extension":
            return 0  # No extended warranty
            
        elif warranty_type == "performance_only":
            # Performance guarantee only - total payment spread over period
            if year >= self.params['base_warranty_years'] and year < 7:
                # Spread the cost over years after base warranty to year 7
                years_in_period = 7 - self.params['base_warranty_years']
                annual_cost = (self.params['initial_capex'] * 1e6 * (self.params['perf_only_y1_7']/100)) / years_in_period
                return annual_cost
            elif year >= 7 and year < 12:
                # Years 8-12: spread over 5 years
                annual_cost = (self.params['initial_capex'] * 1e6 * (self.params['perf_only_y8_12']/100)) / 5
                return annual_cost
            else:
                return 0
                
        elif warranty_type == "full_warranty":
            # Full warranty + performance guarantee - annual payments
            if year < 15:
                return self.params['initial_capex'] * 1e6 * (self.params['extended_warranty_y4_15']/100)
            else:
                return self.params['initial_capex'] * 1e6 * (self.params['extended_warranty_y16_20']/100)
        
        return 0
    
    def simulate_single_scenario(self, warranty_type="full_warranty"):
        """Run single simulation scenario"""
        cash_flows = []
        capacity_path = self.calculate_degradation_path(self.params['degradation_scenario'])
        
        for year in range(self.params['project_lifetime']):
            actual_cycles = self.params['base_cycles_per_year'] * np.random.uniform(0.9, 1.1)
            
            # Simulate random failures
            has_failure = np.random.random() < (self.params['module_failure_rate']/100)
            if has_failure:
                downtime_factor = 1 - (self.params['avg_repair_time']/365)
                actual_cycles *= downtime_factor
                
                # Without warranty, failures have more impact
                if warranty_type == "no_extension" and year >= self.params['base_warranty_years']:
                    downtime_factor *= 0.8  # Additional 20% impact without warranty
            
            revenue = self.calculate_revenue(year, capacity_path[year], actual_cycles)
            opex = self.calculate_opex(year, warranty_type)
            warranty_cost = self.calculate_warranty_cost(year, warranty_type)
            
            net_cf = revenue - opex - warranty_cost
            cash_flows.append(net_cf)
            
        return np.array(cash_flows)
    
    def calculate_npv(self, cash_flows):
        """Calculate NPV of cash flows"""
        discount_factors = [(1 + self.params['discount_rate']/100)**(-i) for i in range(len(cash_flows))]
        return np.sum(cash_flows * discount_factors)
    
    def run_simulation(self, num_simulations=1000):
        """Run full Monte Carlo simulation for all three scenarios"""
        npv_no_extension = []
        npv_performance_only = []
        npv_full_warranty = []
        
        for _ in range(num_simulations):
            # Scenario 1: No extension
            cf_no_ext = self.simulate_single_scenario(warranty_type="no_extension")
            npv_no_extension.append(self.calculate_npv(cf_no_ext))
            
            # Scenario 2: Performance guarantee only
            cf_perf_only = self.simulate_single_scenario(warranty_type="performance_only")
            npv_performance_only.append(self.calculate_npv(cf_perf_only))
            
            # Scenario 3: Full warranty + performance guarantee
            cf_full = self.simulate_single_scenario(warranty_type="full_warranty")
            npv_full_warranty.append(self.calculate_npv(cf_full))
        
        self.results = {
            'npv_no_extension': np.array(npv_no_extension),
            'npv_performance_only': np.array(npv_performance_only),
            'npv_full_warranty': np.array(npv_full_warranty),
            'perf_only_value': np.array(npv_performance_only) - np.array(npv_no_extension),
            'full_warranty_value': np.array(npv_full_warranty) - np.array(npv_no_extension)
        }
        
        return self.results

# Run simulation when button clicked
if run_simulation:
    # Progress indicator
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text('Initializing simulation parameters...')
    progress_bar.progress(10)
    
    # Prepare parameters
    sim_params = {
        'initial_capacity': initial_capacity,
        'power_rating': power_rating,
        'roundtrip_efficiency': roundtrip_efficiency,
        'base_cycles_per_year': base_cycles_per_year,
        'cycles_per_day': cycles_per_day,
        'project_lifetime': project_lifetime,
        'discount_rate': discount_rate,
        'avg_price_spread': avg_price_spread,
        'price_volatility': price_volatility,
        'price_growth_rate': price_growth_rate,
        'capacity_payment_rate': capacity_payment_rate,
        'rte_for_capacity': rte_for_capacity,
        'ancillary_revenue': ancillary_revenue,
        'degradation_scenario': degradation_scenario,
        'annual_degradation': annual_degradation,
        'degradation_uncertainty': degradation_uncertainty,
        'initial_capex': initial_capex,
        'base_warranty_years': base_warranty_years,
        'perf_guarantee_years': perf_guarantee_years,
        'perf_only_y1_7': perf_only_y1_7,
        'perf_only_y8_12': perf_only_y8_12,
        'extended_warranty_y4_15': extended_warranty_y4_15,
        'extended_warranty_y16_20': extended_warranty_y16_20,
        'module_failure_rate': module_failure_rate,
        'serial_defect_prob': serial_defect_prob,
        'avg_repair_time': avg_repair_time,
        'augmentation_threshold': augmentation_threshold
    }
    
    progress_text.text(f'Running {num_simulations:,} Monte Carlo iterations for three scenarios...')
    progress_bar.progress(50)
    
    # Run simulation
    sim = BESSMonteCarloSimulation(sim_params)
    results = sim.run_simulation(num_simulations)
    
    progress_text.text('Analyzing results...')
    progress_bar.progress(90)
    
    # Clear progress indicators
    progress_text.empty()
    progress_bar.empty()
    
    # Display results
    st.markdown("---")
    st.markdown("## Three-Scenario Comparison Results")
    
    # Key metrics comparison
    st.markdown("### NPV Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔷 No Extension")
        mean_no_ext = np.mean(results['npv_no_extension']) / 1e6
        std_no_ext = np.std(results['npv_no_extension']) / 1e6
        st.metric("Expected NPV", f"${mean_no_ext:.2f}M")
        st.caption(f"Std Dev: ${std_no_ext:.2f}M")
        
    with col2:
        st.markdown("#### 🔶 Performance Only")
        mean_perf_only = np.mean(results['npv_performance_only']) / 1e6
        std_perf_only = np.std(results['npv_performance_only']) / 1e6
        value_vs_no_ext = mean_perf_only - mean_no_ext
        st.metric("Expected NPV", f"${mean_perf_only:.2f}M", f"${value_vs_no_ext:+.2f}M")
        st.caption(f"Std Dev: ${std_perf_only:.2f}M")
        
    with col3:
        st.markdown("#### 🟣 Full Warranty")
        mean_full = np.mean(results['npv_full_warranty']) / 1e6
        std_full = np.std(results['npv_full_warranty']) / 1e6
        value_vs_no_ext_full = mean_full - mean_no_ext
        st.metric("Expected NPV", f"${mean_full:.2f}M", f"${value_vs_no_ext_full:+.2f}M")
        st.caption(f"Std Dev: ${std_full:.2f}M")
    
    # Visualization - NPV Distribution Comparison
    st.markdown("### NPV Distribution Analysis")
    
    # Create combined histogram
    fig_combined = go.Figure()
    
    # Add traces for each scenario
    fig_combined.add_trace(go.Histogram(
        x=results['npv_no_extension']/1e6,
        name='No Extension',
        marker_color='#1E88E5',  # Blue
        opacity=0.6,
        nbinsx=30
    ))
    
    fig_combined.add_trace(go.Histogram(
        x=results['npv_performance_only']/1e6,
        name='Performance Only',
        marker_color='#FB8C00',  # Orange
        opacity=0.6,
        nbinsx=30
    ))
    
    fig_combined.add_trace(go.Histogram(
        x=results['npv_full_warranty']/1e6,
        name='Full Warranty',
        marker_color='#7B1FA2',  # Purple
        opacity=0.6,
        nbinsx=30
    ))
    
    # Add vertical lines for means
    fig_combined.add_vline(x=mean_no_ext, line_dash="dash", line_color="#1E88E5", 
                          annotation_text=f"No Ext: ${mean_no_ext:.1f}M", annotation_position="top")
    fig_combined.add_vline(x=mean_perf_only, line_dash="dash", line_color="#FB8C00",
                          annotation_text=f"Perf: ${mean_perf_only:.1f}M", annotation_position="bottom")
    fig_combined.add_vline(x=mean_full, line_dash="dash", line_color="#7B1FA2",
                          annotation_text=f"Full: ${mean_full:.1f}M", annotation_position="top")
    
    fig_combined.update_layout(
        title='NPV Distribution - Three Scenario Comparison',
        xaxis_title='NPV ($M)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        title_font_size=14,
        hovermode='x unified'
    )
    fig_combined.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig_combined.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Value comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Incremental value chart
        fig_value = go.Figure()
        fig_value.add_trace(go.Histogram(
            x=results['perf_only_value']/1e6,
            name='Performance Only vs No Extension',
            marker_color='#FB8C00',  # Orange
            opacity=0.75
        ))
        fig_value.add_trace(go.Histogram(
            x=results['full_warranty_value']/1e6,
            name='Full Warranty vs No Extension',
            marker_color='#7B1FA2',  # Purple
            opacity=0.75
        ))
        fig_value.add_vline(
            x=0,
            line_dash="dash",
            line_color="#ff0000",
            annotation_text="Break-even",
            annotation_position="top"
        )
        fig_value.update_layout(
            title='Incremental Value Distribution',
            xaxis_title='Additional Value ($M)',
            yaxis_title='Frequency',
            height=350,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            barmode='overlay'
        )
        st.plotly_chart(fig_value, use_container_width=True)
        
    with col2:
        # Box plot comparison
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=results['npv_no_extension']/1e6,
            name='No Extension',
            marker_color='#1E88E5'  # Blue
        ))
        fig_box.add_trace(go.Box(
            y=results['npv_performance_only']/1e6,
            name='Performance Only',
            marker_color='#FB8C00'  # Orange
        ))
        fig_box.add_trace(go.Box(
            y=results['npv_full_warranty']/1e6,
            name='Full Warranty',
            marker_color='#7B1FA2'  # Purple
        ))
        fig_box.update_layout(
            title='NPV Range Comparison',
            yaxis_title='NPV ($M)',
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### Detailed Statistical Analysis")
    
    # Calculate all statistics
    percentiles = [5, 25, 50, 75, 95]
    
    stats_data = {
        'Metric': ['Mean NPV', 'Std Deviation', 'Min', 'Max'] + [f'{p}th Percentile' for p in percentiles],
        'No Extension': [
            np.mean(results['npv_no_extension'])/1e6,
            np.std(results['npv_no_extension'])/1e6,
            np.min(results['npv_no_extension'])/1e6,
            np.max(results['npv_no_extension'])/1e6
        ] + [np.percentile(results['npv_no_extension'], p)/1e6 for p in percentiles],
        'Performance Only': [
            np.mean(results['npv_performance_only'])/1e6,
            np.std(results['npv_performance_only'])/1e6,
            np.min(results['npv_performance_only'])/1e6,
            np.max(results['npv_performance_only'])/1e6
        ] + [np.percentile(results['npv_performance_only'], p)/1e6 for p in percentiles],
        'Full Warranty': [
            np.mean(results['npv_full_warranty'])/1e6,
            np.std(results['npv_full_warranty'])/1e6,
            np.min(results['npv_full_warranty'])/1e6,
            np.max(results['npv_full_warranty'])/1e6
        ] + [np.percentile(results['npv_full_warranty'], p)/1e6 for p in percentiles]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    # Add formatting
    for col in ['No Extension', 'Performance Only', 'Full Warranty']:
        stats_df[col] = stats_df[col].apply(lambda x: f'${x:.2f}M')
    
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Probability analysis
    st.markdown("### Risk Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate probabilities
    prob_perf_beats_none = np.sum(results['perf_only_value'] > 0) / len(results['perf_only_value']) * 100
    prob_full_beats_none = np.sum(results['full_warranty_value'] > 0) / len(results['full_warranty_value']) * 100
    prob_full_beats_perf = np.sum(results['npv_full_warranty'] > results['npv_performance_only']) / len(results['npv_full_warranty']) * 100
    
    with col1:
        st.metric("P(Perf > No Ext)", f"{prob_perf_beats_none:.1f}%")
        st.caption("Probability that Performance Only beats No Extension")
        
    with col2:
        st.metric("P(Full > No Ext)", f"{prob_full_beats_none:.1f}%")
        st.caption("Probability that Full Warranty beats No Extension")
        
    with col3:
        st.metric("P(Full > Perf)", f"{prob_full_beats_perf:.1f}%")
        st.caption("Probability that Full Warranty beats Performance Only")
    
    # Cost-benefit analysis
    st.markdown("### Cost-Benefit Analysis")
    
    # Calculate warranty costs
    perf_only_cost = initial_capex * (perf_only_y1_7/100)
    if project_lifetime > 7:
        perf_only_cost += initial_capex * (perf_only_y8_12/100)
    
    full_warranty_cost = initial_capex * (extended_warranty_y4_15/100) * min(12, project_lifetime - base_warranty_years)
    if project_lifetime > 15:
        full_warranty_cost += initial_capex * (extended_warranty_y16_20/100) * (project_lifetime - 15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Performance Only")
        st.write(f"Total Warranty Cost: ${perf_only_cost:.2f}M")
        st.write(f"Expected Benefit: ${value_vs_no_ext:.2f}M")
        roi_perf = (value_vs_no_ext / perf_only_cost * 100) if perf_only_cost > 0 else 0
        st.write(f"**ROI: {roi_perf:.1f}%**")
        
    with col2:
        st.markdown("#### Full Warranty")
        st.write(f"Total Warranty Cost: ${full_warranty_cost:.2f}M")
        st.write(f"Expected Benefit: ${value_vs_no_ext_full:.2f}M")
        roi_full = (value_vs_no_ext_full / full_warranty_cost * 100) if full_warranty_cost > 0 else 0
        st.write(f"**ROI: {roi_full:.1f}%**")
    
    # Strategic recommendation
    st.markdown("---")
    st.markdown("### Strategic Recommendation")
    
    # Determine best option
    if mean_full > mean_perf_only and mean_full > mean_no_ext and prob_full_beats_none > 60:
        recommendation = "Full Warranty + Performance Guarantee"
        rationale = f"""
        **Recommendation: FULL WARRANTY + PERFORMANCE GUARANTEE**
        
        The comprehensive coverage provides the highest expected NPV of ${mean_full:.2f}M, 
        delivering ${value_vs_no_ext_full:.2f}M in additional value over self-insurance.
        
        - Probability of positive return: {prob_full_beats_none:.1f}%
        - ROI on warranty investment: {roi_full:.1f}%
        - Risk mitigation: Maximum coverage for both equipment failures and performance degradation
        """
        st.info(rationale)
    elif mean_perf_only > mean_no_ext and prob_perf_beats_none > 60:
        recommendation = "Performance Guarantee Only"
        rationale = f"""
        **Recommendation: PERFORMANCE GUARANTEE ONLY**
        
        The performance-only option provides optimal value with expected NPV of ${mean_perf_only:.2f}M,
        adding ${value_vs_no_ext:.2f}M over self-insurance at lower cost than full warranty.
        
        - Probability of positive return: {prob_perf_beats_none:.1f}%
        - ROI on warranty investment: {roi_perf:.1f}%
        - Cost efficiency: ${full_warranty_cost - perf_only_cost:.2f}M savings vs full warranty
        """
        st.info(rationale)
    else:
        recommendation = "No Extension (Self-Insure)"
        rationale = f"""
        **Recommendation: NO EXTENSION (SELF-INSURE)**
        
        Self-insurance provides the highest expected NPV of ${mean_no_ext:.2f}M.
        The warranty options do not provide sufficient value to justify their costs.
        
        - Potential savings: ${min(perf_only_cost, full_warranty_cost):.2f}M - ${max(perf_only_cost, full_warranty_cost):.2f}M
        - Establish internal reserve fund: ${perf_only_cost * 0.4:.2f}M recommended
        - Focus on preventive maintenance and monitoring systems
        """
        st.info(rationale)
    
    # Sensitivity note
    st.markdown("---")
    st.caption("""
    **Note:** Results are based on Monte Carlo simulation with stochastic modeling of degradation, 
    market prices, and failure events. Recommendations should be validated against specific contract 
    terms and risk tolerance. Consider running sensitivity analysis on key parameters.
    """)

# Footer
st.markdown("---")
st.caption("Monte Carlo simulation for battery energy storage system warranty valuation - Three scenario comparison")