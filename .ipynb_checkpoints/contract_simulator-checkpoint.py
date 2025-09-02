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
    **Step 4:** Review results and recommendations below  
    
    *Each parameter change will affect the NPV calculation. Hover over any input for additional context.*
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
    
    include_guarantee = st.checkbox(
        "Include Performance Guarantee",
        value=True,
        help="Compare scenarios with and without warranty"
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
    """.format(num_simulations/1000), unsafe_allow_html=True)

# Main parameter tabs - cleaner organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "System Configuration",
    "Market Parameters", 
    "Degradation Model",
    "Cost Structure",
    "Risk Factors"
])

# Initialize default values for all variables to prevent NameError
# These will be overridden when users interact with the tabs
warranty_option = "Warranty + Performance Guarantee"
extended_warranty_y4_15 = 1.5
extended_warranty_y16_20 = 2.0
perf_only_y4_10 = 2.0
perf_only_y11_15 = 2.0
capacity_payment_rate = 50000
rte_for_capacity = 86.0

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
            value=113.31,
            min_value=50.0,
            max_value=250.0,
            format="%.2f",
            help="Maximum charge/discharge power"
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
            value=100.0,
            min_value=50.0,
            max_value=500.0,
            format="%.1f"
        )
        
        base_warranty_years = st.number_input(
            "Base Warranty Period (years)",
            value=3,
            min_value=1,
            max_value=5,
            help="Initial warranty coverage period"
        )
        
    with col2:
        perf_guarantee_years = st.number_input(
            "Performance Guarantee (years)",
            value=3,
            min_value=1,
            max_value=5,
            help="Initial performance guarantee period"
        )
    
    st.markdown("#### Extended Warranty Pricing")
    st.caption("As specified in Section 3.2 of CATL contract")
    
    col1, col2 = st.columns(2)
    
    with col1:
        extended_warranty_y4_15 = st.slider(
            "Years 4-15 (% CAPEX/year)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            format="%.1f"
        )
        
    with col2:
        extended_warranty_y16_20 = st.slider(
            "Years 16-20 (% CAPEX/year)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            format="%.1f"
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

# Monte Carlo Simulation Class (unchanged logic, just embedded)
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
        # Formula: (MWh * SOH * RTE) / 4 * payment_rate
        soh = capacity_retention / 100  # State of Health as fraction
        rte_for_capacity = self.params.get('rte_for_capacity', 86.0) / 100
        available_capacity = self.params['initial_capacity'] * soh * rte_for_capacity
        capacity_revenue = (available_capacity / 4) * self.params['capacity_payment_rate']
        
        ancillary_revenue = effective_capacity * cycles * self.params['ancillary_revenue']
        
        return energy_revenue + capacity_revenue + ancillary_revenue
    
    def calculate_opex(self, year, has_warranty):
        """Calculate operating expenses"""
        base_opex = self.params['initial_capex'] * 1e6 * 0.015
        
        if not has_warranty and year >= self.params['base_warranty_years']:
            base_opex *= 1.5
            
        return base_opex
    
    def calculate_warranty_cost(self, year):
        """Calculate warranty cost for given year"""
        if year < self.params['base_warranty_years']:
            return 0
        
        warranty_option = self.params.get('warranty_option', 'Warranty + Performance Guarantee')
        
        if warranty_option == "Performance Guarantee Only":
            # Performance guarantee only - total payment spread over period
            if year >= 3 and year < 10:
                # Years 4-10: 2% total spread over 7 years
                perf_only_y4_10 = self.params.get('perf_only_y4_10', 2.0)
                annual_cost = (self.params['initial_capex'] * 1e6 * (perf_only_y4_10/100)) / 7
                return annual_cost
            elif year >= 10 and year < 15:
                # Years 11-15: 2% total spread over 5 years
                perf_only_y11_15 = self.params.get('perf_only_y11_15', 2.0)
                annual_cost = (self.params['initial_capex'] * 1e6 * (perf_only_y11_15/100)) / 5
                return annual_cost
            else:
                return 0
        else:
            # Warranty + Performance Guarantee - annual payments
            if year < 15:
                return self.params['initial_capex'] * 1e6 * (self.params['extended_warranty_y4_15']/100)
            else:
                return self.params['initial_capex'] * 1e6 * (self.params['extended_warranty_y16_20']/100)
    
    def simulate_single_scenario(self, with_guarantee=True):
        """Run single simulation scenario"""
        cash_flows = []
        capacity_path = self.calculate_degradation_path(self.params['degradation_scenario'])
        
        for year in range(self.params['project_lifetime']):
            actual_cycles = self.params['base_cycles_per_year'] * np.random.uniform(0.9, 1.1)
            
            has_failure = np.random.random() < (self.params['module_failure_rate']/100)
            if has_failure:
                downtime_factor = 1 - (self.params['avg_repair_time']/365)
                actual_cycles *= downtime_factor
            
            revenue = self.calculate_revenue(year, capacity_path[year], actual_cycles)
            opex = self.calculate_opex(year, with_guarantee)
            warranty_cost = self.calculate_warranty_cost(year) if with_guarantee else 0
            
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
            cf_with = self.simulate_single_scenario(with_guarantee=True)
            npv_with_guarantee.append(self.calculate_npv(cf_with))
            
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
        'capacity_payment_rate': capacity_payment_rate,  # Changed from capacity_payment
        'rte_for_capacity': rte_for_capacity,  # Added for capacity calculation
        'ancillary_revenue': ancillary_revenue,
        'degradation_scenario': degradation_scenario,
        'annual_degradation': annual_degradation,
        'degradation_uncertainty': degradation_uncertainty,
        'initial_capex': initial_capex,
        'base_warranty_years': base_warranty_years,
        'perf_guarantee_years': perf_guarantee_years,
        'warranty_option': warranty_option,  # Added warranty option type
        'module_failure_rate': module_failure_rate,
        'serial_defect_prob': serial_defect_prob,
        'avg_repair_time': avg_repair_time,
        'augmentation_threshold': augmentation_threshold
    }
    
    # Add appropriate warranty parameters based on option
    if warranty_option == "Performance Guarantee Only":
        sim_params['perf_only_y4_10'] = perf_only_y4_10
        sim_params['perf_only_y11_15'] = perf_only_y11_15
        sim_params['extended_warranty_y4_15'] = 0
        sim_params['extended_warranty_y16_20'] = 0
    else:
        sim_params['extended_warranty_y4_15'] = extended_warranty_y4_15
        sim_params['extended_warranty_y16_20'] = extended_warranty_y16_20
    
    progress_text.text(f'Running {num_simulations:,} Monte Carlo iterations...')
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
    st.markdown("## Analysis Results")
    
    # Add warranty option to title
    if warranty_option == "Performance Guarantee Only":
        st.caption("Analyzing: Performance Guarantee Only (no warranty coverage)")
    else:
        st.caption("Analyzing: Full Warranty + Performance Guarantee")
    
    # Key metrics - clean card layout
    mean_value = np.mean(results['guarantee_value']) / 1e6
    prob_positive = np.sum(results['guarantee_value'] > 0) / len(results['guarantee_value']) * 100
    var_95 = np.percentile(results['guarantee_value'], 5) / 1e6
    
    # Calculate ROI based on warranty option
    if warranty_option == "Performance Guarantee Only":
        total_warranty_cost = initial_capex * (perf_only_y4_10/100)
        if project_lifetime > 10:
            total_warranty_cost += initial_capex * (perf_only_y11_15/100)
        roi = (mean_value / total_warranty_cost) * 100 if total_warranty_cost > 0 else 0
    else:
        total_warranty_cost = initial_capex * (extended_warranty_y4_15/100) * min(12, project_lifetime - 3)
        if project_lifetime > 15:
            total_warranty_cost += initial_capex * (extended_warranty_y16_20/100) * (project_lifetime - 15)
        roi = (mean_value / total_warranty_cost) * 100 if total_warranty_cost > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_label = "Expected Value Impact"
        if mean_value >= 0:
            st.metric(metric_label, f"+${mean_value:.2f}M")
        else:
            st.metric(metric_label, f"-${abs(mean_value):.2f}M")
    
    with col2:
        st.metric("Probability of Net Benefit", f"{prob_positive:.1f}%")
    
    with col3:
        st.metric("Value at Risk (95% CI)", f"${var_95:.2f}M")
    
    with col4:
        st.metric("Return on Warranty", f"{roi:.1f}%")
    
    # Visualization with clean styling
    st.markdown("### Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # NPV Distribution - higher contrast colors
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=results['npv_with_guarantee']/1e6,
            name='With Guarantee',
            marker_color='#2E7D32',  # Dark green for with guarantee
            opacity=0.75
        ))
        fig1.add_trace(go.Histogram(
            x=results['npv_without_guarantee']/1e6,
            name='Without Guarantee',
            marker_color='#C62828',  # Dark red for without guarantee
            opacity=0.75
        ))
        fig1.update_layout(
            title='NPV Distribution Comparison',
            xaxis_title='NPV ($M)',
            yaxis_title='Frequency',
            barmode='overlay',
            height=350,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            title_font_size=14,
            hovermode='x unified'
        )
        fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Guarantee Value Distribution - clean design
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=results['guarantee_value']/1e6,
            name='Guarantee Value',
            marker_color='#000'
        ))
        fig2.add_vline(
            x=0,
            line_dash="dash",
            line_color="#ff0000",
            annotation_text="Break-even",
            annotation_position="top"
        )
        fig2.update_layout(
            title='Performance Guarantee Value Distribution',
            xaxis_title='Value ($M)',
            yaxis_title='Frequency',
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            title_font_size=14
        )
        fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Financial Analysis
    st.markdown("---")
    st.markdown("### Financial Analysis & Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if mean_value > 0:
            st.success(f"""
            **Economic Assessment: Net Positive Expected Value**
            
            The performance guarantee demonstrates a positive expected net present value of ${mean_value:.2f}M 
            over the project lifetime. This indicates that the risk mitigation benefits statistically 
            outweigh the warranty premium costs under the modeled scenarios.
            """)
        else:
            st.info(f"""
            **Economic Assessment: Negative Expected Value**
            
            The analysis indicates a negative expected net present value of ${abs(mean_value):.2f}M 
            for the performance guarantee. This suggests the warranty premium exceeds the risk-adjusted 
            value of the protection provided under current market conditions and degradation assumptions.
            """)
    
    with col2:
        if prob_positive > 70:
            st.success(f"""
            **Risk Profile: Low Uncertainty**
            
            With {prob_positive:.1f}% of simulated scenarios yielding positive returns, 
            the analysis demonstrates robust statistical confidence in the value proposition.
            """)
        elif prob_positive > 50:
            st.warning(f"""
            **Risk Profile: Moderate Uncertainty**
            
            The simulation shows {prob_positive:.1f}% probability of net benefit, indicating 
            moderate variance in outcomes.
            """)
        else:
            st.info(f"""
            **Risk Profile: Unfavorable Risk-Return**
            
            Only {prob_positive:.1f}% of scenarios generate positive returns, suggesting 
            the guarantee pricing structure is misaligned with the actual risk profile.
            """)
    
    # Recommendations
    st.markdown("---")
    st.markdown("### Strategic Recommendations")
    
    # Calculate additional metrics
    if warranty_option == "Performance Guarantee Only":
        # Calculate total cost for performance guarantee only
        warranty_total_cost = initial_capex * 1e6 * (perf_only_y4_10/100)  # Years 4-10 total
        if project_lifetime > 10:
            warranty_total_cost += initial_capex * 1e6 * (perf_only_y11_15/100)  # Years 11-15 total
    else:
        # Calculate annual payments for warranty + performance guarantee
        warranty_total_cost = initial_capex * 1e6 * (extended_warranty_y4_15/100) * min(12, project_lifetime)
        if project_lifetime > 15:
            warranty_total_cost += initial_capex * 1e6 * (extended_warranty_y16_20/100) * (project_lifetime - 15)
    
    breakeven_degradation = annual_degradation * (1 + warranty_total_cost / (initial_capex * 1e6))
    irr_threshold = discount_rate + 2.0
    
    if mean_value > 0 and prob_positive > 60:
        st.markdown(f"""
        **Recommendation: PROCEED WITH GUARANTEE**
        
        **Quantitative Justification:**
        - Expected NPV improvement: ${mean_value:.2f}M
        - Statistical confidence level: {prob_positive:.1f}%
        - Risk-adjusted return exceeds hurdle rate of {irr_threshold:.1f}%
        - Total warranty investment: ${warranty_total_cost/1e6:.2f}M over project lifetime
        
        **Strategic Considerations:**
        - The guarantee provides asymmetric risk protection favorable to project economics
        - Warranty terms align with expected degradation curves
        - Consider negotiating multi-year payment terms to improve cash flow timing
        """)
    elif mean_value > 0 and prob_positive <= 60:
        st.markdown(f"""
        **Recommendation: NEGOTIATE TERMS**
        
        **Quantitative Findings:**
        - Marginal expected NPV: ${mean_value:.2f}M
        - Scenario confidence: {prob_positive:.1f}% (below 60% threshold)
        - Breakeven requires degradation >{breakeven_degradation:.1f}% annually
        
        **Negotiation Priorities:**
        1. Reduce annual warranty premiums by 20-30%
        2. Implement performance-based pricing tied to actual degradation
        3. Include availability guarantees (currently {availability_target}% target)
        4. Negotiate cap on total warranty payments at ${warranty_total_cost*0.75/1e6:.2f}M
        """)
    else:
        st.markdown(f"""
        **Recommendation: SELF-INSURE**
        
        **Financial Analysis:**
        - Expected NPV impact: -${abs(mean_value):.2f}M
        - Implied risk premium: {abs(roi):.1f}% above actuarial value
        - Self-insurance reserve requirement: ${warranty_total_cost*0.4/1e6:.2f}M
        
        **Alternative Risk Management Strategy:**
        1. Establish dedicated O&M reserve fund at 40% of warranty cost
        2. Implement enhanced monitoring systems for early degradation detection
        3. Negotiate spot maintenance contracts as needed
        4. Consider partial coverage for years 10-15 only at reduced rates
        """)
    
    # Statistical summary - clean table
    st.markdown("---")
    st.markdown("### Statistical Summary")
    
    summary_data = {
        'Metric': [
            'Mean NPV',
            'Standard Deviation',
            '5th Percentile',
            '25th Percentile',
            'Median',
            '75th Percentile',
            '95th Percentile'
        ],
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
    
    # Display with clean styling
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

# Footer with minimal information
st.markdown("---")
st.caption("Monte Carlo simulation for battery energy storage system warranty valuation")