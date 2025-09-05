import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import time
from datetime import datetime, timedelta

# Set page config with minimalist theme
st.set_page_config(
    page_title="Battery Warranty Analysis Tool", 
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
st.markdown("# Battery Warranty Extension Analysis")
st.markdown("##### Monte Carlo Simulation for Warranty Decision Support")

# Instructions card at the top
with st.container():
    st.markdown("---")
    st.markdown("""
    ### Quick Start Guide
    
    **Step 1:** Configure your battery system parameters and operational profile  
    **Step 2:** Input actual performance data if available (optional but recommended)  
    **Step 3:** Set warranty pricing and O&M cost assumptions  
    **Step 4:** Click 'Run Simulation' to compare three scenarios  
    
    *The tool compares NPV across: No Extension | Performance Only | Full Warranty*
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
    """.format(num_simulations/1000 * 3), unsafe_allow_html=True)

# Main parameter tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "System Configuration",
    "Market Parameters", 
    "Degradation Model",
    "Cost Structure",
    "Risk Factors",
    "Performance Validation"
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
            help="Nameplate energy capacity of the battery system"
        )
        
    with col2:
        power_rating = st.number_input(
            "Power Rating (MW)",
            value=100.0,
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
    
    st.markdown("#### Operating Profile & Timeline")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Cycles per day selector
        cycle_options = [0.5, 0.75, 1.0, 1.15, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, "Custom"]
        cycles_selection = st.selectbox(
            "Daily Cycling Pattern",
            options=cycle_options,
            index=3,  # Default to 1.15 cycles per day
            format_func=lambda x: f"{x} cycle{'s' if x != 1 else ''}/day" if x != "Custom" else "Custom",
            help="Number of complete charge-discharge cycles per day"
        )
        
        if cycles_selection == "Custom":
            cycles_per_day = st.number_input(
                "Enter custom cycles/day",
                min_value=0.1,
                max_value=5.0,
                value=1.15,
                step=0.1,
                format="%.2f"
            )
        else:
            cycles_per_day = cycles_selection
        
        base_cycles_per_year = cycles_per_day * 365
        st.caption(f"→ {base_cycles_per_year:.0f} cycles/year")
        
    with col2:
        # Operations start date
        operations_start = st.date_input(
            "Commercial Operations Date",
            value=datetime(2023, 9, 1),
            help="When the battery started commercial operations"
        )
        
        # Extension decision date (Year 0 for warranty extension)
        extension_decision = st.date_input(
            "Warranty Extension Start Date",
            value=datetime(2025, 10, 15),
            help="Year 0 for warranty extension decision"
        )
        
    with col3:
        # Calculate remaining project life
        years_operated = (extension_decision - operations_start).days / 365.25
        st.info(f"Years operated: {years_operated:.1f}")
        
        total_project_life = st.number_input(
            "Total Project Lifetime (years)",
            value=15,
            min_value=10,
            max_value=20,
            help="Total expected lifetime from commissioning"
        )
        
        remaining_project_life = int(total_project_life - years_operated)
        st.success(f"Remaining lifetime: {remaining_project_life} years")
    
    st.markdown("#### Warranty Timing Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        warranty_start_offset = st.number_input(
            "Initial Warranty Start Before Operations (months)",
            value=14,
            min_value=0,
            max_value=24,
            help="Months between warranty commencement and commercial operations"
        )
        
        base_warranty_years = st.number_input(
            "Initial Warranty Period (years)",
            value=3,
            min_value=1,
            max_value=5,
            help="Length of initial warranty included in CAPEX"
        )
        
    with col2:
        st.markdown("##### Coverage Impact")
        warranty_consumed = warranty_start_offset / 12
        warranty_remaining = max(0, base_warranty_years - years_operated - warranty_consumed)
        
        if warranty_remaining > 0:
            st.warning(f"⚠️ {warranty_remaining:.1f} years of base warranty remaining")
            st.caption("Extension would start after base warranty expires")
        else:
            st.success(f"✓ Base warranty expired {abs(warranty_remaining):.1f} years ago")
            st.caption("Extension would start immediately if selected")
    
    # Discount rate
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
    
    st.markdown("#### Capacity Payment Configuration")
    
    # Display the engineer's formula
    st.info("""
    **Capacity Payment Formula:**
    
    `Annual Capacity Payment ($) = (Available MWh at POC / 4) × $/MWh Rate`
    
    Where: `Available MWh at POC = Nameplate MWh × SOH × RTE`
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        capacity_payment_rate = st.number_input(
            "Capacity Payment Rate ($/MWh)",
            value=50000,
            min_value=0,
            max_value=200000,
            step=5000,
            format="%d",
            help="Annual payment per MWh of available capacity (after division by 4)"
        )
        
    with col2:
        rte_for_capacity = st.number_input(
            "RTE for Capacity (%)",
            value=86.0,
            min_value=80.0,
            max_value=95.0,
            format="%.1f",
            help="Round-trip efficiency at grid connection point (POC)"
        )
        
    with col3:
        # Show example calculation at 100% SOH
        example_capacity = initial_capacity * 1.0 * (rte_for_capacity/100) / 4
        example_payment = example_capacity * capacity_payment_rate / 1e6
        st.metric(
            "Example at 100% SOH",
            f"${example_payment:.2f}M/year",
            help=f"{initial_capacity:.2f} MWh × 100% × {rte_for_capacity:.0f}% / 4 × ${capacity_payment_rate:,}"
        )
    
    st.markdown("#### Additional Revenue Streams")
    col1, col2 = st.columns(2)
    
    with col1:
        ancillary_revenue = st.number_input(
            "Ancillary Services ($/MWh)",
            value=5.0,
            min_value=0.0,
            max_value=20.0,
            format="%.1f"
        )
    
    # Variable O&M Recovery
    st.markdown("#### Variable O&M Recovery Mechanism")
    enable_vom_recovery = st.checkbox(
        "Enable Variable O&M Recovery",
        value=False,
        help="Recover accelerated degradation costs through market mechanisms for high cycling"
    )
    
    if enable_vom_recovery:
        col1, col2 = st.columns(2)
        with col1:
            vom_recovery_threshold = st.slider(
                "Cycling threshold for VO&M recovery",
                min_value=1.0,
                max_value=2.0,
                value=1.5,
                step=0.1,
                format="%.1f",
                help="Above this cycling rate, VO&M mechanism activates"
            )
        with col2:
            vom_recovery_rate = st.number_input(
                "VO&M Recovery Rate ($/MWh)",
                value=10.0,
                min_value=0.0,
                max_value=50.0,
                format="%.1f",
                help="Additional revenue per MWh when cycling above threshold"
            )

with tab3:
    st.markdown("#### Degradation Modeling")
    
    # Performance data input
    use_actual_data = st.checkbox(
        "Use Actual Performance Data",
        value=False,
        help="Input actual capacity test results for calibration"
    )
    
    if use_actual_data:
        st.markdown("##### Actual Performance Input")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_date = st.date_input(
                "Latest Capacity Test Date",
                value=datetime(2025, 6, 3)
            )
            months_operated = (test_date - operations_start).days / 30.44
            st.caption(f"After {months_operated:.1f} months of operation")
            
        with col2:
            current_soh = st.number_input(
                "Current SOH (%)",
                value=95.9,
                min_value=70.0,
                max_value=100.0,
                format="%.1f",
                help="Measured State of Health from capacity test"
            )
            current_capacity = initial_capacity * (current_soh/100)
            st.caption(f"Current: {current_capacity:.2f} MWh")
            
        with col3:
            guaranteed_soh = st.number_input(
                "Guaranteed SOH (%)",
                value=91.9,
                min_value=70.0,
                max_value=100.0,
                format="%.1f",
                help="What warranty would require at this point"
            )
            outperformance = current_soh - guaranteed_soh
            
            if outperformance > 0:
                st.success(f"Outperforming by {outperformance:.1f}%")
            else:
                st.warning(f"Underperforming by {abs(outperformance):.1f}%")
        
        # Calculate implied degradation rate
        if months_operated > 0:
            implied_annual_degradation = (100 - current_soh) / (months_operated/12)
            st.info(f"📊 Implied annual degradation rate: {implied_annual_degradation:.2f}%/year")
    
    # Degradation scenario selection
    st.markdown("##### Degradation Scenario")
    col1, col2 = st.columns(2)
    
    with col1:
        if use_actual_data and months_operated > 0:
            degradation_options = ["Actual (Calibrated)", "Warranty Baseline", "Optimistic (-20%)", "Pessimistic (+20%)", "Custom"]
            default_index = 0
        else:
            degradation_options = ["Warranty Baseline", "Optimistic (-20%)", "Pessimistic (+20%)", "Custom"]
            default_index = 0
            
        degradation_scenario = st.selectbox(
            "Select Degradation Scenario",
            options=degradation_options,
            index=default_index
        )
        
        if degradation_scenario == "Custom":
            annual_degradation = st.slider(
                "Annual Degradation Rate (%)",
                min_value=1.0,
                max_value=5.0,
                value=2.5,
                format="%.1f"
            )
        elif degradation_scenario == "Actual (Calibrated)":
            annual_degradation = implied_annual_degradation
            st.success(f"Using measured rate: {annual_degradation:.2f}%/year")
        elif degradation_scenario == "Optimistic (-20%)":
            annual_degradation = 2.0
        elif degradation_scenario == "Pessimistic (+20%)":
            annual_degradation = 3.0
        else:  # Warranty Baseline
            annual_degradation = 2.5
            
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
            help="SOH level triggering augmentation"
        )

with tab4:
    st.markdown("#### Capital Costs")
    initial_capex = st.number_input(
        "Initial CAPEX ($M)",
        value=55.0,
        min_value=20.0,
        max_value=500.0,
        format="%.1f",
        help="Total capital expenditure for battery system"
    )
    
    st.markdown("#### Extended Warranty Pricing")
    st.info("Configure pricing for warranty extension options starting from Year 0 (extension decision date)")
    
    # Performance Guarantee Only
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
            help="Total cost for first 7 years of extension"
        )
        
    with col2:
        perf_only_y8_12 = st.slider(
            "Years 8-12 Total Cost (% CAPEX)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Total cost for years 8-12 of extension"
        )
    
    # Full Warranty + Performance Guarantee
    st.markdown("##### Option 2: Full Warranty + Performance Guarantee")
    col1, col2 = st.columns(2)
    
    with col1:
        extended_warranty_y1_12 = st.slider(
            "Years 1-12 (% CAPEX/year)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            format="%.1f",
            help="Annual payment for first 12 years"
        )
        
    with col2:
        extended_warranty_y13_plus = st.slider(
            "Years 13+ (% CAPEX/year)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            format="%.1f",
            help="Annual payment for remaining years"
        )
    
    # O&M Cost Configuration
    st.markdown("#### O&M Cost Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        opex_mode = st.selectbox(
            "O&M Cost Basis",
            ["Scenario-Based", "Percentage of CAPEX", "Custom Annual"],
            help="Choose how to model O&M costs"
        )
        
        if opex_mode == "Scenario-Based":
            scenario_option = st.selectbox(
                "Select O&M Scenario",
                ["Low - Minimal Failures", "Medium - Normal Operations", "High - Frequent Failures"],
                index=1
            )
            
            scenario_costs = {
                "Low - Minimal Failures": 40000,
                "Medium - Normal Operations": 150000,
                "High - Frequent Failures": 360000
            }
            base_opex_annual = scenario_costs[scenario_option]
            opex_percentage = (base_opex_annual / (initial_capex * 1e6)) * 100
            st.info(f"Annual O&M: ${base_opex_annual:,} ({opex_percentage:.3f}% of CAPEX)")
            
        elif opex_mode == "Percentage of CAPEX":
            opex_percentage = st.slider(
                "Base O&M (% CAPEX/year)",
                min_value=0.05,
                max_value=2.0,
                value=0.3,
                step=0.05,
                format="%.2f"
            )
            base_opex_annual = initial_capex * 1e6 * (opex_percentage/100)
            st.info(f"Annual O&M: ${base_opex_annual:,.0f}")
            
        else:  # Custom Annual
            base_opex_annual = st.number_input(
                "Annual O&M Cost ($)",
                value=150000,
                min_value=10000,
                max_value=1000000,
                step=10000,
                format="%d"
            )
            opex_percentage = (base_opex_annual / (initial_capex * 1e6)) * 100
            st.info(f"Equivalent to {opex_percentage:.3f}% of CAPEX")
    
    with col2:
        st.markdown("##### O&M Without Coverage")
        opex_multiplier_no_warranty = st.slider(
            "Multiplier - No Warranty",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            format="%.1f",
            help="O&M cost increase without any warranty"
        )
        
        opex_multiplier_perf_only = st.slider(
            "Multiplier - Performance Only",
            min_value=1.0,
            max_value=2.0,
            value=1.25,
            step=0.05,
            format="%.2f",
            help="O&M cost increase with performance guarantee only"
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
            help=">5% modules affected = serial defect"
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
    
    st.markdown("#### Capacity Payment Impact Model")
    capacity_impact_mode = st.selectbox(
        "How does degradation affect capacity payments?",
        ["Minimal Impact (Step Function)", "Linear with SOH", "Conservative (Accelerated)"],
        help="Model how battery degradation impacts capacity payments"
    )

with tab6:
    st.markdown("### Performance Validation & Benchmarking")
    
    if use_actual_data:
        st.success("✓ Using actual performance data for calibration")
        
        # Performance comparison
        st.markdown("#### Performance vs Warranty Requirements")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Actual Performance")
            st.metric("Current SOH", f"{current_soh:.1f}%")
            st.metric("Degradation Rate", f"{implied_annual_degradation:.2f}%/year")
            
        with col2:
            st.markdown("##### Warranty Requirement")
            st.metric("Required SOH", f"{guaranteed_soh:.1f}%")
            st.metric("Max Allowed Degradation", f"{(100-guaranteed_soh)/(months_operated/12):.2f}%/year")
            
        with col3:
            st.markdown("##### Performance Delta")
            st.metric("SOH Advantage", f"+{outperformance:.1f}%", "↑" if outperformance > 0 else "↓")
            buffer_years = outperformance / annual_degradation if annual_degradation > 0 else 0
            st.metric("Time Buffer", f"{buffer_years:.1f} years", "↑" if buffer_years > 0 else "↓")
    else:
        st.info("Enable 'Use Actual Performance Data' in the Degradation Model tab to input capacity test results")
    
    # Cost-benefit preview
    st.markdown("#### Warranty Cost Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Performance Guarantee Only")
        perf_total = initial_capex * (perf_only_y1_7/100)
        if remaining_project_life > 7:
            perf_total += initial_capex * (perf_only_y8_12/100)
        st.metric("Total Cost", f"${perf_total:.2f}M")
        st.caption(f"Over {min(remaining_project_life, 12)} years")
        
    with col2:
        st.markdown("##### Full Warranty")
        full_total = initial_capex * (extended_warranty_y1_12/100) * min(12, remaining_project_life)
        if remaining_project_life > 12:
            full_total += initial_capex * (extended_warranty_y13_plus/100) * (remaining_project_life - 12)
        st.metric("Total Cost", f"${full_total:.2f}M")
        st.caption(f"Over {remaining_project_life} years")
    
    # Key assumptions summary
    st.markdown("#### Key Modeling Assumptions")
    assumptions_df = pd.DataFrame({
        'Parameter': ['Remaining Project Life', 'Annual Degradation', 'Base O&M Cost', 'Cycling Rate', 'Discount Rate'],
        'Value': [f'{remaining_project_life} years', f'{annual_degradation:.2f}%', f'${base_opex_annual:,}', 
                 f'{cycles_per_day:.2f} cycles/day', f'{discount_rate:.1f}%']
    })
    st.dataframe(assumptions_df, use_container_width=True, hide_index=True)

# Monte Carlo Simulation Class
class BESSMonteCarloSimulation:
    def __init__(self, params):
        self.params = params
        self.results = {}
        
    def calculate_degradation_path(self):
        """Calculate capacity degradation over remaining project lifetime"""
        years = np.arange(0, self.params['remaining_project_life'] + 1)
        capacity_retention = np.zeros(len(years))
        
        # Start from current SOH if provided
        if self.params.get('use_actual_data', False):
            capacity_retention[0] = self.params.get('current_soh', 100)
        else:
            # Estimate current SOH based on years operated
            years_operated = self.params.get('years_operated', 0)
            capacity_retention[0] = 100 - (self.params['annual_degradation'] * years_operated)
        
        # Project forward
        cycles_per_day = self.params.get('cycles_per_day', 1.0)
        cycle_stress_factor = 1.0
        if cycles_per_day > 1.0:
            cycle_stress_factor = 1.0 + (cycles_per_day - 1.0) * 0.15
        elif cycles_per_day < 1.0:
            cycle_stress_factor = 0.9 + cycles_per_day * 0.1
        
        for i in range(1, len(years)):
            annual_deg = self.params['annual_degradation'] * cycle_stress_factor
            annual_deg *= (1 + np.random.normal(0, self.params['degradation_uncertainty']/100))
            capacity_retention[i] = capacity_retention[i-1] * (1 - annual_deg/100)
            
        return capacity_retention
    
    def calculate_revenue(self, year, capacity_retention, cycles):
        """Calculate annual revenue with CORRECTED capacity payment formula"""
        effective_capacity = self.params['initial_capacity'] * (capacity_retention/100)
        
        # Energy arbitrage revenue
        base_spread = self.params['avg_price_spread'] * (1 + self.params['price_growth_rate']/100)**year
        price_spread = base_spread * (1 + np.random.normal(0, self.params['price_volatility']/100))
        
        energy_revenue = (
            effective_capacity * 
            cycles * 
            price_spread * 
            (self.params['roundtrip_efficiency']/100)
        )
        
        # CORRECTED CAPACITY PAYMENT CALCULATION
        # Engineer's formula: (Available MWh at POC / 4) × Rate
        soh = capacity_retention / 100
        
        # Calculate available MWh at point of connection (POC)
        available_mwh_at_poc = self.params['initial_capacity'] * soh * (self.params['rte_for_capacity']/100)
        
        # Apply engineer's formula: divide by 4, then multiply by rate
        capacity_revenue = (available_mwh_at_poc / 4) * self.params['capacity_payment_rate']
        
        # Apply capacity impact model (optional degradation factor)
        if self.params.get('capacity_impact_mode') == "Minimal Impact (Step Function)":
            capacity_factor = 1.0 if soh > 0.7 else 0.95
        elif self.params.get('capacity_impact_mode') == "Conservative (Accelerated)":
            capacity_factor = soh ** 1.5
        else:  # Linear with SOH
            capacity_factor = soh
        
        capacity_revenue *= capacity_factor
        
        # Ancillary services
        ancillary_revenue = effective_capacity * cycles * self.params['ancillary_revenue']
        
        # Variable O&M recovery if enabled
        if self.params.get('enable_vom_recovery', False):
            cycles_per_day = self.params.get('cycles_per_day', 1.0)
            vom_threshold = self.params.get('vom_recovery_threshold', 1.5)
            if cycles_per_day > vom_threshold:
                vom_recovery_rate = self.params.get('vom_recovery_rate', 10)
                vom_recovery = (cycles_per_day - vom_threshold) * effective_capacity * vom_recovery_rate
                energy_revenue += vom_recovery
        
        return energy_revenue + capacity_revenue + ancillary_revenue
    
    def calculate_opex(self, year, warranty_type):
        """Calculate operating expenses"""
        base_opex = self.params.get('base_opex_annual', 150000)
        
        if warranty_type == "no_extension":
            base_opex *= self.params.get('opex_multiplier_no_warranty', 1.5)
        elif warranty_type == "performance_only":
            base_opex *= self.params.get('opex_multiplier_perf_only', 1.25)
            
        return base_opex
    
    def calculate_warranty_cost(self, year, warranty_type):
        """Calculate warranty cost for given year"""
        if warranty_type == "no_extension":
            return 0
            
        elif warranty_type == "performance_only":
            if year < 7:
                # Years 1-7: spread total cost
                annual_cost = (self.params['initial_capex'] * 1e6 * (self.params['perf_only_y1_7']/100)) / 7
                return annual_cost
            elif year < 12:
                # Years 8-12: spread total cost
                annual_cost = (self.params['initial_capex'] * 1e6 * (self.params['perf_only_y8_12']/100)) / 5
                return annual_cost
            else:
                return 0
                
        elif warranty_type == "full_warranty":
            if year < 12:
                return self.params['initial_capex'] * 1e6 * (self.params['extended_warranty_y1_12']/100)
            else:
                return self.params['initial_capex'] * 1e6 * (self.params['extended_warranty_y13_plus']/100)
        
        return 0
    
    def simulate_single_scenario(self, warranty_type="full_warranty"):
        """Run single simulation scenario"""
        cash_flows = []
        capacity_path = self.calculate_degradation_path()
        
        for year in range(self.params['remaining_project_life']):
            actual_cycles = self.params['base_cycles_per_year'] * np.random.uniform(0.9, 1.1)
            
            # Simulate failures
            has_failure = np.random.random() < (self.params['module_failure_rate']/100)
            if has_failure:
                downtime_factor = 1 - (self.params['avg_repair_time']/365)
                actual_cycles *= downtime_factor
                
                if warranty_type == "no_extension":
                    downtime_factor *= 0.8  # Additional impact without warranty
            
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
        """Run full Monte Carlo simulation"""
        npv_no_extension = []
        npv_performance_only = []
        npv_full_warranty = []
        
        for _ in range(num_simulations):
            cf_no_ext = self.simulate_single_scenario(warranty_type="no_extension")
            npv_no_extension.append(self.calculate_npv(cf_no_ext))
            
            cf_perf_only = self.simulate_single_scenario(warranty_type="performance_only")
            npv_performance_only.append(self.calculate_npv(cf_perf_only))
            
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
        'remaining_project_life': remaining_project_life,
        'years_operated': years_operated,
        'discount_rate': discount_rate,
        'avg_price_spread': avg_price_spread,
        'price_volatility': price_volatility,
        'price_growth_rate': price_growth_rate,
        'capacity_payment_rate': capacity_payment_rate,
        'rte_for_capacity': rte_for_capacity,
        'ancillary_revenue': ancillary_revenue,
        'annual_degradation': annual_degradation,
        'degradation_uncertainty': degradation_uncertainty,
        'initial_capex': initial_capex,
        'perf_only_y1_7': perf_only_y1_7,
        'perf_only_y8_12': perf_only_y8_12,
        'extended_warranty_y1_12': extended_warranty_y1_12,
        'extended_warranty_y13_plus': extended_warranty_y13_plus,
        'module_failure_rate': module_failure_rate,
        'serial_defect_prob': serial_defect_prob,
        'avg_repair_time': avg_repair_time,
        'augmentation_threshold': augmentation_threshold,
        'base_opex_annual': base_opex_annual,
        'opex_multiplier_no_warranty': opex_multiplier_no_warranty,
        'opex_multiplier_perf_only': opex_multiplier_perf_only,
        'capacity_impact_mode': capacity_impact_mode,
        'enable_vom_recovery': enable_vom_recovery,
        'use_actual_data': use_actual_data,
    }
    
    # Add actual data parameters if enabled
    if use_actual_data:
        sim_params['current_soh'] = current_soh
        sim_params['guaranteed_soh'] = guaranteed_soh
    
    if enable_vom_recovery:
        sim_params['vom_recovery_threshold'] = vom_recovery_threshold
        sim_params['vom_recovery_rate'] = vom_recovery_rate
    
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
    st.markdown("## Simulation Results")
    
    # Key metrics comparison
    st.markdown("### NPV Comparison (Remaining Project Life: {} Years)".format(remaining_project_life))
    
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
    
    # NPV Distribution
    st.markdown("### NPV Distribution Analysis")
    
    fig_combined = go.Figure()
    
    fig_combined.add_trace(go.Histogram(
        x=results['npv_no_extension']/1e6,
        name='No Extension',
        marker_color='#1E88E5',
        opacity=0.6,
        nbinsx=30
    ))
    
    fig_combined.add_trace(go.Histogram(
        x=results['npv_performance_only']/1e6,
        name='Performance Only',
        marker_color='#FB8C00',
        opacity=0.6,
        nbinsx=30
    ))
    
    fig_combined.add_trace(go.Histogram(
        x=results['npv_full_warranty']/1e6,
        name='Full Warranty',
        marker_color='#7B1FA2',
        opacity=0.6,
        nbinsx=30
    ))
    
    fig_combined.add_vline(x=mean_no_ext, line_dash="dash", line_color="#1E88E5", 
                          annotation_text=f"No Ext: ${mean_no_ext:.1f}M")
    fig_combined.add_vline(x=mean_perf_only, line_dash="dash", line_color="#FB8C00",
                          annotation_text=f"Perf: ${mean_perf_only:.1f}M")
    fig_combined.add_vline(x=mean_full, line_dash="dash", line_color="#7B1FA2",
                          annotation_text=f"Full: ${mean_full:.1f}M")
    
    fig_combined.update_layout(
        title='NPV Distribution Comparison',
        xaxis_title='NPV ($M)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400,
        showlegend=True,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Probability analysis
    st.markdown("### Decision Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    prob_perf_beats_none = np.sum(results['perf_only_value'] > 0) / len(results['perf_only_value']) * 100
    prob_full_beats_none = np.sum(results['full_warranty_value'] > 0) / len(results['full_warranty_value']) * 100
    prob_full_beats_perf = np.sum(results['npv_full_warranty'] > results['npv_performance_only']) / len(results['npv_full_warranty']) * 100
    
    with col1:
        st.metric("P(Perf > No Ext)", f"{prob_perf_beats_none:.1f}%")
        st.caption("Probability Performance Only beats No Extension")
        
    with col2:
        st.metric("P(Full > No Ext)", f"{prob_full_beats_none:.1f}%")
        st.caption("Probability Full Warranty beats No Extension")
        
    with col3:
        st.metric("P(Full > Perf)", f"{prob_full_beats_perf:.1f}%")
        st.caption("Probability Full Warranty beats Performance Only")
    
    # ROI Analysis
    st.markdown("### Return on Investment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Performance Only")
        roi_perf = (value_vs_no_ext / perf_total * 100) if perf_total > 0 else 0
        st.metric("ROI", f"{roi_perf:.1f}%")
        st.write(f"Cost: ${perf_total:.2f}M | Benefit: ${value_vs_no_ext:.2f}M")
        
    with col2:
        st.markdown("#### Full Warranty")
        roi_full = (value_vs_no_ext_full / full_total * 100) if full_total > 0 else 0
        st.metric("ROI", f"{roi_full:.1f}%")
        st.write(f"Cost: ${full_total:.2f}M | Benefit: ${value_vs_no_ext_full:.2f}M")
    
    # Recommendation
    st.markdown("---")
    st.markdown("### Recommendation")
    
    if mean_no_ext >= mean_perf_only and mean_no_ext >= mean_full:
        st.info(f"""
        **Recommended Action: NO EXTENSION**
        
        Self-insurance provides the highest expected NPV of ${mean_no_ext:.2f}M.
        
        **Key Factors:**
        - Warranty costs exceed expected benefits
        - Establish reserve fund: ${perf_total * 0.4:.2f}M
        - Focus on preventive maintenance
        """)
    elif mean_full > mean_perf_only and prob_full_beats_none > 60:
        st.info(f"""
        **Recommended Action: FULL WARRANTY**
        
        Comprehensive coverage provides NPV of ${mean_full:.2f}M (+${value_vs_no_ext_full:.2f}M vs no extension).
        
        **Key Factors:**
        - {prob_full_beats_none:.0f}% probability of positive return
        - ROI: {roi_full:.1f}%
        - Maximum risk mitigation
        """)
    else:
        st.info(f"""
        **Recommended Action: PERFORMANCE GUARANTEE ONLY**
        
        Balanced option with NPV of ${mean_perf_only:.2f}M (+${value_vs_no_ext:.2f}M vs no extension).
        
        **Key Factors:**
        - {prob_perf_beats_none:.0f}% probability of positive return
        - ROI: {roi_perf:.1f}%
        - Cost savings: ${full_total - perf_total:.2f}M vs full warranty
        """)

# Footer
st.markdown("---")
st.caption("Battery Warranty Extension Analysis Tool - Monte Carlo Simulation")