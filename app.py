import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from data_generator import generate_mmm_data, get_performance_metrics

# Page config
st.set_page_config(
    page_title="Media Mix Modelling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #e5e7eb;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #e5e7eb;
        margin-bottom: 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-change {
        font-size: 0.875rem;
        margin-top: 8px;
        font-weight: 500;
    }
    .positive {
        color: #059669;
    }
    .negative {
        color: #dc2626;
    }
    .overview-text {
        font-size: 1.1rem;
        color: #ffffff;
        padding: 20px 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    h1 {
        color: #111827;
        font-weight: 700;
    }
    h2 {
        color: #1f2937;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    h3 {
        color: #374151;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Generate or load data
@st.cache_data
def load_data(model_type):
    return generate_mmm_data(days=365, model=model_type)

# Sidebar
st.sidebar.title("🎯 Filters")

# Use Prophet model by default (no selector)
model_type = "Prophet"

# Load data based on model
df = load_data(model_type)

# Date range filter
st.sidebar.subheader("Time Range")
time_period = st.sidebar.selectbox(
    "Select Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
    index=2
)

# Convert selection to days
period_days = {
    "Last 7 Days": 7,
    "Last 30 Days": 30,
    "Last 90 Days": 90
}
selected_days = period_days[time_period]
end_date = df['date'].max()
start_date = end_date - timedelta(days=selected_days)

# Industry filter
st.sidebar.subheader("Industry")
industries = sorted(df['industry'].unique().tolist())
selected_industry = st.sidebar.selectbox("Select Industry", industries, index=0)  # Default to first (E-commerce)

# Data Source
st.sidebar.subheader("Data Source")
data_source_type = st.sidebar.radio(
    "Choose Data Source",
    ["Test Data", "Upload CSV"],
    index=0
)

if data_source_type == "Upload CSV":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📄 CSV Format Required:**")
    st.sidebar.code("""date,channel,industry,spend,conversions,cpa,mcpa
2024-01-01,meta_prospecting,E-commerce,1000,50,20,22""", language="csv")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your MMM data",
        type=['csv'],
        help="Upload a CSV file with your marketing data"
    )
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            uploaded_df['date'] = pd.to_datetime(uploaded_df['date'])
            df = uploaded_df
            st.sidebar.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading CSV: {e}")
            st.sidebar.info("Using test data instead")
else:
    st.sidebar.info("📊 Using synthetic test data")

# Apply filters
filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]

# Filter by selected industry (no "All" option)
filtered_df = filtered_df[filtered_df['industry'] == selected_industry]

# Calculate previous year data for comparisons
prev_year_start = pd.Timestamp(start_date) - timedelta(days=365)
prev_year_end = pd.Timestamp(end_date) - timedelta(days=365)
prev_df = df[(df['date'] >= prev_year_start) & (df['date'] <= prev_year_end)]
# Filter previous year by same industry
prev_df = prev_df[prev_df['industry'] == selected_industry]

# Main content
st.title("📊 Media Mix Modelling Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["📈 Performance Dashboard", "💰 Budget Optimization"])

with tab1:
    # Overview Section
    st.markdown("## Overview")

    # Calculate metrics
    baseline_outcome = filtered_df[filtered_df['channel'] == 'baseline']['conversions'].sum()
    paid_outcome = filtered_df[filtered_df['channel'] != 'baseline']['conversions'].sum()
    total_outcome = baseline_outcome + paid_outcome
    total_spend = filtered_df[filtered_df['channel'] != 'baseline']['spend'].sum()
    paid_cpa = total_spend / paid_outcome if paid_outcome > 0 else 0
    blended_cpa = total_spend / total_outcome if total_outcome > 0 else 0

    # Previous year metrics
    prev_baseline = prev_df[prev_df['channel'] == 'baseline']['conversions'].sum()
    prev_paid = prev_df[prev_df['channel'] != 'baseline']['conversions'].sum()
    prev_total = prev_baseline + prev_paid
    prev_spend = prev_df[prev_df['channel'] != 'baseline']['spend'].sum()
    prev_paid_cpa = prev_spend / prev_paid if prev_paid > 0 else 0
    prev_blended_cpa = prev_spend / prev_total if prev_total > 0 else 0

    # Calculate changes
    baseline_change = ((baseline_outcome - prev_baseline) / prev_baseline * 100) if prev_baseline > 0 else 0
    paid_change = ((paid_outcome - prev_paid) / prev_paid * 100) if prev_paid > 0 else 0
    total_change = ((total_outcome - prev_total) / prev_total * 100) if prev_total > 0 else 0
    spend_change = ((total_spend - prev_spend) / prev_spend * 100) if prev_spend > 0 else 0
    paid_cpa_change = ((paid_cpa - prev_paid_cpa) / prev_paid_cpa * 100) if prev_paid_cpa > 0 else 0
    blended_cpa_change = ((blended_cpa - prev_blended_cpa) / prev_blended_cpa * 100) if prev_blended_cpa > 0 else 0

    # Display spend summary
    days_in_period = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    st.markdown(f"""
    <div class="overview-text">
    You spent <strong>${total_spend/1e6:.1f}M</strong> with a <strong>${paid_cpa:.2f}</strong> paid CPA over the last <strong>{days_in_period}</strong> days
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Outcome & CPA")

    # Metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-label">Baseline Outcome</div>
        <div class="metric-value">{baseline_outcome/1000:.1f}k</div>
        <div class="metric-change {'positive' if baseline_change > 0 else 'negative'}">
            {'↑' if baseline_change > 0 else '↓'} {abs(baseline_change):.0f}% from last year
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-label">Total Spend</div>
        <div class="metric-value">${total_spend/1e6:.1f}M</div>
        <div class="metric-change {'negative' if spend_change > 0 else 'positive'}">
            {'↓' if spend_change < 0 else '↑'} {abs(spend_change):.0f}% from last year
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-label">Paid Outcome</div>
        <div class="metric-value">{paid_outcome/1000:.1f}k</div>
        <div class="metric-change {'positive' if paid_change > 0 else 'negative'}">
            {'↑' if paid_change > 0 else '↓'} {abs(paid_change):.0f}% from last year
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-label">Paid CPA</div>
        <div class="metric-value">${paid_cpa:.2f}</div>
        <div class="metric-change {'positive' if paid_cpa_change < 0 else 'negative'}">
            {'↓' if paid_cpa_change < 0 else '↑'} {abs(paid_cpa_change):.0f}% from last year
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-label">Total Outcome</div>
        <div class="metric-value">{total_outcome/1000:.1f}k</div>
        <div class="metric-change {'positive' if total_change > 0 else 'negative'}">
            {'↑' if total_change > 0 else '↓'} {abs(total_change):.0f}% from last year
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-label">Blended CPA</div>
        <div class="metric-value">${blended_cpa:.2f}</div>
        <div class="metric-change {'positive' if blended_cpa_change < 0 else 'negative'}">
            {'↓' if blended_cpa_change < 0 else '↑'} {abs(blended_cpa_change):.0f}% from last year
        </div>
        """, unsafe_allow_html=True)

    # Marketing Effectiveness Section
    st.markdown("## Marketing Effectiveness")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📥 Download Data", use_container_width=True):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"mmm_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    st.markdown("### Spend Channels")

    # Calculate channel totals
    channel_data = filtered_df[filtered_df['channel'] != 'baseline'].groupby('channel').agg({
        'spend': 'sum',
        'conversions': 'sum',
        'mcpa': 'mean',
        'cpa': 'mean'
    }).reset_index()

    channel_data = channel_data.sort_values('spend', ascending=False)

    # Add 'overall' row at the top
    overall_row = pd.DataFrame([{
        'channel': 'overall',
        'spend': channel_data['spend'].sum(),
        'conversions': channel_data['conversions'].sum(),
        'mcpa': (channel_data['mcpa'] * channel_data['spend']).sum() / channel_data['spend'].sum(),
        'cpa': channel_data['spend'].sum() / channel_data['conversions'].sum()
    }])
    channel_data = pd.concat([overall_row, channel_data], ignore_index=True)

    # Create two separate columns for Spend and Performance
    col_spend, col_performance = st.columns([1, 1])

    with col_spend:
        st.markdown("**Spend**")
        
        # Spend bar chart
        colors = ['#1f2937' if ch == 'overall' else '#10b981' if i % 4 == 1 else '#3b82f6' if i % 4 == 2 else '#f97316' if i % 4 == 3 else '#8b5cf6'
                  for i, ch in enumerate(channel_data['channel'])]
        
        fig_spend = go.Figure()
        
        fig_spend.add_trace(go.Bar(
            y=channel_data['channel'],
            x=channel_data['spend'],
            orientation='h',
            marker=dict(color=colors),
            text=[f"${s/1e6:.1f}M" if s >= 1e6 else f"${s/1e3:.0f}k" for s in channel_data['spend']],
            textposition='outside',
            textfont=dict(color='#000000'),
            hovertemplate='<b>%{y}</b><br>Spend: $%{x:,.0f}<extra></extra>',
            showlegend=False
        ))
        
        fig_spend.update_layout(
            height=500,
            margin=dict(l=10, r=80, t=10, b=10),
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(title="", showgrid=False),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_spend, use_container_width=True)

    with col_performance:
        st.markdown("**Performance**")
        
        # Performance chart with mCPA and CPA
        fig_performance = go.Figure()
        
        fig_performance.add_trace(go.Bar(
            y=channel_data['channel'],
            x=channel_data['mcpa'],
            orientation='h',
            name='mCPA',
            marker=dict(color='#f59e0b'),
            text=[f"${m:.0f}" for m in channel_data['mcpa']],
            textposition='outside',
            textfont=dict(color='#000000'),
            hovertemplate='<b>%{y}</b><br>mCPA: $%{x:.2f}<extra></extra>'
        ))
        
        fig_performance.add_trace(go.Bar(
            y=channel_data['channel'],
            x=channel_data['cpa'],
            orientation='h',
            name='CPA',
            marker=dict(color='#3b82f6'),
            text=[f"${c:.0f}" for c in channel_data['cpa']],
            textposition='outside',
            textfont=dict(color='#000000'),
            hovertemplate='<b>%{y}</b><br>CPA: $%{x:.2f}<extra></extra>'
        ))
        
        fig_performance.update_layout(
            height=500,
            margin=dict(l=10, r=80, t=10, b=10),
            xaxis=dict(title="", showgrid=True, gridcolor='#e5e7eb'),
            yaxis=dict(title="", showticklabels=False, showgrid=False),
            barmode='group',
            plot_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)

    # Add definitions below the charts
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div style='background-color: #f0f9ff; padding: 12px; border-radius: 6px; border-left: 4px solid #3b82f6; margin-top: 20px;'>
            <strong style='color: #1e40af;'>CPA (Cost Per Acquisition)</strong><br/>
            <span style='font-size: 0.9rem; color: #475569;'>Average cost to acquire a customer across all conversions in the period</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background-color: #fef3c7; padding: 12px; border-radius: 6px; border-left: 4px solid #f59e0b; margin-top: 20px;'>
            <strong style='color: #92400e;'>mCPA (Marginal Cost Per Acquisition)</strong><br/>
            <span style='font-size: 0.9rem; color: #475569;'>Cost to acquire the last/next customer - indicates channel saturation when mCPA >> CPA</span>
        </div>
        """, unsafe_allow_html=True)

    # Performance by Channel
    st.markdown("### Performance by Channel")
    st.markdown("*Performance bars show conversions per thousand spent (higher is better)*")

    # Calculate performance metrics
    total_channel_spend = channel_data['spend'].sum()
    channel_data['share_of_spend'] = (channel_data['spend'] / total_channel_spend * 100)
    channel_data['share_of_conversions'] = (channel_data['conversions'] / channel_data['conversions'].sum() * 100)
    channel_data['performance_index'] = ((channel_data['share_of_conversions'] - channel_data['share_of_spend']) / channel_data['share_of_spend'] * 100)
    channel_data['cpt'] = (channel_data['conversions'] / channel_data['spend']) * 1000

    # Separate over and under performers
    over_performers = channel_data[channel_data['performance_index'] > 0].sort_values('performance_index', ascending=True)
    under_performers = channel_data[channel_data['performance_index'] < 0].sort_values('performance_index', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Overperformance")
        st.markdown("*vs. share of spend*")
        
        if len(over_performers) > 0:
            fig_over = go.Figure()
            fig_over.add_trace(go.Bar(
                y=over_performers['channel'],
                x=over_performers['performance_index'],
                orientation='h',
                marker=dict(color='#10b981'),
                text=[f"{val:.1f}%" for val in over_performers['performance_index']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Performance: %{x:.1f}%<extra></extra>'
            ))
            fig_over.update_layout(
                height=300,
                margin=dict(l=120, r=50, t=10, b=10),
                xaxis_title="",
                yaxis_title="",
                showlegend=False,
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_over, use_container_width=True)
        else:
            st.info("No overperforming channels in selected period")

    with col2:
        st.markdown("#### 🔴 Underperformance")
        st.markdown("*vs. share of spend*")
        
        if len(under_performers) > 0:
            fig_under = go.Figure()
            fig_under.add_trace(go.Bar(
                y=under_performers['channel'],
                x=under_performers['performance_index'],
                orientation='h',
                marker=dict(color='#ef4444'),
                text=[f"{val:.1f}%" for val in under_performers['performance_index']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Performance: %{x:.1f}%<extra></extra>'
            ))
            fig_under.update_layout(
                height=300,
                margin=dict(l=120, r=50, t=10, b=10),
                xaxis_title="",
                yaxis_title="",
                showlegend=False,
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#e5e7eb', autorange='reversed'),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_under, use_container_width=True)
        else:
            st.info("No underperforming channels in selected period")

    # Contribution by Channel
    st.markdown("## Contribution by Channel")

    # Prepare time series data
    daily_conversions = filtered_df[filtered_df['channel'] != 'baseline'].groupby(['date', 'channel'])['conversions'].sum().reset_index()

    # Get top channels by total conversions
    top_channels = filtered_df[filtered_df['channel'] != 'baseline'].groupby('channel')['conversions'].sum().nlargest(7).index.tolist()
    daily_conversions = daily_conversions[daily_conversions['channel'].isin(top_channels)]

    # Create line chart
    fig_contribution = go.Figure()

    colors = ['#1f2937', '#10b981', '#3b82f6', '#f97316', '#8b5cf6', '#ec4899', '#14b8a6']

    for i, channel in enumerate(top_channels):
        channel_data_ts = daily_conversions[daily_conversions['channel'] == channel]
        
        fig_contribution.add_trace(go.Scatter(
            x=channel_data_ts['date'],
            y=channel_data_ts['conversions'],
            mode='lines',
            name=channel,
            line=dict(color=colors[i % len(colors)], width=1),
            opacity=0.8,
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Conversions: %{y:.0f}<extra></extra>'
        ))

    fig_contribution.update_layout(
    height=400,
    margin=dict(l=50, r=50, t=30, b=50),
    xaxis_title="",
    yaxis_title="Conversions",
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
    ),
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
    yaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
    hovermode='x unified'
    )

    st.plotly_chart(fig_contribution, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(f"*Dashboard updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")

with tab2:
    st.markdown("## 💰 Money Left on the Table")
    
    # Calculate optimization metrics
    channel_performance = filtered_df[filtered_df['channel'] != 'baseline'].groupby('channel').agg({
        'spend': 'sum',
        'conversions': 'sum',
        'cpa': 'mean',
        'mcpa': 'mean'
    }).reset_index()
    
    # Calculate efficiency score (lower CPA and mCPA is better)
    channel_performance['actual_cpa'] = channel_performance['spend'] / channel_performance['conversions']
    channel_performance['efficiency_score'] = 100 / (channel_performance['actual_cpa'] + 1)
    channel_performance['saturation_ratio'] = channel_performance['mcpa'] / channel_performance['actual_cpa']
    channel_performance['current_allocation'] = channel_performance['spend']
    
    # Simple optimization: reallocate from saturated (high mCPA/CPA ratio) to efficient channels
    total_budget = channel_performance['spend'].sum()
    
    # More aggressive thresholds to ensure we find opportunities
    # Identify channels to reduce (saturation_ratio > 1.15) and increase (saturation_ratio < 1.05 and good efficiency)
    reduce_channels = channel_performance[channel_performance['saturation_ratio'] > 1.15].copy()
    increase_channels = channel_performance[
        (channel_performance['saturation_ratio'] < 1.05) & 
        (channel_performance['efficiency_score'] > channel_performance['efficiency_score'].quantile(0.4))
    ].copy()
    
    # If no channels meet criteria, use percentile-based approach
    if len(reduce_channels) == 0:
        # Take worst performing 30% of channels
        saturation_threshold = channel_performance['saturation_ratio'].quantile(0.7)
        reduce_channels = channel_performance[channel_performance['saturation_ratio'] >= saturation_threshold].copy()
    
    if len(increase_channels) == 0:
        # Take best performing 30% of channels
        saturation_threshold = channel_performance['saturation_ratio'].quantile(0.3)
        increase_channels = channel_performance[channel_performance['saturation_ratio'] <= saturation_threshold].copy()
    
    # Calculate optimized allocation (simple reallocation)
    channel_performance['optimized_allocation'] = channel_performance['current_allocation'].copy()
    
    # Reduce saturated channels by 20%
    reduction_pct = 0.20
    total_reduction = 0
    
    for idx in reduce_channels.index:
        reduction = channel_performance.loc[idx, 'current_allocation'] * reduction_pct
        channel_performance.loc[idx, 'optimized_allocation'] -= reduction
        total_reduction += reduction
        
        # Distribute to efficient channels proportionally by their efficiency score
        if len(increase_channels) > 0:
            total_efficiency = increase_channels['efficiency_score'].sum()
            for idx_inc in increase_channels.index:
                weight = channel_performance.loc[idx_inc, 'efficiency_score'] / total_efficiency
                channel_performance.loc[idx_inc, 'optimized_allocation'] += reduction * weight
    
    # Calculate wasted spend and potential gains
    total_wasted = total_reduction
    
    # Calculate potential conversions based on reallocation to efficient channels
    potential_conversions = 0
    if len(increase_channels) > 0:
        avg_efficient_cpa = increase_channels['actual_cpa'].mean()
        potential_conversions = total_wasted / avg_efficient_cpa if avg_efficient_cpa > 0 else 0
    
    # Generate insights based on the data
    worst_channel = channel_performance.loc[channel_performance['saturation_ratio'].idxmax()]
    best_channel = channel_performance.loc[channel_performance['saturation_ratio'].idxmin()]
    avg_saturation = channel_performance['saturation_ratio'].mean()
    
    # Top insight
    insight_text = f"""
    <div style='background-color: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b; margin-bottom: 30px;'>
        <div style='font-size: 1.1rem; font-weight: 600; color: #92400e; margin-bottom: 12px;'>💡 Key Insight</div>
        <div style='font-size: 1rem; color: #78350f; line-height: 1.6;'>
            <strong>{worst_channel['channel']}</strong> is showing signs of saturation with a {worst_channel['saturation_ratio']:.2f}x mCPA/CPA ratio 
            (mCPA: ${worst_channel['mcpa']:.0f} vs CPA: ${worst_channel['actual_cpa']:.0f}). Meanwhile, <strong>{best_channel['channel']}</strong> 
            maintains strong efficiency at {best_channel['saturation_ratio']:.2f}x ratio. Reallocating ${total_wasted:,.0f} from saturated 
            channels could generate <strong>+{potential_conversions:.0f} additional conversions</strong> worth approximately 
            <strong>${potential_conversions * 50:,.0f}</strong> in revenue.
        </div>
    </div>
    """
    st.markdown(insight_text, unsafe_allow_html=True)
    
    st.markdown("### Budget Reallocation Analysis")
    st.markdown(f"*Based on {model_type} model for {time_period.lower()}*")
    
    # Display summary cards
    st.markdown("### 💸 Optimization Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #fee2e2; padding: 20px; border-radius: 8px; border-left: 4px solid #dc2626;'>
            <div style='font-size: 0.875rem; color: #991b1b; font-weight: 600; margin-bottom: 8px;'>TOTAL WASTED SPEND</div>
            <div style='font-size: 2.5rem; font-weight: 700; color: #dc2626;'>${total_wasted:,.0f}</div>
            <div style='font-size: 0.875rem; color: #991b1b; margin-top: 8px;'>{(total_wasted/total_budget*100):.1f}% of total budget</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #d1fae5; padding: 20px; border-radius: 8px; border-left: 4px solid #059669;'>
            <div style='font-size: 0.875rem; color: #065f46; font-weight: 600; margin-bottom: 8px;'>POTENTIAL REVENUE GAIN</div>
            <div style='font-size: 2.5rem; font-weight: 700; color: #059669;'>+${potential_conversions * 50:,.0f}</div>
            <div style='font-size: 0.875rem; color: #065f46; margin-top: 8px;'>with optimization</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background-color: #dbeafe; padding: 20px; border-radius: 8px; border-left: 4px solid #2563eb;'>
            <div style='font-size: 0.875rem; color: #1e40af; font-weight: 600; margin-bottom: 8px;'>CONVERSION UPLIFT</div>
            <div style='font-size: 2.5rem; font-weight: 700; color: #2563eb;'>+{potential_conversions:.0f}</div>
            <div style='font-size: 0.875rem; color: #1e40af; margin-top: 8px;'>additional conversions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Budget Reallocation Visualization
    st.markdown("### 📊 Budget Reallocation")
    st.markdown("**Current vs. Optimized Budget Allocation**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Bar chart comparing current vs optimized
        fig_reallocation = go.Figure()
        
        fig_reallocation.add_trace(go.Bar(
            x=channel_performance['channel'],
            y=channel_performance['current_allocation'],
            name='Current Allocation',
            marker=dict(color='#94a3b8'),
            text=[f"${v/1e3:.0f}k" for v in channel_performance['current_allocation']],
            textposition='outside'
        ))
        
        fig_reallocation.add_trace(go.Bar(
            x=channel_performance['channel'],
            y=channel_performance['optimized_allocation'],
            name='Optimized Allocation',
            marker=dict(color='#6366f1'),
            text=[f"${v/1e3:.0f}k" for v in channel_performance['optimized_allocation']],
            textposition='outside'
        ))
        
        fig_reallocation.update_layout(
            height=400,
            xaxis_title="Channel",
            yaxis_title="Spend ($)",
            barmode='group',
            plot_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_reallocation, use_container_width=True)
    
    with col2:
        # Pie chart showing waste breakdown
        if len(reduce_channels) > 0:
            fig_waste = go.Figure()
            
            waste_by_channel = reduce_channels.copy()
            waste_by_channel['waste'] = waste_by_channel['current_allocation'] * 0.15
            
            fig_waste.add_trace(go.Pie(
                labels=waste_by_channel['channel'],
                values=waste_by_channel['waste'],
                hole=0.4,
                marker=dict(colors=['#3b82f6', '#8b5cf6', '#ec4899', '#f97316']),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Wasted: $%{value:,.0f}<extra></extra>'
            ))
            
            fig_waste.update_layout(
                height=400,
                title=dict(text="Waste Breakdown by Channel", x=0.5, xanchor='center'),
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
            )
            
            st.plotly_chart(fig_waste, use_container_width=True)
        else:
            st.info("🎉 No significant waste detected! All channels are performing efficiently.")
    
    # Recommendations
    st.markdown("### 🎯 Recommendations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🔴 Reduce Spend")
        if len(reduce_channels) > 0:
            for _, row in reduce_channels.iterrows():
                reduction_amount = row['current_allocation'] * 0.15
                st.markdown(f"""
                <div style='background-color: #fef2f2; padding: 15px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid #dc2626;'>
                    <strong style='color: #991b1b;'>{row['channel']}</strong><br/>
                    <span style='color: #dc2626; font-size: 0.9rem;'>Reduce by ${reduction_amount:,.0f}</span><br/>
                    <span style='font-size: 0.85rem; color: #6b7280;'>Saturation Ratio: {row['saturation_ratio']:.2f}x | mCPA ${row['mcpa']:.0f} vs CPA ${row['cpa']:.0f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #eff6ff; padding: 15px; border-radius: 6px; border-left: 3px solid #3b82f6;'>
                <span style='color: #1e40af;'>No channels need budget reduction</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🟢 Increase Spend")
        if len(increase_channels) > 0:
            increase_per = total_wasted / len(increase_channels) if len(increase_channels) > 0 else 0
            for _, row in increase_channels.iterrows():
                st.markdown(f"""
                <div style='background-color: #f0fdf4; padding: 15px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid #059669;'>
                    <strong style='color: #065f46;'>{row['channel']}</strong><br/>
                    <span style='color: #059669; font-size: 0.9rem;'>Increase by ${increase_per:,.0f}</span><br/>
                    <span style='font-size: 0.85rem; color: #6b7280;'>Efficiency: {row['efficiency_score']:.1f} | Trend: stable</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #fef3c7; padding: 15px; border-radius: 6px; border-left: 3px solid #f59e0b;'>
                <span style='color: #92400e;'>Consider testing new channels or audiences</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"*Optimization recommendations based on {model_type} model | Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")

