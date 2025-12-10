import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mmm_data(days=365, seed=42, model='Prophet'):
    """
    Generate synthetic MMM data for demonstration
    Model affects the conversion efficiency and baseline calculations
    """
    # Different seed for different models
    if model == 'Bayesian MMM':
        np.random.seed(seed + 100)
    else:
        np.random.seed(seed)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Channels
    channels = [
        'meta_prospecting',
        'linear_tv',
        'mailers',
        'meta_retargeting',
        'podcast',
        'search_branded',
        'influencers'
    ]
    
    # Data sources
    sources = ['Google Analytics', 'Facebook Ads', 'TV Attribution', 'CRM']
    
    # Industries
    industries = ['E-commerce', 'SaaS', 'Fintech']
    
    # Generate daily data
    data = []
    
    for date in dates:
        # Seasonal component
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        for channel in channels:
            for industry in industries:
                # Base daily spend varies by channel (these are daily budgets)
                base_spend = {
                    'meta_prospecting': 3500,
                    'linear_tv': 2000,
                    'mailers': 1200,
                    'meta_retargeting': 1000,
                    'podcast': 900,
                    'search_branded': 600,
                    'influencers': 400
                }
                
                # Industry multiplier
                industry_mult = 1.2 if industry == 'E-commerce' else 1.0 if industry == 'Fintech' else 0.8
                
                # Daily spend with variation
                daily_spend = base_spend[channel] * seasonal_factor * industry_mult
                daily_spend *= np.random.uniform(0.7, 1.3)
                
                # Conversions based on channel efficiency
                channel_efficiency = {
                    'meta_prospecting': 0.08,
                    'linear_tv': 0.11,
                    'mailers': 0.06,
                    'meta_retargeting': 0.25,
                    'podcast': 0.04,
                    'search_branded': 0.05,
                    'influencers': 0.22
                }
                
                # Model affects efficiency (Bayesian MMM typically more conservative)
                model_multiplier = 0.85 if model == 'Bayesian MMM' else 1.0
                
                conversions = daily_spend * channel_efficiency[channel] * model_multiplier * np.random.uniform(0.8, 1.2)
                
                # CPA
                cpa = daily_spend / conversions if conversions > 0 else 0
                
                # mCPA (marginal CPA) - varies around CPA
                mcpa = cpa * np.random.uniform(0.9, 1.4)
                
                # Random source
                source = np.random.choice(sources)
                
                data.append({
                    'date': date,
                    'channel': channel,
                    'industry': industry,
                    'source': source,
                    'spend': daily_spend,
                    'conversions': conversions,
                    'cpa': cpa,
                    'mcpa': mcpa
                })
    
    df = pd.DataFrame(data)
    
    # Add baseline conversions (organic/non-paid)
    baseline_data = []
    for date in dates:
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        
        for industry in industries:
            base_conversions = 250 if industry == 'E-commerce' else 200 if industry == 'Fintech' else 180
            # Bayesian MMM attributes more to baseline
            baseline_multiplier = 1.15 if model == 'Bayesian MMM' else 1.0
            daily_baseline = base_conversions * baseline_multiplier * seasonal_factor * np.random.uniform(0.85, 1.15)
            
            baseline_data.append({
                'date': date,
                'channel': 'baseline',
                'industry': industry,
                'source': 'Organic',
                'spend': 0,
                'conversions': daily_baseline,
                'cpa': 0,
                'mcpa': 0
            })
    
    baseline_df = pd.DataFrame(baseline_data)
    df = pd.concat([df, baseline_df], ignore_index=True)
    
    return df


def get_performance_metrics(df, channel, share_of_spend):
    """
    Calculate performance vs share of spend
    """
    # Conversions per thousand spent
    total_spend = df[df['channel'] == channel]['spend'].sum()
    total_conversions = df[df['channel'] == channel]['conversions'].sum()
    
    if total_spend > 0:
        cpt = (total_conversions / total_spend) * 1000
    else:
        cpt = 0
    
    # Performance index
    if share_of_spend > 0:
        share_of_conversions = total_conversions / df[df['channel'] != 'baseline']['conversions'].sum() * 100
        performance_index = (share_of_conversions - share_of_spend) / share_of_spend * 100
    else:
        performance_index = 0
    
    return performance_index, cpt


if __name__ == "__main__":
    df = generate_mmm_data()
    df.to_csv('mmm_data.csv', index=False)
    print(f"Generated {len(df)} rows of MMM data")
    print(df.head())
