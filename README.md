# Media Mix Modelling Dashboard - MVP

A streamlined Media Mix Modelling (MMM) dashboard built with Streamlit for analyzing marketing channel performance and budget optimization.

## 🚀 Features

- **Performance Analytics** - Real-time tracking of spend, conversions, CPA, and mCPA across channels
- **Budget Optimization** - Identifies wasted spend and reallocation opportunities
- **Multi-Industry Support** - Pre-configured for E-commerce, SaaS, and Fintech
- **Interactive Visualizations** - Powered by Plotly for deep-dive analysis
- **CSV Upload** - Use test data or upload your own marketing data

## 📊 Live Demo

[View Dashboard](https://mmm-mvp.streamlit.app) *(update with your URL after deployment)*

## 🛠️ Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/spravin3/mmm-mvp.git
cd mmm-mvp

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork/clone this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository: `spravin3/mmm-mvp`
5. Set main file: `app.py`
6. Click "Deploy"

## 📁 Project Structure

```
mmm-mvp/
├── app.py                      # Main Streamlit application
├── data_generator.py           # Synthetic data generation
├── requirements.txt            # Python dependencies
├── sample_data_template.csv    # CSV upload format reference
└── README.md                   # Documentation
```

## 📈 Usage

### Filters
- **Time Range** - Last 7, 30, or 90 days
- **Industry** - E-commerce, SaaS, or Fintech
- **Data Source** - Test data or CSV upload

### Tabs
1. **Performance Dashboard** - Channel spend, conversions, CPA/mCPA analysis
2. **Budget Optimization** - Reallocation recommendations and potential gains

## 🔧 Tech Stack

- **Streamlit** 1.29.0 - Web framework
- **Plotly** 5.18.0 - Interactive visualizations
- **Pandas** 2.1.4 - Data manipulation
- **NumPy** 1.26.2 - Numerical computing

## 📄 License

MIT License - feel free to use and modify

## 🤝 Contributing

Built as MVP - contributions welcome for production enhancements

---

*Built with Streamlit 🎈*
