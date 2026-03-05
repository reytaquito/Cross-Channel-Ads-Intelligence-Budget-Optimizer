# Cross-Channel-Ads-Intelligence-Budget-Optimizer
Interactive data analysis project for paid media performance across Facebook, Google Ads, and TikTok Ads.
## Overview
This project combines 3 ad platform datasets into one analytics pipeline and dashboard to:
- Compare performance across channels
- Calculate core marketing KPIs
- Detect underperforming campaigns
- Simulate budget reallocation for better conversion outcomes

## Key Features
- Unified ETL for Facebook, Google, TikTok CSV files
- KPI engine: CTR, CPC, CPM, CVR, CPA, ROAS
- Campaign efficiency scoring
- Performance alerts (high CPA, low CTR, low ROAS)
- Budget optimizer with diminishing returns logic
- Interactive Streamlit dashboard

## Input Datasets
Default local paths:
- `/Users/rodrigomateos/Desktop/01_facebook_ads.csv`
- `/Users/rodrigomateos/Desktop/02_google_ads.csv`
- `/Users/rodrigomateos/Desktop/03_tiktok_ads.csv`

You can also upload files directly in the app sidebar.

## Tech Stack
- Python
- pandas, numpy
- streamlit
- plotly

## Run Locally
```bash
cd "/Users/rodrigomateos/Documents/New project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
