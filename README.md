#  NIFTY-50 Investment Intelligence Platform

An AI-powered investment intelligence platform that transforms 21 years of NIFTY-50 stock market data into actionable investment insights.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---
##  Live Demo
 [Click here to view the Live Dashboard](https://nifty50-investment-platform-tettekpot9neqsrw84bevc.streamlit.app/)

---

##  Project Overview

This platform provides:
-  **Exploratory Data Analysis** of 49 NIFTY-50 stocks (2000–2021)
-  **LSTM Deep Learning** model for stock price prediction
-  **Portfolio Optimization** for 3 investor profiles
-  **Risk Assessment** with Sharpe Ratio, VaR, Drawdown
-  **Explainability Module** with BUY/SELL/HOLD signals
-  **Interactive Streamlit Dashboard**

---

##  Project Structure
```
nifty50-investment-platform/
│
├── app/
│   └── dashboard.py
├── data/
│   ├── raw/stocks/
│   │   └── stock_metadata.csv
│   └── processed/
│       ├── all_stocks_featured.csv
│       ├── portfolio_weights.csv
│       ├── portfolio_risk.csv
│       ├── risk_metrics.csv
│       └── hdfcbank_predictions.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_stock_predictor.ipynb
│   ├── 04_portfolio_construction.ipynb
│   └── 05_risk_assessment.ipynb
├── outputs/
│   └── (16 chart PNG files)
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── portfolio.py
│   └── risk.py
├── models/
│   └── lstm_hdfcbank.keras
├── Technical_Report.pdf
├── runtime.txt
├── requirements.txt
└── README.md
```
---

##  Getting Started

### Prerequisites
- Python 3.12+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/KanishkaGupta28/nifty50-investment-platform.git
cd nifty50-investment-platform
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download the NIFTY-50 dataset from Kaggle:
- [Primary Dataset](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
- [Additional Dataset](https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020)

Place stock CSV files in `data/raw/stocks/` and index files in `data/raw/indices/`

### 5. Run Notebooks in Order
```bash
jupyter notebook
```
Run notebooks in this order:
1. `01_EDA.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_stock_predictor.ipynb`
4. `04_portfolio_construction.ipynb`
5. `05_risk_assessment.ipynb`

### 6. Launch Dashboard
```bash
cd app
python -m streamlit run dashboard.py
```

Open browser at `http://localhost:8501`

---

##  Dashboard Features

| Page | Description |
|------|-------------|
|  Home | Market overview, KPIs, sector analysis |
|  Stock Analysis | Technical indicators, RSI, MACD, Bollinger Bands |
|  Portfolio Builder | 3 investor profiles with growth simulation |
|  Risk Assessment | Sharpe Ratio, VaR, Drawdown analysis |
|  Price Predictor | LSTM predictions vs actual prices |
|  Explainability | BUY/SELL/HOLD signals, feature importance |

---

##  Model Performance

### LSTM Stock Price Predictor (HDFCBANK)
| Metric | Value |
|--------|-------|
| RMSE | ₹202.25 |
| MAE | ₹159.43 |
| R² Score | 0.7418 |
| Direction Accuracy | 51.54% |

### Model Comparison
| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | ₹312.45 | 0.62 |
| Random Forest | ₹245.80 | 0.74 |
| **LSTM (Ours)** | **₹202.25** | **0.7418** |

---

##  Portfolio Results

| Profile | Annual Return | Risk | Sharpe Ratio |
|---------|--------------|------|--------------|
| Conservative | ~15% | ~18% | ~0.5 |
| Balanced | ~22% | ~24% | ~0.67 |
| Aggressive | ~28% | ~32% | ~0.69 |

---

##  Risk Metrics (Key Findings)

- **Best Sharpe Ratio:** BAJAJFINSV (0.89)
- **Most Volatile:** BAJFINANCE
- **Least Volatile:** POWERGRID
- **Best Total Return:** BAJAJFINSV (+2019%)

---

##  Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| ML Models | Scikit-learn |
| Optimization | SciPy |
| Technical Indicators | TA Library |

---

##  Requirements
```
See `requirements.txt` for full list. Key dependencies:
pandas
numpy
tensorflow
scikit-learn
streamlit
plotly
scipy
ta
matplotlib
seaborn
statsmodels
```
---

##  Key Insights

1. **BAJAJFINSV** delivered the best risk-adjusted returns (Sharpe: 0.89)
2. **Financial Services** sector dominates NIFTY-50 with 9 companies
3. **2008 and 2020** caused the largest market drawdowns
4. **Moving Averages** are the strongest predictors of stock price
5. **Long-term investing** works — NIFTY-50 grew 190% overall
6. **POWERGRID** is the safest stock with lowest volatility

---
## Team Members

| Name | Enrollment No. |
|------|---------------|
| Kanishka Gupta | (23118037) |
| Anurag Sain | (23115018) |
 
