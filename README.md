# Sri Lanka Mobile Phone Price Prediction

An end-to-end Machine Learning project to predict the price of mobile phones in Sri Lanka, based on data scraped from ikman.lk. This project demonstrates web scraping, data preprocessing, machine learning with XGBoost, model explainability using SHAP, and an interactive Streamlit UI.

## Folder Structure

```
├── app/
│   └── streamlit_app.py      # Interactive web application for price prediction
├── data/
│   ├── mobiles.csv           # Raw scraped dataset
│   └── processed/            # Cleaned train, test, val splits, and reference data
├── models/                   # Saved XGBoost model and LabelEncoders
├── outputs/
│   ├── metrics.json          # Test evaluation metrics (RMSE, MAE, R2)
│   ├── metrics_table.csv     # Tabular test evaluation metrics
│   └── plots/                # SHAP plots, Predicted vs Actual, and Residual histograms
├── src/
│   ├── preprocess.py         # Data cleaning, feature engineering, and splitting
│   ├── train.py              # XGBoost training and hyperparameter tuning 
│   ├── evaluate.py           # Model evaluation and plot generation
│   ├── explain.py            # SHAP model explainability global/local plots
│   └── utils.py              # Helper functions (empty placeholder)
├── scrape.py                 # (Optional/Archive) Script to scrape data from ikman.lk
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── report_outline.md         # Outline for the final project report
```

## Installation Steps

1. **Clone the repository** (if applicable) or ensure you are in the project root directory.
2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Run Commands

Execute the exact sequence below to run the project entirely from end-to-end:

**1. Data Scraping (Optional - run only if `data/mobiles.csv` is missing)**
```bash
python scrape.py
```
*(Note: A grid-search scraper was used to collect 5,500 records.)*

**2. Data Preprocessing**
```bash
python src/preprocess.py --input data/mobiles.csv --output data/processed/
```

**3. Model Training**
```bash
python src/train.py --data data/processed/
```

**4. Model Evaluation**
```bash
python src/evaluate.py
```

**5. Model Explainability**
```bash
python src/explain.py
```

**6. Launch Streamlit Frontend**
```bash
streamlit run app/streamlit_app.py
```

## Ethical Web Scraping Note

The accompanying `scrape.py` script adheres to general ethical scraping principles:
- Randomized pacing (time.sleep) is used between requests to prevent overwhelming the server.
- The `robots.txt` guidelines of the source website should be respected. 
- Scraped data is intended purely for educational and machine learning assignment purposes without attempting to replicate or compete with the original website's business logic.
