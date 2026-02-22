# Report Outline: Sri Lanka Mobile Phone Price Prediction

## 1. Introduction
*   **Problem Statement:** Predicting the price of mobile phones in the Sri Lankan second-hand and retail market based on brand, model, condition, storage capacity, and location.
*   **Motivation:** To provide buyers and sellers a fair market value estimate based on real-time data from classifieds.
*   **Target Audience:** Consumers, mobile phone vendors, and e-commerce platforms.

## 2. Data Collection (Web Scraping)
*   **Source:** ikman.lk classified advertisements.
*   **Methodology:** 
    *   Bypassing strict pagination limits utilizing a location-and-query "grid search" mechanism.
    *   Direct embedded JSON extraction (`window.initialData`) via `requests` and Python `re` modules for reliable, fast scraping over BeautifulSoup HTML parsing.
*   **Ethical Considerations:** Implemented random request pacing to limit server load.

## 3. Data Preprocessing & Feature Engineering
*   **Currency Cleaning:** Stripping 'Rs', commas, and translating terms like "Lakh" and "Mn" into standard numeric formats.
*   **Feature Engineering:**
    *   **Brand & Model:** Extracting key identifiers from the unstructured ad title.
    *   **Storage (GB):** Using Regular Expressions to parse storage capacities (e.g., "128GB", "1TB").
*   **Handling Missing Values:** Imputing missing storage values with the median for the specific brand.
*   **Encoding:** Applying `LabelEncoder` for high-cardinality categorical variables keeping unseen categories as 'Other'.

## 4. Model Selection & Training
*   **Algorithm Chosen:** XGBoost Regressor (chosen for its robust handling of non-linear relationships and interactions in tabular data).
*   **Hyperparameter Tuning:** 
    *   Used `RandomizedSearchCV` to efficiently scour the hyperparameter space (learning rate, depth, estimators) saving substantial compute time over Grid Search.
*   **Early Stopping:** Evaluated against a distinct validation set (15%) to prevent overfitting and halt training at optimal epochs.

## 5. Model Evaluation
*   **Metrics Explained:**
    *   **RMSE (Root Mean Squared Error):** Indicates the average prediction error magnitude in Rupees.
    *   **MAE (Mean Absolute Error):** Indicates the absolute typical misestimation in Rupees.
    *   **R² Score:** Shows the percentage of price variance the model currently explains.
*   **Visual Analysis:**
    *   **Predicted vs Actual Scatter Plot:** Assessing variance alignment along the perfect prediction diagonal.
    *   **Residual Histogram:** Checking if errors are normally distributed and centered around zero.

## 6. Model Explainability (SHAP)
*   Leveraged `TreeExplainer` for complex XGBoost trees.
*   **Global Explainability:**
    *   *Feature Importance Bar Chart*
    *   *SHAP Summary Plot:* Analyzing the directional impact of features (e.g., does high storage increase or decrease price uniformly?).
    *   *Dependence Plot:* Visualizing the isolated impact of the most dominant feature (often 'Brand').
*   **Local Explainability:**
    *   Used inside the UI to break down individual predictions through a Waterfall plot to build user trust.

## 7. Streamlit Application Deployment
*   **Architecture:** Zero-backend UI purely relying on the saved `xgb_model.joblib` and encoding mappings.
*   **User Interface:** Dynamic filtering of dropdown inputs relying on valid combinations extracted during preprocessing.
*   **Delivery:** Providing immediate insights via the final predicted price, model accuracy proofs, and the localized SHAP waterfall chart.

## 8. Critical Discussion & Conclusion
*   **Limitations (Guideline #5):** 
    *   **Feature Scope:** The model lacks physical inspection data (e.g., screen scratches, battery health percentage) which significantly affects price.
    *   **Static Data:** Prices are a snapshot; LKR inflation or new phone releases (e.g., iPhone 16) will make the current model obsolete.
*   **Data Quality & Bias:**
    *   **Source Bias:** Data from `ikman.lk` represents people who use online classifieds, potentially missing the offline market or high-end dealer prices.
    *   **Cleaning:** Removing "Unknown" models improved UX but slightly reduced the volume of niche records.
*   **Ethical Considerations & Real-World Impact:**
    *   **Impact:** Empowers non-technical users to avoid overpaying or being scammed.
    *   **Ethical Use:** No private seller information (names, phone numbers) was collected or stored.
*   **Conclusion:** The pipeline successfully handles raw scraped unstructured data to produce meaningful, explainable real-world pricing intelligence with a functional front-end.
