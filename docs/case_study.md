# Case Study: Olist Order Delay Early-Warning System

## 1. The Problem
*   **Context:** At the moment an order is placed, we don't know if it will arrive late.
*   **Action:** If flagged, the Ops team can warn the customer, upgrade shipping, or flag the seller.
*   **Costs:** Missed delay costs R$50 (refunds + bad reviews). False alarm costs R$5 (ops time).

## 2. Data & Methodology
*   **Data:** 100,000 orders across 9 relational tables (Brazilian e-commerce).
*   **Key Decision:** We used a **Time-Based Split** (trained on 2016-2018, tested on 2018). We intentionally avoided data leakage by excluding post-delivery information (e.g., actual delivery date, review scores) from the feature set.

## 3. Results (Tested on Future Data)
*   **Baseline (Never Late):** 94% Accuracy, 0% Recall (catches zero delays).
*   **Our Model (Logistic Regression):** ROC-AUC **0.873**, Recall **67%** (catches 1,032 delays).
*   **Business ROI:** At a threshold of **0.60**, the campaign generates a net positive value of **R$ 29,415**. (At the default threshold of 0.50, the value is lower).

## 4. What We Learned (The AI PM Insights)
*   **Feature Engineering > Model Complexity:** Our first model had an AUC of 0.44 (worse than a coin flip). We engineered two historical rolling features (`seller_historic_delay_rate` and `customer_region_delay_rate`) which pushed the AUC to 0.873. The simple Logistic Regression beat the Random Forest.
*   **The "Cost of Good Intentions":** Optimizing purely for Recall (catching every delay) destroys precision and generates massive false alarms (wasted money). The threshold must be tied strictly to the business's cost assumptions.

## 5. Limitations
*   **Data:** Public Brazilian e-commerce dataset. May not generalize perfectly to other markets.
*   **Missing Features:** We don't have seller processing times, holiday calendars, or live weather data.
*   **Monitoring:** We generated a data drift report (Evidently), but no live alerting pipeline is in place yet.

## 6. Next Steps for Production
1. **Live Pipeline:** Move the DuckDB data prep into dbt/Airflow to refresh daily.
2. **API Integration:** Deploy the FastAPI endpoint to AWS/GCP and connect it to the checkout page.
3. **Automated Retraining:** Set up a trigger to retrain the model if Evidently detects high drift.