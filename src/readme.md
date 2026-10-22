# Olist Order Delay Early-Warning System

## The Problem
At the moment an order is placed, we do not know if it will arrive late. Currently, we only find out after the promised delivery date has passed. By then, the customer is frustrated, and the operations team is reacting to complaints rather than preventing them.

## The Solution
We built a predictive model that scores an order's risk of delay **at the moment of checkout**. If an order is flagged as high-risk, the operations team is proactively warned to intervene (warn the customer, upgrade shipping, or flag the seller).

## The Results (Time-Based Split, Tested on Future Data)
* **ROC-AUC:** 0.873 
* **Recall:** 67% (We catch 1,032 out of 1,540 late deliveries)
* **Precision:** 19% (We generate 4,437 false alarms)
* **Business Value:** At our optimal threshold of **0.60**, we generate a net positive value of **R$ 29,415** (assuming a missed delay costs R$ 50, and a false alarm costs R$ 5).

## Key AI PM Decisions & Insights
1. **Data Leakage:** We strictly excluded actual delivery dates, review scores, and delivery times from the features, because they are unknown at the moment of checkout.
2. **Time-Based Split:** We trained on orders from 2016-2018 (Train) and tested on the last 3 months (Test), mimicking real-world deployment.
3. **Feature Engineering (The Breakthrough):** Our initial model had an ROC-AUC of 0.44 (worse than a coin flip). We engineered **historical rolling windows** (the historical delay rate of each seller, and the historical delay rate of the customer's zip code). This single change pushed the ROC-AUC to **0.873**.
4. **Threshold Optimization:** We scanned thresholds from 0.10 to 0.90 to maximize business ROI. Moving the default from 0.50 to 0.60 changed the campaign from a money-losing operation to a highly profitable one.

## Limitations
* **Data:** Public Brazilian e-commerce dataset (Olist). Real-world logistics networks may differ.
* **Costs:** The business values (R$50 cost of delay, R$5 cost of false alarm) are assumptions, not hard financial data.
* **Missing Features:** We do not have seller processing times, holiday calendars, or live weather data, which would significantly improve predictions.
* **Monitoring:** While we generated an Evidently data drift report, we have not set up live production monitoring or automated retraining.

## Repository Structure
- `data/` (ignored by git)
- `docs/` (problem brief, PRD, decision log, drift report)
- `src/` (feature engineering, training, evaluation, threshold optimization)
- `README.md`
- `requirements.txt`