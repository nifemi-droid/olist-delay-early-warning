# Decisions Log

## Decision 1: Dropping Orders with Missing Delivery Dates
**Date:** [Today's Date]
**Context:** We queried the orders table and found 99,441 total orders. However, 2,965 orders (about 3%) have a missing `order_delivered_customer_date`. 
**Decision:** We will exclude these orders from model training and evaluation. 
**Why:** If an order has no actual delivery date, it was likely canceled or stuck in transit. We cannot determine if it was "late" relative to the estimated date. Including them would introduce noise and confuse the model.

## Decision 2: The Baseline Challenge (Class Imbalance)
**Context:** After filtering out missing dates, the overall late delivery rate is **8.11%**. 
**Decision:** We will not use accuracy as our primary metric. 
**Why:** A dumb model that predicts "On Time" for every order will achieve 91.89% accuracy, but catch 0% of delays. We must evaluate using Recall (catching delays) and Precision (not wasting ops team time), guided by our Success Metric (Recall ≥ 0.70, Precision constraint).

## Decision 3: Feature Engineering & Model Selection
**Context:** Our first model failed (AUC 0.44) because we only used basic checkout features (weight, price). We engineered two new features: `promised_days` (transit window) and `seller_id` (historical seller performance). 
**Decision:** We switched to Logistic Regression with `class_weight="balanced"` and dropped Random Forest.
**Results:** The AUC jumped to 0.706. We achieved a Recall of 73.9% (catching 755 delays). However, Precision is only 8.8% (7,784 false alarms). 
**Next Step:** We need to optimize the threshold to balance the cost of a false alarm vs. the cost of a missed delay.

## Decision 4: Threshold Optimization & The Business Reality
**Context:** We scanned thresholds from 0.10 to 0.90 using business costs (Cost of a missed delay = R$50, Cost of a false alarm = R$5).
**Decision:** We selected a threshold of **0.60** for production.
**Why:** The default threshold of 0.50 resulted in a negative net value (-R$1,170) because the model generated too many false alarms. At 0.60, the net value is positive (+R$2,855). 
**The Trade-off:** By optimizing for profit, we were forced to accept a Recall of 53% (catching 545 out of 1,021 delays). This missed our PRD goal of 70% recall. To improve this, we would need richer features (e.g., seller historical delay rate, holiday calendars) to increase the model's fundamental predictive power.


## Decision 5: Historical Rolling Features (v3)
**Context:** The v2 model plateaued at AUC 0.706. We built v3 by adding historical rolling windows (the historical delay rate of each seller, and the historical delay rate of each customer region) to prevent data leakage while capturing behavioral patterns.
**Results:** The ROC-AUC jumped to **0.873** and Recall jumped to **87.0%**. The model now catches 1,340 delays (out of 1,540). 
**The Trade-off:** Precision is only 14.1%, meaning we generate 8,146 false alarms. 
**Next Step:** We need to re-run the threshold optimization to find the new profit-maximizing threshold.

## Decision 6: Production Threshold Selection
**Context:** We scanned thresholds to maximize business value (Missed Delay = R$ 50, False Alarm = R$ 5).
**Decision:** We recommend a **production threshold of 0.60**.
**Why:** It yields the maximum net value (R$ 29,415). It catches 1,032 delays (67% recall) while generating 4,437 alerts (which is operationally manageable). Notably, thresholds between 0.60 and 0.75 all produce roughly R$ 28k–R$ 29k in value, giving the ops team flexibility to raise the threshold if they get overwhelmed without destroying ROI.