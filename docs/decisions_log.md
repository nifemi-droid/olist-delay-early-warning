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