# Problem Brief: Order Delay Early-Warning System

## 1. The Problem
At the moment an order is placed, we do not know if it will arrive late. Currently, we only find out after the promised delivery date has passed. By then, the customer is frustrated, and the operations team is reacting to complaints rather than preventing them.

## 2. The Action
We will build a predictive model that scores an order's risk of delay **at the moment of checkout**. If an order is flagged as high-risk, the operations team will take one of three actions:
1. **Proactively warn the customer** (setting realistic expectations).
2. **Upgrade the shipping option** to expedite delivery.
3. **Flag the seller** if the bottleneck is on their end.

## 3. The Costs (The Trade-off)
*   **False Positive (Wasted Action):** We spend money on a shipping upgrade or customer service outreach for an order that would have arrived on time anyway. This also risks annoying a customer with unnecessary warnings.
*   **False Negative (Missed Delay):** We do nothing, the order arrives late, and we suffer the consequences: refunds, chargebacks, and damaging 1-star reviews on the platform.

## 4. Success Metric
We will evaluate the model against the following business constraints:
*   **Target:** Catch at least **70% of late deliveries** (Recall ≥ 0.70).
*   **Constraint:** Do not flag more than **20% of on-time orders** (Precision constraint).

## 5. Definition of "Late"
To ensure our metrics are mathematically precise, "Late" is defined as:
`actual_delivery_date > estimated_delivery_date`