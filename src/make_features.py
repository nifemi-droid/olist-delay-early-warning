import duckdb

print("Building Advanced Feature Table (v3)...")
con = duckdb.connect(database=':memory:')

# We break this into two steps to avoid DuckDB errors
# Step 1: Create the base table with window functions
base_query = """
    CREATE TABLE raw_features AS
    WITH base_orders AS (
        SELECT 
            o.order_id,
            o.customer_id,
            oi.seller_id,
            TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) as purchase_time,
            TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) as est_delivery,
            TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) as actual_delivery,
            CASE 
                WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) > TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) THEN 1 
                ELSE 0 
            END as is_late
        FROM read_csv_auto('data/olist_orders_dataset.csv') o
        LEFT JOIN read_csv_auto('data/olist_order_items_dataset.csv') oi ON o.order_id = oi.order_id
        WHERE o.order_delivered_customer_date IS NOT NULL 
          AND o.order_estimated_delivery_date IS NOT NULL
    )
    SELECT 
        bo.order_id,
        bo.purchase_time,
        bo.est_delivery,
        bo.is_late,
        date_diff('day', bo.purchase_time, bo.est_delivery) as promised_days,
        EXTRACT(dayofweek FROM bo.purchase_time) as purchase_day_of_week,
        EXTRACT(month FROM bo.purchase_time) as purchase_month,
        
        -- Window functions
        AVG(is_late) OVER (
            PARTITION BY seller_id 
            ORDER BY purchase_time 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) as seller_delay_rate,
        
        AVG(is_late) OVER (
            PARTITION BY customer_id 
            ORDER BY purchase_time 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) as customer_region_delay_rate,
        
        oi.price as item_price,
        oi.freight_value as item_freight,
        c.customer_state,
        p.product_weight_g,
        p.product_category_name
    FROM base_orders bo
    LEFT JOIN read_csv_auto('data/olist_order_items_dataset.csv') oi ON bo.order_id = oi.order_id
    LEFT JOIN read_csv_auto('data/olist_customers_dataset.csv') c ON bo.customer_id = c.customer_id
    LEFT JOIN read_csv_auto('data/olist_products_dataset.csv') p ON oi.product_id = p.product_id
"""

print("Executing base query (this takes ~10 seconds)...")
con.execute(base_query)

# Step 2: Aggregate the table and save to CSV
print("Aggregating and saving...")
final_query = """
    COPY (
        SELECT 
            order_id,
            purchase_time,
            est_delivery,
            is_late,
            promised_days,
            purchase_day_of_week,
            purchase_month,
            COALESCE(seller_delay_rate, 0.08) as seller_delay_rate,
            COALESCE(customer_region_delay_rate, 0.08) as customer_region_delay_rate,
            SUM(item_price) as total_price,
            SUM(item_freight) as total_freight,
            COUNT(order_id) as num_items,
            customer_state,
            AVG(product_weight_g) as avg_weight_g,
            MAX(product_category_name) as primary_category
        FROM raw_features
        GROUP BY 
            order_id, purchase_time, est_delivery, is_late, promised_days, 
            purchase_day_of_week, purchase_month, seller_delay_rate, 
            customer_region_delay_rate, customer_state
    ) TO 'data/features.csv' (HEADER, DELIMITER ',')
"""
con.execute(final_query)

print("Success! v3 Advanced Feature table saved.")

# Check correlations
corr_query = """
    SELECT 
        corr(seller_delay_rate, is_late) as corr_seller_history,
        corr(customer_region_delay_rate, is_late) as corr_region_history,
        corr(promised_days, is_late) as corr_promised_days
    FROM read_csv_auto('data/features.csv')
"""
print("\nNew Correlations with 'is_late':")
print(con.execute(corr_query).fetchdf().to_string(index=False))