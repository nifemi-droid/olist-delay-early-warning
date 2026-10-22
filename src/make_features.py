import duckdb

print("Building Advanced Feature Table (v3)...")
con = duckdb.connect(database=':memory:')

feature_query = """
    WITH base_orders AS (
        SELECT 
            o.order_id,
            o.customer_id,
            oi.seller_id,
            TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) as purchase_time,
            TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) as est_delivery,
            TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) as actual_delivery,
            
            -- Target
            CASE 
                WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) > TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) THEN 1 
                ELSE 0 
            END as is_late
            
        FROM read_csv_auto('data/olist_orders_dataset.csv') o
        LEFT JOIN read_csv_auto('data/olist_order_items_dataset.csv') oi 
            ON o.order_id = oi.order_id
        WHERE o.order_delivered_customer_date IS NOT NULL 
          AND o.order_estimated_delivery_date IS NOT NULL
    ),

    -- Calculate historical seller delay rate (Window Function!)
    seller_history AS (
        SELECT 
            order_id,
            seller_id,
            AVG(is_late) OVER (
                PARTITION BY seller_id 
                ORDER BY purchase_time 
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) as seller_historic_delay_rate
        FROM base_orders
    ),

    -- Calculate historical customer (zip code) delay rate
    customer_history AS (
        SELECT 
            bo.order_id,
            AVG(bo.is_late) OVER (
                PARTITION BY c.customer_zip_code_prefix 
                ORDER BY bo.purchase_time 
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) as customer_region_delay_rate
        FROM base_orders bo
        JOIN read_csv_auto('data/olist_customers_dataset.csv') c 
            ON bo.customer_id = c.customer_id
    )

    SELECT 
        bo.order_id,
        bo.purchase_time,
        bo.est_delivery,
        bo.is_late,
        
        -- Time based features
        date_diff('day', bo.purchase_time, bo.est_delivery) as promised_days,
        EXTRACT(dayofweek FROM bo.purchase_time) as purchase_day_of_week,
        EXTRACT(month FROM bo.purchase_time) as purchase_month,
        
        -- Historic features (fill nulls with 0.08 which is the global average)
        COALESCE(sh.seller_historic_delay_rate, 0.08) as seller_delay_rate,
        COALESCE(ch.customer_region_delay_rate, 0.08) as customer_region_delay_rate,
        
        -- Aggregated features
        COALESCE(SUM(TRY_CAST(oi.price AS DOUBLE)), 0) as total_price,
        COALESCE(SUM(TRY_CAST(oi.freight_value AS DOUBLE)), 0) as total_freight,
        COUNT(oi.order_item_id) as num_items,
        c.customer_state,
        COALESCE(AVG(TRY_CAST(p.product_weight_g AS DOUBLE)), 0) as avg_weight_g,
        COALESCE(MAX(p.product_category_name), 'unknown') as primary_category

    FROM base_orders bo
    LEFT JOIN read_csv_auto('data/olist_order_items_dataset.csv') oi ON bo.order_id = oi.order_id
    LEFT JOIN read_csv_auto('data/olist_customers_dataset.csv') c ON bo.customer_id = c.customer_id
    LEFT JOIN read_csv_auto('data/olist_products_dataset.csv') p ON oi.product_id = p.product_id
    LEFT JOIN seller_history sh ON bo.order_id = sh.order_id
    LEFT JOIN customer_history ch ON bo.order_id = ch.order_id
    
    GROUP BY 
        bo.order_id, bo.purchase_time, bo.est_delivery, bo.is_late, 
        sh.seller_historic_delay_rate, ch.customer_region_delay_rate,
        c.customer_state
"""

con.execute(feature_query)
con.execute("COPY features TO 'data/features.csv' (HEADER, DELIMITER ',')")
print("Success! v3 Advanced Feature table saved.")

# Let's see if these new features correlate better
corr_query = """
    SELECT 
        corr(seller_delay_rate, is_late) as corr_seller_history,
        corr(customer_region_delay_rate, is_late) as corr_region_history,
        corr(promised_days, is_late) as corr_promised_days
    FROM features
"""
print("\nNew Correlations with 'is_late':")
print(con.execute(corr_query).fetchdf().to_string(index=False))