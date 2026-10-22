import duckdb

print("Building the Feature Table (v2)...")

con = duckdb.connect(database=':memory:')

feature_query = """
    CREATE TABLE features AS
    SELECT 
        o.order_id,
        o.order_purchase_timestamp,
        o.order_estimated_delivery_date,
        
        -- Target
        CASE 
            WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) > TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) THEN 1 
            ELSE 0 
        END as is_late,
        
        -- NEW FEATURE: Promised transit time in days
        date_diff('day', TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP), TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP)) as promised_days,
        
        -- NEW FEATURE: Distance proxy (Zip code difference)
        ABS(TRY_CAST(c.customer_zip_code_prefix AS INTEGER) - TRY_CAST(s.seller_zip_code_prefix AS INTEGER)) as zip_distance,
        
        -- NEW FEATURE: Seller ID (Some sellers are just bad at shipping)
        oi.seller_id,
        
        -- Old features
        COALESCE(SUM(TRY_CAST(oi.price AS DOUBLE)), 0) as total_price,
        COALESCE(SUM(TRY_CAST(oi.freight_value AS DOUBLE)), 0) as total_freight,
        COUNT(oi.order_item_id) as num_items,
        c.customer_state,
        COALESCE(AVG(TRY_CAST(p.product_weight_g AS DOUBLE)), 0) as avg_weight_g,
        COALESCE(MAX(p.product_category_name), 'unknown') as primary_category

    FROM read_csv_auto('data/olist_orders_dataset.csv') o
    LEFT JOIN read_csv_auto('data/olist_order_items_dataset.csv') oi 
        ON o.order_id = oi.order_id
    LEFT JOIN read_csv_auto('data/olist_customers_dataset.csv') c 
        ON o.customer_id = c.customer_id
    LEFT JOIN read_csv_auto('data/olist_sellers_dataset.csv') s
        ON oi.seller_id = s.seller_id
    LEFT JOIN read_csv_auto('data/olist_products_dataset.csv') p 
        ON oi.product_id = p.product_id
        
    WHERE o.order_delivered_customer_date IS NOT NULL 
      AND o.order_estimated_delivery_date IS NOT NULL
      
    GROUP BY 
        o.order_id, 
        o.order_purchase_timestamp, 
        o.order_estimated_delivery_date, 
        o.order_delivered_customer_date,
        c.customer_state,
        c.customer_zip_code_prefix,
        s.seller_zip_code_prefix,
        oi.seller_id
"""

con.execute(feature_query)
con.execute("COPY features TO 'data/features.csv' (HEADER, DELIMITER ',')")

print("Success! v2 Feature table saved.")

# Check the correlation of the new features with is_late
corr_query = """
    SELECT 
        corr(promised_days, is_late) as corr_promised_days,
        corr(zip_distance, is_late) as corr_zip_distance
    FROM features
"""
print("\nCorrelation with 'is_late':")
print(con.execute(corr_query).fetchdf().to_string(index=False))