import duckdb

con = duckdb.connect(database=':memory:')

print("Building the Feature Table...")

# This query joins the tables and creates the features we know AT CHECKOUT
feature_query = """
    CREATE TABLE features AS
    SELECT 
        o.order_id,
        o.order_purchase_timestamp,
        o.order_estimated_delivery_date,
        
        -- Our Target (1 = Late, 0 = On Time)
        CASE 
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
            ELSE 0 
        END as is_late,
        
        -- Features from Order Items (Freight and Price)
        SUM(oi.price) as total_price,
        SUM(oi.freight_value) as total_freight,
        COUNT(oi.order_item_id) as num_items,
        
        -- Features from Customer (Location)
        c.customer_state,
        
        -- Features from Products (Physical characteristics)
        AVG(p.product_weight_g) as avg_weight_g,
        MAX(p.product_category_name) as primary_category

    FROM read_csv_auto('data/olist_orders_dataset.csv') o
    LEFT JOIN read_csv_auto('data/olist_order_items_dataset.csv') oi 
        ON o.order_id = oi.order_id
    LEFT JOIN read_csv_auto('data/olist_customers_dataset.csv') c 
        ON o.customer_id = c.customer_id
    LEFT JOIN read_csv_auto('data/olist_products_dataset.csv') p 
        ON oi.product_id = p.product_id
        
    -- We only want delivered orders
    WHERE o.order_delivered_customer_date IS NOT NULL 
      AND o.order_estimated_delivery_date IS NOT NULL
      
    GROUP BY 
        o.order_id, 
        o.order_purchase_timestamp, 
        o.order_estimated_delivery_date, 
        o.order_delivered_customer_date,
        c.customer_state
"""

con.execute(feature_query)

# Save the final table to a CSV so our model can use it later
con.execute("COPY features TO 'data/features.csv' (HEADER, DELIMITER ',')")

print("Success! Feature table saved to data/features.csv")

# Let's verify by looking at the first 3 rows
print("\nSample of the new Feature Table:")
print(con.execute("SELECT * FROM features LIMIT 3").fetchdf().to_string())