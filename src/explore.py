import duckdb
import pandas as pd

# 1. Connect to DuckDB (this creates an in-memory database)
con = duckdb.connect(database=':memory:')

# 2. Point DuckDB to your CSV files
# This automatically creates a "view" of the data without loading it into memory yet
print("Linking CSV files...")
con.execute("""
    CREATE VIEW orders AS SELECT * FROM read_csv_auto('data/olist_orders_dataset.csv');
    CREATE VIEW customers AS SELECT * FROM read_csv_auto('data/olist_customers_dataset.csv');
    CREATE VIEW order_items AS SELECT * FROM read_csv_auto('data/olist_order_items_dataset.csv');
    CREATE VIEW payments AS SELECT * FROM read_csv_auto('data/olist_order_payments_dataset.csv');
    CREATE VIEW reviews AS SELECT * FROM read_csv_auto('data/olist_order_reviews_dataset.csv');
    CREATE VIEW products AS SELECT * FROM read_csv_auto('data/olist_products_dataset.csv');
    CREATE VIEW sellers AS SELECT * FROM read_csv_auto('data/olist_sellers_dataset.csv');
""")

# 3. Data Quality Check: How many orders are there?
count_query = "SELECT COUNT(*) as total_orders FROM orders"
total_orders = con.execute(count_query).fetchone()[0]
print(f"\nTotal Orders: {total_orders:,}")

# 4. Data Quality Check: How many missing delivery dates?
# (If an order was cancelled, the delivery date is null)
null_query = """
    SELECT 
        COUNT(*) FILTER (WHERE order_delivered_customer_date IS NULL) as missing_delivery_date,
        COUNT(*) FILTER (WHERE order_estimated_delivery_date IS NULL) as missing_estimated_date
    FROM orders
"""
missing_data = con.execute(null_query).fetchdf()
print("\nData Quality Issues (Missing Dates):")
print(missing_data)

# 5. The "Late" Metric: Let's see the baseline delay rate
# We only want to look at orders that were actually delivered
# 5. The "Late" Metric
delay_query = """
    SELECT 
        CASE 
            WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 
            ELSE 0 
        END as is_late
    FROM orders
    WHERE order_delivered_customer_date IS NOT NULL 
      AND order_estimated_delivery_date IS NOT NULL
"""
delay_df = con.execute(delay_query).fetchdf()
late_rate = delay_df['is_late'].mean()

print(f"\nOverall Late Delivery Rate: {late_rate:.2%}")