import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# 1. LOAD DATA
print("Step 1: Loading data...")
cols = ['user_id', 'product_id', 'rating', 'timestamp']
df = pd.read_csv('ratings_Electronics.csv', names=cols, dtype={'rating': 'float32'})

# 2. AGGRESSIVE FILTERING (To fit in RAM)
print("Step 2: Filtering data...")

# Keep only products with at least 100 ratings
product_counts = df['product_id'].value_counts()
popular_products = product_counts[product_counts >= 100].index

# Keep only users who have given at least 50 ratings
user_counts = df['user_id'].value_counts()
active_users = user_counts[user_counts >= 50].index

# Apply both filters
df_filtered = df[df['product_id'].isin(popular_products) & df['user_id'].isin(active_users)]

print(f"Filtered down to {df_filtered.shape[0]} rows.")

# 3. CREATE ITEM-USER MATRIX
print("Step 3: Creating Item-User Matrix...")
# This will be MUCH smaller now
item_user_matrix = df_filtered.pivot_table(index='product_id', columns='user_id', values='rating').fillna(0)

# 4. TRAINING SVD
print("Step 4: Compressing data with SVD...")
# We use SVD to find the 'essence' of the products
SVD = TruncatedSVD(n_components=10, random_state=42)
decomposed_matrix = SVD.fit_transform(item_user_matrix)

# 5. CALCULATE CORRELATION
print("Step 5: Calculating product similarities...")
correlation_matrix = np.corrcoef(decomposed_matrix)

# 6. RECOMMENDATION FUNCTION
def get_recommendations(product_id):
    product_names = list(item_user_matrix.index)
    if product_id not in product_names:
        return "Product ID not in the filtered model. Try another."
        
    product_idx = product_names.index(product_id)
    similarity_row = correlation_matrix[product_idx]
    
    # Get top 5 similar items
    similar_indices = similarity_row.argsort()[-6:-1][::-1]
    
    print(f"\nSince you looked at Product: {product_id}")
    print("Recommended Products:")
    for i in similar_indices:
        print(f"- {product_names[i]} (Score: {similarity_row[i]:.2f})")

# 7. TEST
print("\n--- MODEL READY ---")
if not item_user_matrix.empty:
    test_id = item_user_matrix.index[0] 
    get_recommendations(test_id)
else:
    print("Filter was too strict! No data left. Lower the numbers in Step 2.")