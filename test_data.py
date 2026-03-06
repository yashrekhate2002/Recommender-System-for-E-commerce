import pandas as pd
import os

# 1. Check file size on disk
file_path = 'ratings_Electronics.csv'
file_size_gb = os.path.getsize(file_path) / (1024**3)

print(f"File size on disk: {file_size_gb:.2f} GB")

# 2. Check memory usage while loading
try:
    print("\nAttempting to load data and checking RAM usage...")
    
    # We load the first 100,000 rows first to estimate total memory
    df_sample = pd.read_csv(file_path, names=['user', 'item', 'rating', 'timestamp'], nrows=100000)
    
    # Calculate memory per row in bytes
    mem_per_row = df_sample.memory_usage(deep=True).sum() / 100000
    estimated_total_gb = (mem_per_row * 1000000) / (1024**3)
    
    print(f"Estimated RAM needed for 10 lakh rows: {estimated_total_gb:.2f} GB")
    
    # 3. Memory Optimization Test
    # Using 'category' for IDs and 'float32' for ratings saves massive space
    print("\nTesting Optimized Loading...")
    df_opt = pd.read_csv(
        file_path, 
        names=['user', 'item', 'rating', 'timestamp'],
        dtype={
            'rating': 'float32',
            'timestamp': 'int32'
        }
    )
    
    actual_mem = df_opt.memory_usage(deep=True).sum() / (1024**2)
    print(f"Actual RAM used for full dataset: {actual_mem:.2f} MB")
    print("Top 5 rows:")
    print(df_opt.head())

except MemoryError:
    print("CRITICAL: Not enough RAM to load the dataset normally.")
except FileNotFoundError:
    print("ERROR: Could not find 'ratings_Electronics.csv'. Make sure the file is in the same folder as this script.")