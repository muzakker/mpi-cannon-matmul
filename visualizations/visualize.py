# I used Google Colab to generate the figure, so the following is based on the setup of Google Colab

# Upload the CSV file to Google Colab
from google.colab import files
uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
csv_file = list(uploaded.keys())[0]
df = pd.read_csv(csv_file)

# Colors for different matrix sizes and process counts
colors = ['b', 'g', 'r', 'm']
markers = ['o', 's', '^', 'x']

matrix_sizes = [500, 1000, 2000, 4000]
process_counts = [1, 4, 9, 16]

# Plot 1: Execution Time vs Process Count
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title('Execution Time vs Process Count')
ax1.set_xlabel('Process Count')
ax1.set_ylabel('Execution Time (ms)')
ax1.grid(True)

# Loop through each matrix size and plot data
for i, matrix_size in enumerate(matrix_sizes):
    df_subset = df[df['Matrix Size'] == matrix_size]
    
    # Serial data
    serial_data = df_subset[df_subset['Type'] == 'Serial']
    ax1.plot(serial_data['Process Count'], serial_data['Execution Time (ms)'], 
             marker=markers[i], color=colors[i], label=f'Serial {matrix_size}')
    
    # Parallel data
    parallel_data = df_subset[df_subset['Type'] == 'Parallel']
    ax1.plot(parallel_data['Process Count'], parallel_data['Execution Time (ms)'], 
             marker=markers[i], linestyle='--', color=colors[i], label=f'Parallel {matrix_size}')
    
ax1.legend()

# Save Execution Time vs Process Count plot as PNG
fig1.savefig('execution_vs_process_count.png')

# Plot 2: Performance vs Process Count
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.set_title('Performance vs Process Count')
ax2.set_xlabel('Process Count')
ax2.set_ylabel('Performance (GFLOPS)')
ax2.grid(True)

# Loop through each matrix size and plot performance
for i, matrix_size in enumerate(matrix_sizes):
    df_subset = df[df['Matrix Size'] == matrix_size]
    
    # Serial performance
    serial_data = df_subset[df_subset['Type'] == 'Serial']
    serial_performance = (2 * matrix_size ** 3) / serial_data['Execution Time (ms)'] / 1e9
    ax2.plot(serial_data['Process Count'], serial_performance, 
             marker=markers[i], color=colors[i], label=f'Serial {matrix_size}')
    
    # Parallel performance
    parallel_data = df_subset[df_subset['Type'] == 'Parallel']
    parallel_performance = (2 * matrix_size ** 3) / parallel_data['Execution Time (ms)'] / 1e9
    ax2.plot(parallel_data['Process Count'], parallel_performance, 
             marker=markers[i], linestyle='--', color=colors[i], label=f'Parallel {matrix_size}')

ax2.legend()

# Save Performance vs Process Count plot as PNG
fig2.savefig('performance_vs_process_count.png')

# Show the plots (Optional)
plt.show()
