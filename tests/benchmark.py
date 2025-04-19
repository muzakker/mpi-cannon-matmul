# This was run on Pronto: Interactive Node with 1 Node and 64 Cores CPU
# To run: python3 benchmark.py

import os
import subprocess
import time
import csv

# Parameters for benchmarking
matrix_sizes = [500, 1000, 2000, 4000]
process_counts = [1, 4, 9, 16]

# Paths to the executables
serial_executable = "../src/serial_mult"
parallel_executable = "../src/parallel_mult"

# Output directory for results
results_dir = "../results"
# os.makedirs(results_dir, exist_ok=True)

# CSV file to store the results
csv_file = os.path.join(results_dir, "benchmark_results.csv")

# Initialize CSV file with headers
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Matrix Size", "Process Count", "Execution Time (ms)", "Performance (GFLOPS)", "Type"])

def run_benchmark(matrix_size, process_count):
    # Run and benchmark the serial program
    start_time = time.time()
    serial_result = subprocess.run([serial_executable, str(matrix_size)], capture_output=True, text=True)
    serial_duration = time.time() - start_time

    serial_output_file = os.path.join(results_dir, f"serial_{matrix_size}.txt")
    with open(serial_output_file, "w") as file:
        file.write(serial_result.stdout)
        file.write(f"\nSerial execution time: {serial_duration * 1000:.2f} ms\n")

    # Calculate performance metrics for serial
    flops = 2.0 * matrix_size * matrix_size * matrix_size
    serial_gflops = (flops / serial_duration) / 1e9

    with open(serial_output_file, "a") as file:
        file.write(f"Serial Performance: {serial_gflops:.2f} GFLOPS\n")

    # Append results to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([matrix_size, 1, serial_duration * 1000, serial_gflops, "Serial"])

    # Run and benchmark the parallel program
    start_time = time.time()
    parallel_result = subprocess.run(["mpirun", "-np", str(process_count), parallel_executable, str(matrix_size)], capture_output=True, text=True)
    parallel_duration = time.time() - start_time

    parallel_output_file = os.path.join(results_dir, f"parallel_{matrix_size}_{process_count}.txt")
    with open(parallel_output_file, "w") as file:
        file.write(parallel_result.stdout)
        file.write(f"\nParallel execution time: {parallel_duration * 1000:.2f} ms\n")

    # Calculate performance metrics for parallel
    parallel_gflops = (flops / parallel_duration) / 1e9

    with open(parallel_output_file, "a") as file:
        file.write(f"Parallel Performance: {parallel_gflops:.2f} GFLOPS\n")

    # Append results to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([matrix_size, process_count, parallel_duration * 1000, parallel_gflops, "Parallel"])

# Run benchmarks for each matrix size and process count
for size in matrix_sizes:
    run_benchmark(size, 1)  # Serial run for each matrix size
    for count in process_counts:
        if count <= os.cpu_count():  # Ensure the process count does not exceed available CPUs
            run_benchmark(size, count)  # Parallel run for each matrix size and process count
