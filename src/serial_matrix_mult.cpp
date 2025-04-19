// To compile: g++ -O3 -std=c++11 serial_matrix_mult.cpp -o src/serial_mult
// To run: ./src/serial_mult

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <stdexcept>

// Matrix class definition to handle matrix operations and data
class Matrix {
private:
    std::vector<double> data; // 1D vector to store matrix elements
    size_t rows, cols; // Dimensions of the matrix (rows x cols)

public:
    // Constructor to initialize a matrix with given dimensions
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}

    // Randomize matrix elements with values between 0 and 1
    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (auto& element : data) {
            element = dis(gen);
        }
    }

    // Accessor for matrix element at row i, column j
    double& at(size_t i, size_t j) {
        return data[i * cols + j];
    }

    // Const accessor for matrix element at row i, column j
    const double& at(size_t i, size_t j) const {
        return data[i * cols + j];
    }

    // Return number of rows in the matrix
    size_t getRows() const { return rows; }

    // Return number of columns in the matrix
    size_t getCols() const { return cols; }

    // Print a preview of the top-left part of the matrix (e.g., 3x3)
    void print_preview(size_t preview_size = 3) const {
        preview_size = std::min(preview_size, std::min(rows, cols)); // Avoid going beyond matrix bounds
        for (size_t i = 0; i < preview_size; ++i) {
            for (size_t j = 0; j < preview_size; ++j) {
                std::cout << std::fixed << std::setprecision(4) << at(i, j) << " "; // Print each element
            }
            std::cout << std::endl;
        }
        if (preview_size < rows || preview_size < cols) {
            std::cout << "..." << std::endl; // Indicate more data if preview is truncated
        }
    }

    // Calculate the Frobenius norm of the matrix
    double frobenius_norm() const {
        double sum = 0.0;
        for (const auto& val : data) {
            sum += val * val; 
        }
        return std::sqrt(sum);
    }
};

// Function to multiply two matrices A and B serially
Matrix multiply_serial(const Matrix& A, const Matrix& B) {
    if (A.getCols() != B.getRows()) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication"); // Ensure compatibility for multiplication
    }

    Matrix C(A.getRows(), B.getCols()); // Result matrix C with appropriate dimensions
    
    // Perform matrix multiplication
    for (size_t i = 0; i < A.getRows(); ++i) {
        for (size_t j = 0; j < B.getCols(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A.getCols(); ++k) {
                sum += A.at(i, k) * B.at(k, j); // Multiply and accumulate
            }
            C.at(i, j) = sum; // Assign the computed value to the result matrix
        }
    }
    
    return C; 
}

int main(int argc, char** argv) {
    // Check for correct argument count (matrix size)
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>\n"; // Show usage if incorrect arguments
        return 1;
    }

    try {
        // Read matrix size from command-line argument
        const size_t N = std::stoi(argv[1]);
        
        std::cout << "\n=== Serial Matrix Multiplication ===\n";
        std::cout << "Matrix dimensions: " << N << " x " << N << "\n\n";
        
        // Initialize two NxN matrices A and B
        Matrix A(N, N);
        Matrix B(N, N);
        
        std::cout << "Initializing matrices with random values...\n";
        A.randomize(); // Randomize matrix A
        B.randomize(); // Randomize matrix B
        
        // Print previews of matrices A and B (top-left 3x3 elements)
        std::cout << "\nMatrix A preview (top-left corner):\n";
        A.print_preview();
        
        std::cout << "\nMatrix B preview (top-left corner):\n";
        B.print_preview();
        
        std::cout << "\nPerforming multiplication...\n";
        
        // Measure the time taken for multiplication
        auto start = std::chrono::high_resolution_clock::now();
        Matrix C = multiply_serial(A, B); // Perform matrix multiplication
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate the duration of multiplication
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Print the result matrix preview (top-left 3x3 elements)
        std::cout << "\nResult matrix C preview (top-left corner):\n";
        C.print_preview();
        
        // Performance Metrics: Time and GFLOPS (Giga FLOPS)
        std::cout << "\nPerformance Metrics:\n";
        std::cout << "- Multiplication time: " << duration.count() << " ms\n";
        std::cout << "- Matrix size: " << N << " x " << N << "\n";
        
        double flops = 2.0 * N * N * N; // Floating point operations: 2 * N^3 for matrix multiplication
        double gflops = (flops / duration.count()) / 1e6; // Convert to GFLOPS (Giga FLOPS)
        std::cout << "- Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
        
        // Frobenius norms for verification (to check matrix consistency)
        std::cout << "\nVerification Metrics (Frobenius norms):\n";
        std::cout << "||A|| = " << std::fixed << std::setprecision(4) << A.frobenius_norm() << "\n";
        std::cout << "||B|| = " << B.frobenius_norm() << "\n";
        std::cout << "||C|| = " << C.frobenius_norm() << "\n";

    } catch (const std::exception &e) {
        // Catch any exceptions (e.g., invalid matrix size or memory issues)
        std::cerr << "An error occurred: " << e.what() << '\n';
        return 1;
    }

    return 0; 
}
