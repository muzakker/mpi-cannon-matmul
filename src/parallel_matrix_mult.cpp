// It was run on Pronto: Interactive Node with 1 Node and 64 Cores CPU
// To replicate the results, you need to load openmpi module
// Use "module load openmpi" to load the MPI module
// To compile: mpic++ -O3 parallel_matrix_mult.cpp -o parallel_mult
// To run with 4 processes: mpirun -np 4 ./src/parallel_mult
// To run with 9 processes: mpirun -np 9 ./src/parallel_mult

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <random>

class MatrixBlock {
private:
    std::vector<double> data;
    int block_size;

public:
    MatrixBlock(int size) : block_size(size), data(size * size, 0.0) {}

    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (auto& element : data) {
            element = dis(gen);
        }
    }

    double& at(int i, int j) { return data[i * block_size + j]; }
    const double& at(int i, int j) const { return data[i * block_size + j]; }
    int size() const { return block_size; }
};

class CannonMPI {
private:
    MPI_Comm cart_comm;    // Cartesian communicator
    int rank, p;           // Rank and total processes
    int grid_size;         // Grid dimensions (sqrt(p))
    int coords[2];         // Process coordinates in grid
    int block_size;        // Size of local matrix block

public:
    CannonMPI(int matrix_size) {
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        
        // Check if we can create a square grid
        grid_size = static_cast<int>(sqrt(p));
        if (grid_size * grid_size != p) {
            throw std::runtime_error("Number of processes must be a perfect square");
        }
        
        // Check if matrix size is divisible by grid size
        if (matrix_size % grid_size != 0) {
            throw std::runtime_error("Matrix size must be divisible by grid size");
        }
        
        block_size = matrix_size / grid_size;
        
        // Create 2D cartesian grid
        int dims[2] = {grid_size, grid_size};
        int periods[2] = {1, 1};  // Periodic boundaries for shifting
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
        
        MPI_Comm_rank(cart_comm, &rank);
        MPI_Cart_coords(cart_comm, rank, 2, coords);
    }

    void multiply(MatrixBlock& A_block, MatrixBlock& B_block, MatrixBlock& C_block) {
        // Initial alignment
        int shift_src, shift_dst;
        MPI_Cart_shift(cart_comm, 1, -coords[0], &shift_src, &shift_dst);
        shift_matrix(A_block, shift_src, shift_dst);
        
        MPI_Cart_shift(cart_comm, 0, -coords[1], &shift_src, &shift_dst);
        shift_matrix(B_block, shift_src, shift_dst);

        // Main computation loop
        for (int k = 0; k < grid_size; ++k) {
            // Local multiplication
            multiply_blocks(A_block, B_block, C_block);
            
            // Shift A left by 1
            MPI_Cart_shift(cart_comm, 1, -1, &shift_src, &shift_dst);
            shift_matrix(A_block, shift_src, shift_dst);
            
            // Shift B up by 1
            MPI_Cart_shift(cart_comm, 0, -1, &shift_src, &shift_dst);
            shift_matrix(B_block, shift_src, shift_dst);
        }
    }

private:
    void multiply_blocks(const MatrixBlock& A, const MatrixBlock& B, MatrixBlock& C) {
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                double sum = 0.0;
                for (int k = 0; k < block_size; ++k) {
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) += sum;
            }
        }
    }

    void shift_matrix(MatrixBlock& block, int src, int dst) {
        std::vector<double> send_buf(block.size() * block.size());
        std::vector<double> recv_buf(block.size() * block.size());
        
        // Pack data
        for (int i = 0; i < block.size(); ++i) {
            for (int j = 0; j < block.size(); ++j) {
                send_buf[i * block.size() + j] = block.at(i, j);
            }
        }
        
        MPI_Sendrecv(send_buf.data(), block.size() * block.size(), MPI_DOUBLE, dst, 0,
                     recv_buf.data(), block.size() * block.size(), MPI_DOUBLE, src, 0,
                     cart_comm, MPI_STATUS_IGNORE);
        
        // Unpack data
        for (int i = 0; i < block.size(); ++i) {
            for (int j = 0; j < block.size(); ++j) {
                block.at(i, j) = recv_buf[i * block.size() + j];
            }
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    
    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    const int N = std::stoi(argv[1]);  // Matrix size
    
    try {
        CannonMPI cannon(N);
        int block_size = N / static_cast<int>(sqrt(p));
        
        MatrixBlock A_block(block_size);
        MatrixBlock B_block(block_size);
        MatrixBlock C_block(block_size);
        
        // Initialize local blocks
        A_block.randomize();
        B_block.randomize();
        
        // Start timing
        double start_time = MPI_Wtime();
        
        // Perform multiplication
        cannon.multiply(A_block, B_block, C_block);
        
        // End timing
        double end_time = MPI_Wtime();
        
        if (rank == 0) {
            std::cout << "Parallel execution time: " << (end_time - start_time) * 1000 << " ms\n";
            double gflops = (2.0 * N * N * N) / ((end_time - start_time) * 1e9);
            std::cout << "Performance: " << gflops << " GFLOPS\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error on rank " << rank << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}
