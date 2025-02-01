#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stack>
#include <tuple>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <unordered_map>

#define BOARD_SIZE 7 // Board size is 7x7
#define NUM_THREADS 256
#define MAX_BOARDS 10000000
#define BOARDS_FILENAME "gpu_solution.txt"
#define HASH_FILENAME "gpu_hash.txt"
#define INITIAL_BOARD_FILENAME "initial_board.txt"

// Macro to check CUDA errors
#define CUDA_CHECK(call)                                                                 \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// Board representation constants
#define OUT_OF_BOUNDS 2
#define PEG 1
#define EMPTY 0

// Valid cells on the board for decoding purposes
const bool valid_cells[BOARD_SIZE * BOARD_SIZE] = {
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
    1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
};

// Function to decode the hash into a board
__host__ int* decode_board(uint64_t hash)
{
    int *board = (int *)malloc(BOARD_SIZE * BOARD_SIZE * sizeof(int));
    if (!board)  // Check if allocation was successful
    {
        fprintf(stderr, "Memory allocation failed in decode_board()\n");
        exit(EXIT_FAILURE);
    }

    for (int i = BOARD_SIZE * BOARD_SIZE - 1; i >= 0; --i)
    {
        board[i] = OUT_OF_BOUNDS;
        if (valid_cells[i])
        {
            board[i] = hash % 3;
            hash /= 3;
        }
    }
    return board;  // Valid because board is allocated on the heap
}

// Function to read the initial board from a file
__host__ int* read_board_from_file()
{
    FILE *file = fopen(INITIAL_BOARD_FILENAME, "r");
    uint64_t hash;
    if(fscanf(file, "%lu", &hash) != 1)
    {
        fprintf(stderr, "Failed to read the initial board from file\n");
        exit(EXIT_FAILURE);
    }
    int *board = decode_board(hash);
    fclose(file);
    return board;
}

// A move is valid if there is a peg at (x, y) and an empty space at (x + 2*dx, y + 2*dy) and a peg at (x + dx, y + dy)
__device__ __host__ bool is_valid_move(int *board, int x, int y, int dx, int dy)
{
    int nx = x + dx * 2;
    int ny = y + dy * 2;
    int mx = x + dx;
    int my = y + dy;
    if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE ||
        mx < 0 || mx >= BOARD_SIZE || my < 0 || my >= BOARD_SIZE)
    {
        return false;
    }
    return (board[x * BOARD_SIZE + y] == PEG &&
            board[mx * BOARD_SIZE + my] == PEG &&
            board[nx * BOARD_SIZE + ny] == EMPTY);
}

// Function to apply a move on the board
__device__ __host__ void apply_move(int *board, int x, int y, int dx, int dy)
{
    int nx = x + dx * 2;
    int ny = y + dy * 2;
    int mx = x + dx;
    int my = y + dy;
    board[x * BOARD_SIZE + y] = EMPTY;
    board[mx * BOARD_SIZE + my] = EMPTY;
    board[nx * BOARD_SIZE + ny] = PEG;
}

// Function to print the board
__host__ void print_board(int *board)
{
    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        for (int j = 0; j < BOARD_SIZE; ++j)
        {
            if (board[i * BOARD_SIZE + j] == OUT_OF_BOUNDS)
            {
                printf(" ");
            }
            else if (board[i * BOARD_SIZE + j] == PEG)
            {
                printf("1");
            }
            else
            {
                printf("0");
            }
        }
        printf("\n");
    }
    printf("\n");
}

// Function to print the board to a file
__host__ void print_board_to_file(int *board)
{
    // Open the file
    FILE *file = fopen(BOARDS_FILENAME, "a");
    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        for (int j = 0; j < BOARD_SIZE; ++j)
        {
            if (board[i * BOARD_SIZE + j] == OUT_OF_BOUNDS)
            {
                fprintf(file, " ");
            }
            else if (board[i * BOARD_SIZE + j] == PEG)
            {
                fprintf(file, "1");
            }
            else
            {
                fprintf(file, "0");
            }
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);
}

// Function to print the hash to a file
__host__ void print_hash_to_file(uint64_t hash)
{
    FILE *file = fopen(HASH_FILENAME, "a");
    fprintf(file, "%lu\n", hash);
    fclose(file);
}

// Function to encode the board into a hash
__device__ __host__ uint64_t encode_board(const int *board)
{
    uint64_t hash = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
    {
        if (board[i] != OUT_OF_BOUNDS)
        {
            hash = hash * 3 + board[i]; // Base 3 encoding for valid cells only
        }
    }
    return hash;
}

// Function to retrace moves taken from the initail board to the solution board
void retrace_solution(uint64_t solution_hash, std::unordered_map<uint64_t, uint64_t> &board_map)
{
    std::stack<uint64_t> solution_stack;
    while (solution_hash != (uint64_t)-1)
    {
        solution_stack.push(solution_hash);
        solution_hash = board_map[solution_hash];
    }

    // remove the file if it exists
    FILE *board_file = fopen(BOARDS_FILENAME, "w");
    fclose(board_file);

    FILE *hash_file = fopen(HASH_FILENAME, "w");
    fclose(hash_file);

    while (!solution_stack.empty())
    {
        uint64_t original_hash = solution_stack.top();
        uint64_t hash = original_hash;
        solution_stack.pop();

        int *board = decode_board(hash);
        print_hash_to_file(original_hash);
        print_board_to_file(board);
    }
    
}

// Kernel to process the boards
__global__ void process_boards(
    int *current_boards,
    uint64_t *current_hashes,
    int *next_boards,
    uint64_t *next_hashes,
    uint64_t *next_parents,
    int *next_count,
    int *solution_found,
    int *solution_idx,
    int num_current_boards,
    int *statistic_analyzed_boards,
    int *statistic_pegs_taken,
    int *statistic_boards_without_moves)
{
    // Get the index of the current board
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is valid
    if (idx >= num_current_boards || *solution_found)
        return;

    // Copy the board to be processed
    int board[BOARD_SIZE * BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
        board[i] = current_boards[idx * BOARD_SIZE * BOARD_SIZE + i];

    // Increment the count of analyzed boards
    atomicAdd(statistic_analyzed_boards, 1);

    bool moves_available = false;

    // For each cell that is a peg check if a move is possible
    for (int x = 0; x < BOARD_SIZE; ++x)
    {
        for (int y = 0; y < BOARD_SIZE; ++y)
        {
            if (board[x * BOARD_SIZE + y] == PEG)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        if (abs(dx) != abs(dy) && is_valid_move(board, x, y, dx, dy))
                        {
                            // Move is valid

                            moves_available = true;
                            int new_board[BOARD_SIZE * BOARD_SIZE];
                            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
                                new_board[i] = board[i];
                            // Aply the move thus generating a new board
                            apply_move(new_board, x, y, dx, dy);

                            // Increment the count of the next boards, if the count exceeds the maximum number of boards return
                            int new_board_idx = atomicAdd(next_count, 1);
                            if (new_board_idx >= MAX_BOARDS)
                            {
                                atomicAdd(next_count, -1);
                                return;
                            }

                            // Copy the new board to the next boards array
                            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
                                next_boards[new_board_idx * BOARD_SIZE * BOARD_SIZE + i] = new_board[i];

                            // Encode the new board to a hash
                            // Store the hash and the parent hash for retracing the solution
                            next_hashes[new_board_idx] = encode_board(new_board);
                            next_parents[new_board_idx] = current_hashes[idx];

                            // Increment the count of pegs taken
                            atomicAdd(statistic_pegs_taken, 1);

                            // Check if the new board has only one peg left
                            int peg_count = 0;
                            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
                            {
                                if (new_board[i] == PEG)
                                    peg_count++;
                            }

                            // If the new board has only one peg left and no solution has been found yet
                            // Set the solution found flag and the solution index
                            if (peg_count == 1 && atomicCAS(solution_found, 0, 1) == 0)
                            {
                                *solution_idx = new_board_idx;
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    // If no moves are available increment the count of boards without moves
    if (!moves_available)
    {
        atomicAdd(statistic_boards_without_moves, 1);
    }
}

// Function to solve the peg solitaire problem
void host_solve(int *initial_board)
{
    std::unordered_map<uint64_t, uint64_t> board_map;
    int *d_current_boards;
    uint64_t *d_current_hashes;
    int *d_next_boards;
    uint64_t *d_next_hashes;
    uint64_t *d_next_parents;
    int *d_next_count;
    int *d_solution_found;
    int *d_solution_idx;
    int *d_pegs_taken;
    int *d_statistic_analyzed_boards;
    int *d_statistic_pegs_taken;
    int *d_statistic_boards_without_moves;

    // Alocate memory for the next boards hashes and parents for filtering
    uint64_t *next_hashes = (uint64_t *)malloc(MAX_BOARDS * sizeof(uint64_t));
    uint64_t *next_parents = (uint64_t *)malloc(MAX_BOARDS * sizeof(uint64_t));
    int *filtered_boards = (int *)malloc(MAX_BOARDS * BOARD_SIZE * BOARD_SIZE * sizeof(int));
    uint64_t *filtered_hashes = (uint64_t *)malloc(MAX_BOARDS * sizeof(uint64_t));
    int *next_boards = (int *)malloc(MAX_BOARDS * BOARD_SIZE * BOARD_SIZE * sizeof(int));

    // Allocate memory on the device
    std::chrono::high_resolution_clock::time_point allocation_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_current_boards, MAX_BOARDS * BOARD_SIZE * BOARD_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_current_hashes, MAX_BOARDS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_next_boards, MAX_BOARDS * BOARD_SIZE * BOARD_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_hashes, MAX_BOARDS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_next_parents, MAX_BOARDS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_next_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_solution_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_solution_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_statistic_analyzed_boards, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_statistic_pegs_taken, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_statistic_boards_without_moves, sizeof(int)));
    std::chrono::high_resolution_clock::time_point allocation_end = std::chrono::high_resolution_clock::now();
    printf("Allocation time: %f seconds\n", std::chrono::duration<double>(allocation_end - allocation_start).count());

    // Add the initial board to the map
    uint64_t initial_hash = encode_board(initial_board);
    board_map[initial_hash] = (uint64_t)-1;

    // Copy the initial board to the device
    std::chrono::high_resolution_clock::time_point copy_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_current_boards, initial_board, BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_current_hashes, &initial_hash, sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_solution_found, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_solution_idx, -1, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_statistic_analyzed_boards, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_statistic_pegs_taken, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_statistic_boards_without_moves, 0, sizeof(int)));
    std::chrono::high_resolution_clock::time_point copy_end = std::chrono::high_resolution_clock::now();
    printf("Copy time: %f seconds\n", std::chrono::duration<double>(copy_end - copy_start).count());

    int num_current_boards = 1;
    int threads_per_block = NUM_THREADS;
    int blocks;
    int solution_found = 0;
    uint64_t solution_hash;


    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    while (true)
    {
        // Process the current boards
        blocks = (num_current_boards + threads_per_block - 1) / threads_per_block;
        CUDA_CHECK(cudaMemset(d_next_count, 0, sizeof(int)));

        // Launch the kernel to process the boards
        process_boards<<<blocks, threads_per_block>>>(
            d_current_boards, d_current_hashes, d_next_boards, d_next_hashes, d_next_parents, d_next_count,
            d_solution_found, d_solution_idx, num_current_boards, d_statistic_analyzed_boards, d_statistic_pegs_taken, d_statistic_boards_without_moves);

        
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy the solution found flag and the number of next boards
        CUDA_CHECK(cudaMemcpy(&solution_found, d_solution_found, sizeof(int), cudaMemcpyDeviceToHost));
        int next_count;
        CUDA_CHECK(cudaMemcpy(&next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));

        int filtered_count = 0;

        CUDA_CHECK(cudaMemcpy(next_hashes, d_next_hashes, next_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(next_parents, d_next_parents, next_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(next_boards, d_next_boards, next_count * BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyDeviceToHost));


        if (solution_found)
        {
            // Solution has been found, copy the solution index and the solution hash
            int solution_idx;
            CUDA_CHECK(cudaMemcpy(&solution_idx, d_solution_idx, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&solution_hash, d_next_hashes + solution_idx, sizeof(uint64_t), cudaMemcpyDeviceToHost));

            // Add the solution hash to the map
            board_map[solution_hash] = next_parents[solution_idx];
            break;
        }

        // No solution found
        if (next_count == 0)
        {
            std::cout << "no solution found" << std::endl;
            break;
        }

        // Filter the next boards
        for (int i = 0; i < next_count; ++i)
        {
            // Check if the hash is already in the map
            if (board_map.find(next_hashes[i]) == board_map.end())
            {
                // Add the hash to the map
                board_map[next_hashes[i]] = next_parents[i];

                int *dest = filtered_boards + filtered_count * BOARD_SIZE * BOARD_SIZE;
                int *src = next_boards + i * BOARD_SIZE * BOARD_SIZE;

                // Copy the filred board to the next boards to be processed
                memcpy(dest, src, BOARD_SIZE * BOARD_SIZE * sizeof(int));

                filtered_hashes[filtered_count++] = next_hashes[i];
            }
        }


        num_current_boards = filtered_count;

        // Copy the filtered boards and hashes to the device
        CUDA_CHECK(cudaMemcpy(d_current_boards, filtered_boards, num_current_boards * BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_current_hashes, filtered_hashes, num_current_boards * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // Get the statistics
    int statistic_analyzed_boards;
    int statistic_pegs_taken;
    int statistic_boards_without_moves;

    CUDA_CHECK(cudaMemcpy(&statistic_analyzed_boards, d_statistic_analyzed_boards, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&statistic_pegs_taken, d_statistic_pegs_taken, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&statistic_boards_without_moves, d_statistic_boards_without_moves, sizeof(int), cudaMemcpyDeviceToHost));

    if (solution_found)
    {
        printf("Solution found!\n");
        retrace_solution(solution_hash, board_map);
    }
    else
    {
        printf("No solution found.\n");
    }

    printf("Statistics:\n");
    printf("Elapsed time: %f seconds\n", std::chrono::duration<double>(end - start).count());
    printf("Analyzed boards: %d\n", statistic_analyzed_boards);
    printf("Pegs taken: %d\n", statistic_pegs_taken);
    printf("Boards without moves: %d\n", statistic_boards_without_moves);

    // Free the memory
    CUDA_CHECK(cudaFree(d_current_boards));
    CUDA_CHECK(cudaFree(d_current_hashes));
    CUDA_CHECK(cudaFree(d_next_boards));
    CUDA_CHECK(cudaFree(d_next_hashes));
    CUDA_CHECK(cudaFree(d_next_parents));
    CUDA_CHECK(cudaFree(d_next_count));
    CUDA_CHECK(cudaFree(d_solution_found));
    CUDA_CHECK(cudaFree(d_solution_idx));
    CUDA_CHECK(cudaFree(d_pegs_taken));
    CUDA_CHECK(cudaFree(d_statistic_analyzed_boards));
    CUDA_CHECK(cudaFree(d_statistic_pegs_taken));
    CUDA_CHECK(cudaFree(d_statistic_boards_without_moves));

    free(next_hashes);
    free(next_parents);
    free(filtered_boards);
    free(filtered_hashes);
}

int main()
{
    int initial_board[BOARD_SIZE * BOARD_SIZE];
    memcpy(initial_board, read_board_from_file(), BOARD_SIZE * BOARD_SIZE * sizeof(int));
    print_board(initial_board);
    host_solve(initial_board);
    return 0;
}
