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
#define MAX_BOARDS 10000000
#define BOARDS_FILENAME "cpu_solution.txt"
#define HASH_FILENAME "cpu_hash.txt"
#define INITIAL_BOARD_FILENAME "initial_board.txt"

#define OUT_OF_BOUNDS 2
#define PEG 1
#define EMPTY 0

const bool valid_cells[BOARD_SIZE * BOARD_SIZE] = {
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
    1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
};
int* decode_board(uint64_t hash)
{
    int* board = (int*)malloc(BOARD_SIZE * BOARD_SIZE * sizeof(int));
    if(board == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
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
    return board;
}

int* read_board_from_file()
{
    FILE *file = fopen(INITIAL_BOARD_FILENAME, "r");
    uint64_t hash;
    if (fscanf(file, "%lu", &hash) != 1)
    {
        printf("Failed to read initial hash.\n");
        exit(1);
    }
    int* board = decode_board(hash);
    fclose(file);
    return board;
}

bool is_valid_move(int *board, int x, int y, int dx, int dy)
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

void apply_move(int *board, int x, int y, int dx, int dy)
{
    int nx = x + dx * 2;
    int ny = y + dy * 2;
    int mx = x + dx;
    int my = y + dy;
    board[x * BOARD_SIZE + y] = EMPTY;
    board[mx * BOARD_SIZE + my] = EMPTY;
    board[nx * BOARD_SIZE + ny] = PEG;
}

uint64_t encode_board(const int *board)
{
    uint64_t hash = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
    {
        if (valid_cells[i])
            hash = hash * 3 + board[i];
    }
    return hash;
}

void print_board(int *board)
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

void print_board_to_file(int *board)
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

void print_hash_to_file(uint64_t hash)
{
    FILE *file = fopen(HASH_FILENAME, "a");
    fprintf(file, "%lu\n", hash);
    fclose(file);
}

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

void solve(int *initial_board)
{
    std::unordered_map<uint64_t, uint64_t> board_map;
    std::vector<int> queue;
    std::vector<uint64_t> parent_map;
    
    queue.insert(queue.end(), initial_board, initial_board + BOARD_SIZE * BOARD_SIZE);
    uint64_t initial_hash = encode_board(initial_board);
    board_map[initial_hash] = (uint64_t)-1;
    parent_map.push_back(initial_hash);

    int pegs_taken = 0;
    int boards_without_moves = 0;
    int boards_processed = 0;

    int solution_idx = -1;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int iteration = 0;
    while (!queue.empty())
    {
        int board[BOARD_SIZE * BOARD_SIZE];
        memcpy(board, queue.data(), BOARD_SIZE * BOARD_SIZE * sizeof(int));
        queue.erase(queue.begin(), queue.begin() + BOARD_SIZE * BOARD_SIZE);
        uint64_t parent_hash = parent_map.front();
        parent_map.erase(parent_map.begin());

        bool has_moves = false;
        boards_processed++;

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
                                has_moves = true;
                                pegs_taken++;
                                int new_board[BOARD_SIZE * BOARD_SIZE];
                                memcpy(new_board, board, BOARD_SIZE * BOARD_SIZE * sizeof(int));
                                apply_move(new_board, x, y, dx, dy);
                                uint64_t new_hash = encode_board(new_board);
                                if (board_map.find(new_hash) == board_map.end())
                                {
                                    board_map[new_hash] = parent_hash;
                                    queue.insert(queue.end(), new_board, new_board + BOARD_SIZE * BOARD_SIZE);
                                    parent_map.push_back(new_hash);
                                    
                                    int peg_count = 0;
                                    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
                                        peg_count += (new_board[i] == PEG);
                                    if (peg_count == 1)
                                    {
                                        solution_idx = new_hash;
                                        goto end_search;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!has_moves)
            boards_without_moves++;
    }
    end_search:
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    if (solution_idx != -1)
    {
        std::cout << "Solution found!" << std::endl;
        retrace_solution(solution_idx, board_map);
    }
    else
    {
        std::cout << "No solution found." << std::endl;
    }

    std::cout << "Statistics:" << std::endl;
    std::cout << "CPU Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    std::cout << "Analyzed boards: " << boards_processed << std::endl;
    std::cout << "Pegs taken: " << pegs_taken << std::endl;
    std::cout << "Boards without moves: " << boards_without_moves << std::endl;
}

int main()
{
    int initial_board[BOARD_SIZE * BOARD_SIZE];
    memcpy(initial_board, read_board_from_file(), BOARD_SIZE * BOARD_SIZE * sizeof(int));
    print_board(initial_board);
    solve(initial_board);
    return 0;
}