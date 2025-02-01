#include <iostream>
#include <vector>
#include <cstdio>

#define BOARD_SIZE 7 // Board size is 7x7
#define MAX_BOARDS 10000000
#define GPU_HASH_FILENAME "gpu_hash.txt"
#define CPU_HASH_FILENAME "cpu_hash.txt"

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

uint64_t encode_board(const std::vector<int>& board)
{
    uint64_t hash = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
    {
        if (board[i] != OUT_OF_BOUNDS)
        {
            hash = hash * 3 + board[i];
        }
    }
    return hash;
}

std::vector<int> decode_board(uint64_t hash)
{
    std::vector<int> board(BOARD_SIZE * BOARD_SIZE, OUT_OF_BOUNDS);
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

bool is_valid_move(const std::vector<int>& board, int x, int y, int dx, int dy)
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
void apply_move(std::vector<int>& board, int x, int y, int dx, int dy)
{
    int nx = x + dx * 2;
    int ny = y + dy * 2;
    int mx = x + dx;
    int my = y + dy;
    board[x * BOARD_SIZE + y] = EMPTY;
    board[mx * BOARD_SIZE + my] = EMPTY;
    board[nx * BOARD_SIZE + ny] = PEG;
}

std::vector<uint64_t> process_board(const std::vector<int>& board)
{
    std::vector<uint64_t> new_hashes;
    std::vector<int> new_board(BOARD_SIZE * BOARD_SIZE);
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
                            new_board = board;
                            apply_move(new_board, x, y, dx, dy);
                            uint64_t new_hash = encode_board(new_board);
                            new_hashes.push_back(new_hash);
                        }
                    }
                }
            }
        }
    }
    return new_hashes;
}

bool check_solution(const char *filename)
{
    FILE *hash_file = fopen(filename, "r");
    if (!hash_file)
    {
        std::cerr << "Failed to open hash file." << std::endl;
        return false;
    }

    uint64_t hash, next_hash;
    if (fscanf(hash_file, "%lu", &hash) != 1)
    {
        std::cerr << "Failed to read initial hash." << std::endl;
        fclose(hash_file);
        return false;
    }

    while (fscanf(hash_file, "%lu", &next_hash) != EOF)
    {
        std::vector<int> board = decode_board(hash);
        std::vector<uint64_t> new_hashes = process_board(board);
        bool found = false;
        for (uint64_t new_hash : new_hashes)
        {
            if (new_hash == next_hash)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            fclose(hash_file);
            return false;
        }
        hash = next_hash;
    }
    fclose(hash_file);
    return true;
}

int main()
{
    if (check_solution(GPU_HASH_FILENAME))
    {
        std::cout << "GPU solution is correct!" << std::endl;
    }
    else
    {
        std::cout << "GPU solution is incorrect!" << std::endl;
    }
    if(check_solution(CPU_HASH_FILENAME))
    {
        std::cout << "CPU solution is correct!" << std::endl;
    }
    else
    {
        std::cout << "CPU solution is incorrect!" << std::endl;
    }
    return 0;
}