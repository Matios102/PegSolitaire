#include <iostream>

#define BOARD_SIZE 7 // Board size is 7x7
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

// Function to initialize the board with pegs and a random empty cell
void initialize_board(int *board)
{
    for (int i = 0; i < BOARD_SIZE; ++i)
    {
        for (int j = 0; j < BOARD_SIZE; ++j)
        {
            if (i < 2 && j < 2)
            {
                board[i * BOARD_SIZE + j] = OUT_OF_BOUNDS;
            }
            else if (i < 2 && j > 4)
            {
                board[i * BOARD_SIZE + j] = OUT_OF_BOUNDS;
            }
            else if (i > 4 && j < 2)
            {
                board[i * BOARD_SIZE + j] = OUT_OF_BOUNDS;
            }
            else if (i > 4 && j > 4)
            {
                board[i * BOARD_SIZE + j] = OUT_OF_BOUNDS;
            }
            else
            {
                board[i * BOARD_SIZE + j] = PEG;
            }
        }
    }
    srand(time(NULL));
    while (true)
    {
        int x = rand() % BOARD_SIZE;
        int y = rand() % BOARD_SIZE;
        if (board[x * BOARD_SIZE + y] == PEG)
        {
            board[x * BOARD_SIZE + y] = EMPTY;
            break;
        }
    }
}


void print_hash_to_file(uint64_t hash)
{
    FILE *file = fopen(INITIAL_BOARD_FILENAME, "a");
    fprintf(file, "%lu\n", hash);
    fclose(file);
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


int main()
{
    int board[BOARD_SIZE * BOARD_SIZE];
    initialize_board(board);
    print_hash_to_file(encode_board(board));
    return 0;
}