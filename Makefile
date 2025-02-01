# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17
CPPFLAGS = -O3 -std=c++17

# Targets
TARGET_GPU = gpu_solver
TARGET_CPU = cpu_solver
TARGET_INITIAL_BOARD = initial_board
TARGET_VALIDATOR = validator

# Sources
GPU_SRC = solution.cu
CPU_SRC = solution.cpp
INITIAL_BOARD_SRC = initial_board.cpp
VALIDATOR_SRC = validator.cpp

# Build validator
validator: $(VALIDATOR_SRC)
	$(CXX) $(CPPFLAGS) -o $(TARGET_VALIDATOR) $(VALIDATOR_SRC)

# Run validator
run_validator: validator
	./$(TARGET_VALIDATOR) $(SOLUTION_TARGET_GPU)

# Build initial board generator
initial_board: $(INITIAL_BOARD_SRC)
	$(CXX) $(CPPFLAGS) -o $(TARGET_INITIAL_BOARD) $(INITIAL_BOARD_SRC)

# Run initial board generator
run_initial_board: initial_board
	./$(TARGET_INITIAL_BOARD)

# Build CPU solver
cpu: $(CPU_SRC)
	$(CXX) $(CPPFLAGS) -o $(TARGET_CPU) $(CPU_SRC)

# Run CPU solver
run_cpu: cpu
	./$(TARGET_CPU)

# Build GPU solver
gpu: $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET_GPU) $(GPU_SRC)

# Run GPU solver
run_gpu: gpu
	./$(TARGET_GPU)
