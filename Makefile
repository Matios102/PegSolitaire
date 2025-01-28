# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17

# Targets
TARGET_GPU = gpu_solver
SOLUTION_TARGET_GPU = gpu_solution.txt

# Sources
GPU_SRC = solution.cu

# Build GPU solver
gpu: $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET_GPU) $(GPU_SRC)

# Run GPU solver
run_gpu: gpu
	./$(TARGET_GPU)

# Clean
clean:
	rm -f $(TARGET_GPU) $(SOLUTION_TARGET_GPU)
