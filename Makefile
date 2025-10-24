CUDA_DIR := /usr/local/cuda-12

NVCC := $(CUDA_DIR)/bin/nvcc
CUDA_LIBS = -L$(CUDA_DIR)/lib64
CFLAGS := -lcublas -lcusparse -std=c++20

SRC_DIR := src
BUILD_DIR := build
BIN := main

SRC_FILES := $(shell find $(SRC_DIR) -type f \( -name "*.cu" -o -name "*.cpp" \))
OBJ_FILES := $(patsubst $(SRC_DIR)/%, $(BUILD_DIR)/%, $(SRC_FILES))
OBJ_FILES := $(OBJ_FILES:.cu=.o)
OBJ_FILES := $(OBJ_FILES:.cpp=.o)

all: build run

build: $(BIN)

$(BIN): $(OBJ_FILES)
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ $^ $(CUDA_LIBS) $(CFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) -x cu -c $< -o $@

run: $(BIN)
	./$(BIN)

clean:
	rm -rf $(BUILD_DIR) $(BIN)
