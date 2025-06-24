
include ../common/make.config

# C compiler
CC = clang++
CC_FLAGS = -O1 -std=c++17
OPT = opt

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -O1 -arch=compute_80 -code=sm_80 -ccbin clang++ --cuda

# Target
NAME = b10

$(NAME): $(SRC)
	$(NVCC) ${NVCC_FLAGS} $(NAME).cu -o /tmp/$(NAME).ii
	$(NVCC) ${NVCC_FLAGS} $(NAME).cu --ptxas-options=-v -c > kernel_info.txt 2>&1
	# original version
	$(CC) $(CC_FLAGS) /tmp/$(NAME).ii -c -o $(NAME).o > ./build.log 2>&1
	$(CC) $(CC_FLAGS) $(NAME).o -o $(NAME)_nowrap -L$(CUDA_LIB_DIR) -lcudart

	$(CC) $(CC_FLAGS) -Xclang -load -Xclang $(WRAPPER_PASS) -flegacy-pass-manager /tmp/$(NAME).ii -c -o $(NAME).o > ./build.log 2>&1
	$(CC) $(CC_FLAGS) $(NAME).o -o $(NAME) -L$(CUDA_LIB_DIR) -L$(LIB_DIR) -lcudart -lm -l${LIB_WRAPPER_NAME}
    
	# multi-stream version
	$(CC) $(CC_FLAGS) -Xclang -load -Xclang $(SCHEDULER_PASS) -mllvm -num-stream=10 -mllvm -use-async-memcpy=true -Xclang -load -Xclang $(WRAPPERNOWRAP_PASS) -flegacy-pass-manager /tmp/$(NAME).ii -c -o $(NAME).o > ./build.log 2>&1
	$(CC) $(CC_FLAGS) $(NAME).o -o $(NAME)_ms_nowrap -L$(CUDA_LIB_DIR) -lcudart
    
	$(CC) $(CC_FLAGS) -Xclang -load -Xclang $(SCHEDULER_PASS) -mllvm -num-stream=10 -mllvm -use-async-memcpy=true -Xclang -load -Xclang $(WRAPPER_PASS) -flegacy-pass-manager /tmp/$(NAME).ii -c -o $(NAME).o > ./build.log 2>&1
	$(CC) $(CC_FLAGS) $(NAME).o -o $(NAME)_ms -L$(CUDA_LIB_DIR) -L$(LIB_DIR) -lcudart -lm -l${LIB_WRAPPER_NAME}
    
	# multi-stream version with memory management
	$(CC) $(CC_FLAGS) -Xclang -load -Xclang $(SCHEDULER_PASS) -mllvm -num-stream=10 -mllvm -use-async-memcpy=true -Xclang -load -Xclang $(WRAPPERMEMORYNOWRAP_PASS) -flegacy-pass-manager /tmp/$(NAME).ii -c -o $(NAME).o > ./build.log 2>&1
	$(CC) $(CC_FLAGS) $(NAME).o -o $(NAME)_ms_mem_nowrap -L$(CUDA_LIB_DIR) -lcudart

	$(CC) $(CC_FLAGS) -Xclang -load -Xclang $(SCHEDULER_PASS) -mllvm -num-stream=10 -mllvm -use-async-memcpy=true -Xclang -load -Xclang $(WRAPPERMEMORY_PASS) -flegacy-pass-manager /tmp/$(NAME).ii -c -o $(NAME).o > ./build.log 2>&1
	$(CC) $(CC_FLAGS) $(NAME).o -o $(NAME)_ms_mem -L$(CUDA_LIB_DIR) -L$(LIB_DIR) -lcudart -lm -l${LIB_WRAPPER_NAME}

clean:
	rm -f *.o *.ll /tmp/*.ii $(NAME) $(NAME)_ms $(NAME)_ms_mem $(NAME)_nowrap $(NAME)_ms_nowrap $(NAME)_ms_mem_nowrap *.log *.dot *.png *.sqlite *.nsys-rep
