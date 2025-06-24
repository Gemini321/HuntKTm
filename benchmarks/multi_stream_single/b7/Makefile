include ../common/make.config

# C compiler
CC = clang
CXX = clang++
CXX_FLAGS = -O1 -fno-exceptions
OPT = opt

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -O0
else
	NVCC_FLAGS += -O1
endif

# Target
NAME = b7

$(NAME): $(SRC)
	$(NVCC) -ccbin clang++ --cuda $(NAME).cu -o /tmp/$(NAME).ii
	$(NVCC) -O3 -DB7First $(NAME).cu -o $(NAME)_data -lm -lstdc++ -lpthread
	$(CC) $(CXX_FLAGS) -Xclang -load -Xclang $(WRAPPER_PASS) -flegacy-pass-manager /tmp/$(NAME).ii -c -o $(NAME).o > ./build.log 2>&1
	$(CC) $(CXX_FLAGS) $(NAME).o -o $(NAME) -L$(CUDA_LIB_DIR) -L$(LIB_DIR) -lcudart -lm -l${LIB_WRAPPER_NAME} -lstdc++ -lpthread
	$(CC) $(CXX_FLAGS) -Xclang -load -Xclang $(SCHEDULER_PASS) -Xclang -load -Xclang $(WRAPPER_PASS) -flegacy-pass-manager /tmp/$(NAME).ii -c -o $(NAME).o > ./build.log 2>&1
	$(CC) $(CXX_FLAGS) $(NAME).o -o $(NAME)_ms -L$(CUDA_LIB_DIR) -L$(LIB_DIR) -lcudart -lm -l${LIB_WRAPPER_NAME} -lstdc++ -lpthread

clean:
	rm -f *.o *.ll /tmp/*.ii $(NAME) $(NAME)_data $(NAME)_ms
