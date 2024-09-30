# Compiler settings
CC := gcc
NVCC := nvcc

# Compiler flags
CFLAGS := -g -O3 -msse3
NVCCFLAGS := -g -O3 -arch=sm_75   # For RTX 2070

# Libraries
LIBS := -lm -lcudart -lcublas
LIBSGCC := m
# GCC warnings
GCC_HINTS := all \
    unused \
    uninitialized \
    no-unused-variable \
    extra \
    unused-parameter

# Source files
C_SRCS := layers.c \
          lstm.c \
          main.c \
          set.c \
          utilities.c

CU_SRCS := lstm_cuda.cu

# Object files
C_OBJS := $(C_SRCS:.c=.o)
CU_OBJS := $(CU_SRCS:.cu=.o)

# Phony targets
.PHONY: clean all 

# Default target
all: clean net

# Compile C source files
%.o: %.c
	$(CC) -c $< $(CFLAGS) $(addprefix -W, $(GCC_HINTS)) $(addprefix -l, $(LIBSGCC)) -o $@

# Compile CUDA source files
%.o: %.cu
	$(NVCC) -c $< $(NVCCFLAGS) -o $@

# Link object files
net: $(C_OBJS) $(CU_OBJS)
	$(NVCC) $^ -o $@ $(LIBS)

# Clean up
clean:
	rm -f $(C_OBJS) $(CU_OBJS) net
