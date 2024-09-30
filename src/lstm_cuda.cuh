#ifndef LSTM_CUDA_CUH
#define LSTM_CUDA_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "lstm.h"

#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                               \
    if(e!=cudaSuccess) {                                            \
        printf("CUDA Error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

void lstm_forward_propagate_cuda(lstm_model_t* model, float *input, lstm_values_cache_t* cache_in, lstm_values_cache_t* cache_out,int softmax);

#ifdef __cplusplus
}
#endif

#endif // LSTM_CUDA_CUH
