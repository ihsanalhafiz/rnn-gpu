

#include "lstm_cuda.cuh"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void copy_vector_kernel_cuda(float* A, float* B, int L)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < L) {
    A[idx] = B[idx];
  }
}

void copy_vector_cuda(float* A, float* B, int L)
{
  int blockSize = 128;
  int gridSize = (L + blockSize - 1) / blockSize;
  copy_vector_kernel_cuda<<<gridSize, blockSize>>>(A, B, L);
}

__global__ void fully_connected_forward_kernel_cuda(float* Y, float* A, float* X, float* b, int R, int C)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R) {
    float sum = b[i];
    for (int n = 0; n < C; ++n) {
      sum += A[i * C + n] * X[n];
    }
    Y[i] = sum;
  }
}

void fully_connected_forward_cuda(float* Y, float* A, float* X, float* b, int R, int C)
{
  int blockSize = 128;
  int gridSize = (R + blockSize - 1) / blockSize;
  fully_connected_forward_kernel_cuda<<<gridSize, blockSize>>>(Y, A, X, b, R, C);
  cudaDeviceSynchronize();
}

__global__ void sigmoid_forward_kernel_cuda(float* Y, float* X, int L)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < L) {
    Y[idx] = 1.0 / (1.0 + exp(-X[idx]));
  }
}

void sigmoid_forward_cuda(float* Y, float* X, int L)
{
  int blockSize = 128;
  int gridSize = (L + blockSize - 1) / blockSize;
  sigmoid_forward_kernel_cuda<<<gridSize, blockSize>>>(Y, X, L);
  cudaDeviceSynchronize();
}

__global__ void tanh_forward_kernel_cuda(float* Y, float* X, int L)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < L) {
    Y[idx] = tanh(X[idx]);
  }
}

void tanh_forward_cuda(float* Y, float* X, int L)
{
  int blockSize = 128;
  int gridSize = (L + blockSize - 1) / blockSize;
  tanh_forward_kernel_cuda<<<gridSize, blockSize>>>(Y, X, L);
  cudaDeviceSynchronize();
}

__global__ void vectors_multiply_kernel_cuda(float* A, float* B, int L)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < L) {
    A[idx] *= B[idx];
  }
}

void vectors_multiply_cuda(float* A, float* B, int L)
{
  int blockSize = 128;
  int gridSize = (L + blockSize - 1) / blockSize;
  vectors_multiply_kernel_cuda<<<gridSize, blockSize>>>(A, B, L);
  cudaDeviceSynchronize();
}

__global__ void vectors_add_kernel_cuda(float* A, float* B, int L)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < L) {
    A[idx] += B[idx];
  }
}

void vectors_add_cuda(float* A, float* B, int L)
{
  int blockSize = 128;
  int gridSize = (L + blockSize - 1) / blockSize;
  vectors_add_kernel_cuda<<<gridSize, blockSize>>>(A, B, L);
  cudaDeviceSynchronize();
}

__global__ void softmax_layers_forward_kernel_cuda(float* cache, float* Y, int F, float temperature)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < F) {
    cache[idx] = exp(Y[idx] / temperature);
  }
}

__global__ void softmax_normalization_kernel_cuda(float* P, float* cache, float sum, int F)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < F) {
    P[idx] = cache[idx] / sum;
  }
}

__global__ void reduce_sum_kernel_cuda(float* input, float* output, int N)
{
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < N) ? input[i] : 0.0;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    output[blockIdx.x] = sdata[0];
  }
}

void softmax_layers_forward_cuda(float* P, float* Y, int F, float temperature)
{
  int blockSize = 128;
  int gridSize = (F + blockSize - 1) / blockSize;

  float* cache;
  cudaMalloc((void**)&cache, F * sizeof(float));

  softmax_layers_forward_kernel_cuda<<<gridSize, blockSize>>>(cache, Y, F, temperature);
  cudaDeviceSynchronize();
  // Reduction to compute the sum
  float* sum_array;
  cudaMalloc((void**)&sum_array, gridSize * sizeof(float));

  reduce_sum_kernel_cuda<<<gridSize, blockSize, blockSize * sizeof(float)>>>(cache, sum_array, F);
  cudaDeviceSynchronize();
  // Copy sum_array back to host and compute total sum
  float* h_sum_array = (float*)malloc(gridSize * sizeof(float));
  cudaMemcpy(h_sum_array, sum_array, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

  float sum = 0;
  for (int i = 0; i < gridSize; ++i) {
    sum += h_sum_array[i];
  }

  free(h_sum_array);
  cudaFree(sum_array);

  softmax_normalization_kernel_cuda<<<gridSize, blockSize>>>(P, cache, sum, F);
  cudaDeviceSynchronize();
  cudaFree(cache);
}

__global__ void initialize_X_one_hot_kernel_cuda(float* X_one_hot, float* h_old, float* input, int N, int S)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < S)
  {
    if (i < N)
    {
      X_one_hot[i] = h_old[i];
    }
    else
    {
      X_one_hot[i] = input[i - N];
    }
  }
}

// Assuming lstm_model_t is defined elsewhere
void lstm_forward_propagate_cuda(lstm_model_t* model, float *input,
  lstm_values_cache_t* cache_in, lstm_values_cache_t* cache_out,
  int softmax)
{
  int N, Y, S, i = 0;
  float *h_old, *c_old, *X_one_hot;

  cudaMallocManaged((void**)&h_old, model->N * sizeof(float));
  cudaCheckError();
  cudaMallocManaged((void**)&c_old, model->N * sizeof(float));
  cudaCheckError();
  //h_old = cache_in->h;
  //c_old = cache_in->c;
  cudaMemcpy(h_old, cache_in->h, model->N * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaCheckError();
  cudaMemcpy(c_old, cache_in->c, model->N * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaCheckError();

  N = model->N;
  Y = model->Y;
  S = model->S;
  
  //float tmp[N]; // VLA must be supported.. May cause portability problems.. If so use init_zero_vector (will be slower).
  float* tmp;
  cudaMallocManaged((void**)&tmp, N * sizeof(float));
  cudaCheckError();

  //copy_vector(cache_out->h_old, h_old, N);
  cudaMemcpy(cache_out->h_old, cache_in->h, N * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaCheckError();
  //copy_vector(cache_out->c_old, c_old, N);
  cudaMemcpy(cache_out->c_old, cache_in->c, N * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaCheckError();

  //X_one_hot = cache_out->X;
  cudaMallocManaged((void**)&X_one_hot, S * sizeof(float));
  cudaCheckError();
  cudaMemcpy(X_one_hot, cache_out->X, S * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaCheckError();

  while ( i < S ) {
    if ( i < N ) {
      X_one_hot[i] = h_old[i];
    } else {
      X_one_hot[i] = input[i - N];
    }
    ++i;
  }

  // Fully connected + sigmoid layers 
  fully_connected_forward_cuda(cache_out->hf, model->Wf, X_one_hot, model->bf, N, S);
  fully_connected_forward_cuda(cache_out->hi, model->Wi, X_one_hot, model->bi, N, S);
  fully_connected_forward_cuda(cache_out->ho, model->Wo, X_one_hot, model->bo, N, S);
  fully_connected_forward_cuda(cache_out->hc, model->Wc, X_one_hot, model->bc, N, S);
    
  //cudaCheckError();
  sigmoid_forward_cuda(cache_out->hi, cache_out->hi, N);
  //cudaCheckError();
  sigmoid_forward_cuda(cache_out->hf, cache_out->hf, N);
  //cudaCheckError();
  sigmoid_forward_cuda(cache_out->ho, cache_out->ho, N);
  //cudaCheckError();
  
  tanh_forward_cuda(cache_out->hc, cache_out->hc, N);

  // c = hf * c_old + hi * hc
  //copy_vector(cache_out->c, cache_out->hf, N);
  cudaMemcpy(cache_out->c, cache_out->hf, N * sizeof(float), cudaMemcpyDeviceToDevice);
  vectors_multiply_cuda(cache_out->c, c_old, N);
  //copy_vector(tmp, cache_out->hi, N);
  cudaMemcpy(tmp, cache_out->hi, N * sizeof(float), cudaMemcpyDeviceToDevice);
  vectors_multiply_cuda(tmp, cache_out->hc, N);

  vectors_add_cuda(cache_out->c, tmp, N);

  // h = ho * tanh_c_cache
  tanh_forward_cuda(cache_out->tanh_c_cache, cache_out->c, N);
  //copy_vector(cache_out->h, cache_out->ho, N);
  cudaMemcpy(cache_out->h, cache_out->ho, N * sizeof(float), cudaMemcpyDeviceToDevice);
  vectors_multiply_cuda(cache_out->h, cache_out->tanh_c_cache, N);

  // probs = softmax ( Wy*h + by )
  fully_connected_forward_cuda(cache_out->probs, model->Wy, cache_out->h, model->by, Y, N);
  if ( softmax > 0 ) {
    softmax_layers_forward_cuda(cache_out->probs, cache_out->probs, Y, model->params->softmax_temp);
  } 
#ifdef INTERLAYER_SIGMOID_ACTIVATION
  if ( softmax <= 0 ) {
    sigmoid_forward(cache_out->probs, cache_out->probs, Y);
    copy_vector(cache_out->probs_before_sigma, cache_out->probs, Y);
  }
#endif
  //copy_vector(cache_out->X, X_one_hot, S);
  cudaMemcpy(cache_out->X, X_one_hot, S * sizeof(float), cudaMemcpyDeviceToDevice);

  cudaFree(tmp);
  cudaFree(X_one_hot);
  cudaFree(h_old);
  cudaFree(c_old);
}

#ifdef __cplusplus
}
#endif