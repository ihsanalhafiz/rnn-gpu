/*
* This file is part of the LSTM Network implementation In C made by Rickard Hallerbäck
* 
*                 Copyright (c) 2018 Rickard Hallerbäck
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), 
* to deal in the Software without restriction, including without limitation the rights 
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
* Software, and to permit persons to whom the Software is furnished to do so, subject to 
* the following conditions:
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
*
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
* OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "lstm.h"
#include "lstm_cuda.cuh"
#include <time.h>

// Function to handle errors during LSTM initialization, prints an error message and exits the program
void lstm_init_fail(const char * msg)
{
  printf("%s: %s", __func__, msg);  // Print the error message along with the function name
  exit(-1);  // Exit the program with a non-zero value indicating failure
}

// Function to initialize an LSTM model with specific input (X), neurons (N), and output (Y) dimensions
// Also sets the model parameters and initializes the weights and biases
int lstm_init_model(int X, int N, int Y, 
  lstm_model_t **model_to_be_set, int zeros, 
  lstm_model_parameters_t *params)
{
  int S = X + N;  // S is the sum of input size X and number of neurons N (important for LSTM gates)
  lstm_model_t* lstm = e_calloc(1, sizeof(lstm_model_t));  // Allocate memory for the LSTM model

  // Store the dimensions and parameters in the LSTM model
  lstm->X = X;
  lstm->N = N;
  lstm->S = S;
  lstm->Y = Y;
  lstm->params = params;

  // Initialize the weight matrices and bias vectors depending on the 'zeros' flag
  // If 'zeros' is set, initialize them to zero vectors, else initialize with random values
  if (zeros) {
    lstm->Wf = get_zero_vector(N * S);  // Forget gate weight matrix
    lstm->Wi = get_zero_vector(N * S);  // Input gate weight matrix
    lstm->Wc = get_zero_vector(N * S);  // Cell state gate weight matrix
    lstm->Wo = get_zero_vector(N * S);  // Output gate weight matrix
    lstm->Wy = get_zero_vector(Y * N);  // Output weight matrix
  } else {
    lstm->Wf = get_random_vector(N * S, S);  // Random initialization for forget gate
    lstm->Wi = get_random_vector(N * S, S);  // Random initialization for input gate
    lstm->Wc = get_random_vector(N * S, S);  // Random initialization for cell state gate
    lstm->Wo = get_random_vector(N * S, S);  // Random initialization for output gate
    lstm->Wy = get_random_vector(Y * N, N);  // Random initialization for output weights
  }

  // Initialize the bias vectors for each gate and output
  lstm->bf = get_zero_vector(N);  // Forget gate bias
  lstm->bi = get_zero_vector(N);  // Input gate bias
  lstm->bc = get_zero_vector(N);  // Cell state gate bias
  lstm->bo = get_zero_vector(N);  // Output gate bias
  lstm->by = get_zero_vector(Y);  // Output bias

  // Allocate memory for the gradient variables used during backpropagation
  lstm->dldhf = get_zero_vector(N);  // Derivative of loss with respect to forget gate output
  lstm->dldhi = get_zero_vector(N);  // Derivative of loss with respect to input gate output
  lstm->dldhc = get_zero_vector(N);  // Derivative of loss with respect to cell state gate output
  lstm->dldho = get_zero_vector(N);  // Derivative of loss with respect to output gate output
  lstm->dldc  = get_zero_vector(N);  // Derivative of loss with respect to cell state
  lstm->dldh  = get_zero_vector(N);  // Derivative of loss with respect to hidden state

  // Initialize memory for partial derivatives with respect to the input gates
  lstm->dldXc = get_zero_vector(S);  // Derivative of loss with respect to input at cell state gate
  lstm->dldXo = get_zero_vector(S);  // Derivative of loss with respect to input at output gate
  lstm->dldXi = get_zero_vector(S);  // Derivative of loss with respect to input at input gate
  lstm->dldXf = get_zero_vector(S);  // Derivative of loss with respect to input at forget gate

  // Initialize memory for gradient descent momentum variables (used in optimization)
  lstm->Wfm = get_zero_vector(N * S);  // Momentum for forget gate weights
  lstm->Wim = get_zero_vector(N * S);  // Momentum for input gate weights
  lstm->Wcm = get_zero_vector(N * S);  // Momentum for cell state gate weights
  lstm->Wom = get_zero_vector(N * S);  // Momentum for output gate weights
  lstm->Wym = get_zero_vector(Y * N);  // Momentum for output weights

  lstm->bfm = get_zero_vector(N);  // Momentum for forget gate bias
  lstm->bim = get_zero_vector(N);  // Momentum for input gate bias
  lstm->bcm = get_zero_vector(N);  // Momentum for cell state bias
  lstm->bom = get_zero_vector(N);  // Momentum for output gate bias
  lstm->bym = get_zero_vector(Y);  // Momentum for output bias

  *model_to_be_set = lstm;  // Set the LSTM model pointer

  return 0;  // Return success
}

/*
int lstm_init_model_cuda(int X, int N, int Y, 
  lstm_model_t **model_to_be_set, int zeros, 
  lstm_model_parameters_t *params){

    int S = N + X;
    cudaMallocManaged((void**)&model_to_be_set, sizeof(lstm_model_t), cudaMemAttachGlobal);
    cudaMallocManaged((void)model_to_be_set->Wf, N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wi, N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wc, N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wo, N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wy, N * Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bf, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bi, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bc, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bo, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->by, Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->params, sizeof(lstm_model_parameters_t), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bf, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bi, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bc, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bo, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->by, Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldhf, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldhi, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldhc, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldho, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldc, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldh, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldXc, (N+X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldXo, (N+X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldXi, (N+X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->dldXf, (N+X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wfm, N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wim, N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wcm, N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wom, N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->Wym, S * Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bfm, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bim, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bcm, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bom, N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged(model_to_be_set->bym, Y * sizeof(float), cudaMemAttachGlobal);
    cudaMemset(model_to_be_set->Wf, 0, N * S);
    cudaMemset(model_to_be_set->Wi, 0, N * S);
    cudaMemset(model_to_be_set->Wc, 0, N * S);
    cudaMemset(model_to_be_set->Wo, 0, N * S);
    cudaMemset(model_to_be_set->Wy, 0, N * Y);
    cudaMemset(model_to_be_set->bf, 0, N);
    cudaMemset(model_to_be_set->bi, 0, N);
    cudaMemset(model_to_be_set->bc, 0, N);
    cudaMemset(model_to_be_set->bo, 0, N);
    cudaMemset(model_to_be_set->by, 0, Y);
    cudaMemset(model_to_be_set->dldhf, 0, N);
    cudaMemset(model_to_be_set->dldhi, 0, N);
    cudaMemset(model_to_be_set->dldhc, 0, N);
    cudaMemset(model_to_be_set->dldho, 0, N);
    cudaMemset(model_to_be_set->dldc, 0, N);
    cudaMemset(model_to_be_set->dldh, 0, N);
    cudaMemset(model_to_be_set->dldXc, 0, (N+X));
    cudaMemset(model_to_be_set->dldXo, 0, (N+X));
    cudaMemset(model_to_be_set->dldXi, 0, (N+X));
    cudaMemset(model_to_be_set->dldXf, 0, (N+X));
    cudaMemset(model_to_be_set->Wfm, 0, N * S);
    cudaMemset(model_to_be_set->Wim, 0, N * S);
    cudaMemset(model_to_be_set->Wcm, 0, N * S);
    cudaMemset(model_to_be_set->Wom, 0, N * S);
    cudaMemset(model_to_be_set->Wym, 0, S * Y);
    cudaMemset(model_to_be_set->bfm, 0, N);
    cudaMemset(model_to_be_set->bim, 0, N);
    cudaMemset(model_to_be_set->bcm, 0, N);
    cudaMemset(model_to_be_set->bom, 0, N);
    cudaMemset(model_to_be_set->bym, 0, Y);
    model_to_be_set[i]->X = X;
    model_to_be_set[i]->N = N;
    model_to_be_set[i]->Y = Y;
    model_to_be_set[i]->S = N + X;
    cudaMemcpy(model_to_be_set[i]->params, params, sizeof(lstm_model_parameters_t), cudaMemcpyHostToDevice);
}
*/
// Function to free the allocated memory for an LSTM model
void lstm_free_model(lstm_model_t* lstm)
{
  // Free the memory allocated for the weight matrices
  free_vector(&lstm->Wf);
  free_vector(&lstm->Wi);
  free_vector(&lstm->Wc);
  free_vector(&lstm->Wo);
  free_vector(&lstm->Wy);

  // Free the memory allocated for the bias vectors
  free_vector(&lstm->bf);
  free_vector(&lstm->bi);
  free_vector(&lstm->bc);
  free_vector(&lstm->bo);
  free_vector(&lstm->by);

  // Free the memory allocated for the gradient variables
  free_vector(&lstm->dldhf);
  free_vector(&lstm->dldhi);
  free_vector(&lstm->dldhc);
  free_vector(&lstm->dldho);
  free_vector(&lstm->dldc);
  free_vector(&lstm->dldh);

  // Free the memory allocated for the input gate derivatives
  free_vector(&lstm->dldXc);
  free_vector(&lstm->dldXo);
  free_vector(&lstm->dldXi);
  free_vector(&lstm->dldXf);

  // Free the memory allocated for gradient descent momentum caches
  free_vector(&lstm->Wfm);
  free_vector(&lstm->Wim);
  free_vector(&lstm->Wcm);
  free_vector(&lstm->Wom);
  free_vector(&lstm->Wym);

  free_vector(&lstm->bfm);
  free_vector(&lstm->bim);
  free_vector(&lstm->bcm);
  free_vector(&lstm->bom);
  free_vector(&lstm->bym);

  // Finally, free the memory for the LSTM model itself
  free(lstm);
}

// Function to free the cache used to store intermediate values during LSTM forward pass
void lstm_cache_container_free(lstm_values_cache_t* cache_to_be_freed)
{
  free_vector(&(cache_to_be_freed)->probs);  // Free memory for output probabilities
  free_vector(&(cache_to_be_freed)->probs_before_sigma);  // Free pre-sigmoid activation memory
  free_vector(&(cache_to_be_freed)->c);  // Free memory for cell state
  free_vector(&(cache_to_be_freed)->h);  // Free memory for hidden state
  free_vector(&(cache_to_be_freed)->c_old);  // Free memory for previous cell state
  free_vector(&(cache_to_be_freed)->h_old);  // Free memory for previous hidden state
  free_vector(&(cache_to_be_freed)->X);  // Free memory for input vector
  free_vector(&(cache_to_be_freed)->hf);  // Free memory for forget gate output
  free_vector(&(cache_to_be_freed)->hi);  // Free memory for input gate output
  free_vector(&(cache_to_be_freed)->ho);  // Free memory for output gate output
  free_vector(&(cache_to_be_freed)->hc);  // Free memory for cell state gate output
  free_vector(&(cache_to_be_freed)->tanh_c_cache);  // Free memory for cached tanh activation of cell state
}

// Function to initialize the cache for storing LSTM forward pass values
lstm_values_cache_t*  lstm_cache_container_init(int X, int N, int Y)
{
  int S = N + X;  // S is the sum of input size X and number of neurons N

  lstm_values_cache_t* cache = e_calloc(1, sizeof(lstm_values_cache_t));  // Allocate memory for the cache

  // Initialize cache memory with zero vectors for each of the LSTM components
  cache->probs = get_zero_vector(Y);  // Output probabilities
  cache->probs_before_sigma = get_zero_vector(Y);  // Pre-sigmoid activation of output
  cache->c = get_zero_vector(N);  // Cell state
  cache->h = get_zero_vector(N);  // Hidden state
  cache->c_old = get_zero_vector(N);  // Previous cell state
  cache->h_old = get_zero_vector(N);  // Previous hidden state
  cache->X = get_zero_vector(S);  // Input vector
  cache->hf = get_zero_vector(N);  // Forget gate output
  cache->hi = get_zero_vector(N);  // Input gate output
  cache->ho = get_zero_vector(N);  // Output gate output
  cache->hc = get_zero_vector(N);  // Cell state gate output
  cache->tanh_c_cache = get_zero_vector(N);  // Cached tanh of the cell state

  return cache;
}

// Initialize an LSTM values state container, which holds the cell and hidden states at a specific time step
void lstm_values_state_init(lstm_values_state_t** d_next_to_set, int N)
{
  lstm_values_state_t * d_next = e_calloc(1, sizeof(lstm_values_state_t));  // Allocate memory for state

  init_zero_vector(&d_next->c, N);  // Initialize cell state vector with zeros
  init_zero_vector(&d_next->h, N);  // Initialize hidden state vector with zeros

  *d_next_to_set = d_next;  // Set the pointer for the initialized state
}

// Function to check if the gradient values exceed a certain limit, returning a message if so
int gradients_fit(lstm_model_t* gradients, float limit)
{
  int msg = 0;  // Message accumulator to track how many values exceed the limit

  // Check each gradient vector and accumulate any over-the-limit values
  msg += vectors_fit(gradients->Wy, limit, gradients->Y * gradients->N);
  msg += vectors_fit(gradients->Wi, limit, gradients->N * gradients->S);
  msg += vectors_fit(gradients->Wc, limit, gradients->N * gradients->S);
  msg += vectors_fit(gradients->Wo, limit, gradients->N * gradients->S);
  msg += vectors_fit(gradients->Wf, limit, gradients->N * gradients->S);

  msg += vectors_fit(gradients->by, limit, gradients->Y);
  msg += vectors_fit(gradients->bi, limit, gradients->N);
  msg += vectors_fit(gradients->bc, limit, gradients->N);
  msg += vectors_fit(gradients->bf, limit, gradients->N);
  msg += vectors_fit(gradients->bo, limit, gradients->N);

  return msg;  // Return the number of violations (if any)
}

// Function to clip the gradients (limit their values to a maximum magnitude) to avoid exploding gradients
int gradients_clip(lstm_model_t* gradients, float limit)
{
  int msg = 0; // Message accumulator to track clipped values
    // Clip each gradient vector to the specified limit

  msg += vectors_clip(gradients->Wy, limit, gradients->Y * gradients->N);
  msg += vectors_clip(gradients->Wi, limit, gradients->N * gradients->S);
  msg += vectors_clip(gradients->Wc, limit, gradients->N * gradients->S);
  msg += vectors_clip(gradients->Wo, limit, gradients->N * gradients->S);
  msg += vectors_clip(gradients->Wf, limit, gradients->N * gradients->S);

  msg += vectors_clip(gradients->by, limit, gradients->Y);
  msg += vectors_clip(gradients->bi, limit, gradients->N);
  msg += vectors_clip(gradients->bc, limit, gradients->N);
  msg += vectors_clip(gradients->bf, limit, gradients->N);
  msg += vectors_clip(gradients->bo, limit, gradients->N);

  return msg;  // Return the number of values clipped
}

// Function to sum gradients across multiple samples in mini-batch training
void sum_gradients(lstm_model_t* gradients, lstm_model_t* gradients_entry)
{
   // Add each entry gradient to the main gradient accumulator
  vectors_add(gradients->Wy, gradients_entry->Wy, gradients->Y * gradients->N);
  vectors_add(gradients->Wi, gradients_entry->Wi, gradients->N * gradients->S);
  vectors_add(gradients->Wc, gradients_entry->Wc, gradients->N * gradients->S);
  vectors_add(gradients->Wo, gradients_entry->Wo, gradients->N * gradients->S);
  vectors_add(gradients->Wf, gradients_entry->Wf, gradients->N * gradients->S);

  vectors_add(gradients->by, gradients_entry->by, gradients->Y);
  vectors_add(gradients->bi, gradients_entry->bi, gradients->N);
  vectors_add(gradients->bc, gradients_entry->bc, gradients->N);
  vectors_add(gradients->bf, gradients_entry->bf, gradients->N);
  vectors_add(gradients->bo, gradients_entry->bo, gradients->N);
}

// A -= alpha * Am_hat / (np.sqrt(Rm_hat) + epsilon)
// Am_hat = Am / ( 1 - betaM ^ (iteration) )
// Rm_hat = Rm / ( 1 - betaR ^ (iteration) )

void gradients_adam_optimizer(lstm_model_t* model, lstm_model_t* gradients, lstm_model_t* M, lstm_model_t* R, unsigned int t) 
{
  // Extract the beta parameters from the model
  float beta1 = model->params->beta1;  // Beta1 is the exponential decay rate for the first moment estimates
  float beta2 = model->params->beta2;  // Beta2 is the exponential decay rate for the second moment estimates

  // Correct bias for both beta1 and beta2 using the iteration count (t)
  float beta1t = 1.0 / (1.0 - pow(beta1, t+1));  // Bias-corrected beta1
  float beta2t = 1.0 / (1.0 - pow(beta2, t+1));  // Bias-corrected beta2

  // Error check: If beta2t is NaN (not a number), print an error message and exit
  if (!(beta2t == beta2t)) {
    printf("beta2t: %lf\n", beta2t);
    exit(0);
  }
  // Copy current gradients (weight updates) to momentum (M)
  copy_vector(gradients->Wym, gradients->Wy, model->Y * model->N);
  copy_vector(gradients->Wim, gradients->Wi, model->N * model->S);
  copy_vector(gradients->Wcm, gradients->Wc, model->N * model->S);
  copy_vector(gradients->Wom, gradients->Wo, model->N * model->S);
  copy_vector(gradients->Wfm, gradients->Wf, model->N * model->S);

  copy_vector(gradients->bym, gradients->by, model->Y);
  copy_vector(gradients->bim, gradients->bi, model->N);
  copy_vector(gradients->bcm, gradients->bc, model->N);
  copy_vector(gradients->bom, gradients->bo, model->N);
  copy_vector(gradients->bfm, gradients->bf, model->N);

  // Update momentum term by multiplying the current gradients by (1 - beta1)
  vectors_mutliply_scalar(gradients->Wym, 1.0 - beta1, model->Y * model->N);
  vectors_mutliply_scalar(gradients->Wim, 1.0 - beta1, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wcm, 1.0 - beta1, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wom, 1.0 - beta1, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wfm, 1.0 - beta1, model->N * model->S);

  vectors_mutliply_scalar(gradients->bym, 1.0 - beta1, model->Y);
  vectors_mutliply_scalar(gradients->bim, 1.0 - beta1, model->N);
  vectors_mutliply_scalar(gradients->bcm, 1.0 - beta1, model->N);
  vectors_mutliply_scalar(gradients->bom, 1.0 - beta1, model->N);
  vectors_mutliply_scalar(gradients->bfm, 1.0 - beta1, model->N);

  // Scale the momentum term (M) by beta1
  vectors_mutliply_scalar(M->Wy, beta1, model->Y * model->N);
  vectors_mutliply_scalar(M->Wi, beta1, model->N * model->S);
  vectors_mutliply_scalar(M->Wc, beta1, model->N * model->S);
  vectors_mutliply_scalar(M->Wo, beta1, model->N * model->S);
  vectors_mutliply_scalar(M->Wf, beta1, model->N * model->S);

  vectors_mutliply_scalar(M->by, beta1, model->Y);
  vectors_mutliply_scalar(M->bi, beta1, model->N);
  vectors_mutliply_scalar(M->bc, beta1, model->N);
  vectors_mutliply_scalar(M->bo, beta1, model->N);
  vectors_mutliply_scalar(M->bf, beta1, model->N);

  // Update momentum with the current gradients
  vectors_add(M->Wy, gradients->Wy, model->Y * model->N);
  vectors_add(M->Wi, gradients->Wi, model->N * model->S);
  vectors_add(M->Wc, gradients->Wc, model->N * model->S);
  vectors_add(M->Wo, gradients->Wo, model->N * model->S);
  vectors_add(M->Wf, gradients->Wf, model->N * model->S);

  vectors_add(M->by, gradients->by, model->Y);
  vectors_add(M->bi, gradients->bi, model->N);
  vectors_add(M->bc, gradients->bc, model->N);
  vectors_add(M->bo, gradients->bo, model->N);
  vectors_add(M->bf, gradients->bf, model->N);

  // Momentum calculation done!
  // Now compute the second moment estimate (R)

  // Square each gradient value for the second moment estimate
  vectors_multiply(gradients->Wy, gradients->Wy, model->Y * model->N);
  vectors_multiply(gradients->Wi, gradients->Wi, model->N * model->S);
  vectors_multiply(gradients->Wc, gradients->Wc, model->N * model->S);
  vectors_multiply(gradients->Wo, gradients->Wo, model->N * model->S);
  vectors_multiply(gradients->Wf, gradients->Wf, model->N * model->S);

  vectors_multiply(gradients->by, gradients->by, model->Y );
  vectors_multiply(gradients->bi, gradients->bi, model->N );
  vectors_multiply(gradients->bc, gradients->bc, model->N );
  vectors_multiply(gradients->bo, gradients->bo, model->N );
  vectors_multiply(gradients->bf, gradients->bf, model->N );

  // Copy the squared gradients to R (second moment estimate) memory
  copy_vector(gradients->Wym, gradients->Wy, model->Y * model->N);
  copy_vector(gradients->Wim, gradients->Wi, model->N * model->S);
  copy_vector(gradients->Wcm, gradients->Wc, model->N * model->S);
  copy_vector(gradients->Wom, gradients->Wo, model->N * model->S);
  copy_vector(gradients->Wfm, gradients->Wf, model->N * model->S);

  copy_vector(gradients->bym, gradients->by, model->Y);
  copy_vector(gradients->bim, gradients->bi, model->N);
  copy_vector(gradients->bcm, gradients->bc, model->N);
  copy_vector(gradients->bom, gradients->bo, model->N);
  copy_vector(gradients->bfm, gradients->bf, model->N);

  // Scale the second moment estimate by (1 - beta2)
  vectors_mutliply_scalar(gradients->Wym, 1.0 - beta2, model->Y * model->N);
  vectors_mutliply_scalar(gradients->Wim, 1.0 - beta2, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wcm, 1.0 - beta2, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wom, 1.0 - beta2, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wfm, 1.0 - beta2, model->N * model->S);

  vectors_mutliply_scalar(gradients->bym, 1.0 - beta2, model->Y);
  vectors_mutliply_scalar(gradients->bim, 1.0 - beta2, model->N);
  vectors_mutliply_scalar(gradients->bcm, 1.0 - beta2, model->N);
  vectors_mutliply_scalar(gradients->bom, 1.0 - beta2, model->N);
  vectors_mutliply_scalar(gradients->bfm, 1.0 - beta2, model->N);

  // Update the second moment estimate R by scaling with beta2 and adding the new gradient
  vectors_mutliply_scalar(R->Wy, beta2, model->Y * model->N);
  vectors_mutliply_scalar(R->Wi, beta2, model->N * model->S);
  vectors_mutliply_scalar(R->Wc, beta2, model->N * model->S);
  vectors_mutliply_scalar(R->Wo, beta2, model->N * model->S);
  vectors_mutliply_scalar(R->Wf, beta2, model->N * model->S);

  vectors_mutliply_scalar(R->by, beta2, model->Y);
  vectors_mutliply_scalar(R->bi, beta2, model->N);
  vectors_mutliply_scalar(R->bc, beta2, model->N);
  vectors_mutliply_scalar(R->bo, beta2, model->N);
  vectors_mutliply_scalar(R->bf, beta2, model->N);

  vectors_add(R->Wy, gradients->Wy, model->Y * model->N);
  vectors_add(R->Wi, gradients->Wi, model->N * model->S);
  vectors_add(R->Wc, gradients->Wc, model->N * model->S);
  vectors_add(R->Wo, gradients->Wo, model->N * model->S);
  vectors_add(R->Wf, gradients->Wf, model->N * model->S);

  vectors_add(R->by, gradients->by, model->Y);
  vectors_add(R->bi, gradients->bi, model->N);
  vectors_add(R->bc, gradients->bc, model->N);
  vectors_add(R->bo, gradients->bo, model->N);
  vectors_add(R->bf, gradients->bf, model->N);

  // R done!

  copy_vector(M->Wym, M->Wy, model->Y * model->N);
  copy_vector(M->Wim, M->Wi, model->N * model->S);
  copy_vector(M->Wcm, M->Wc, model->N * model->S);
  copy_vector(M->Wom, M->Wo, model->N * model->S);
  copy_vector(M->Wfm, M->Wf, model->N * model->S);

  copy_vector(M->bym, M->by, model->Y);
  copy_vector(M->bim, M->bi, model->N);
  copy_vector(M->bcm, M->bc, model->N);
  copy_vector(M->bom, M->bo, model->N);
  copy_vector(M->bfm, M->bf, model->N);

  vectors_mutliply_scalar(M->Wym, beta1t, model->Y * model->N);
  vectors_mutliply_scalar(M->Wim, beta1t, model->N * model->S);
  vectors_mutliply_scalar(M->Wcm, beta1t, model->N * model->S);
  vectors_mutliply_scalar(M->Wom, beta1t, model->N * model->S);
  vectors_mutliply_scalar(M->Wfm, beta1t, model->N * model->S);

  vectors_mutliply_scalar(M->bym, beta1t, model->Y);
  vectors_mutliply_scalar(M->bim, beta1t, model->N);
  vectors_mutliply_scalar(M->bcm, beta1t, model->N);
  vectors_mutliply_scalar(M->bom, beta1t, model->N);
  vectors_mutliply_scalar(M->bfm, beta1t, model->N);

  // M hat done!

  copy_vector(R->Wym, R->Wy, model->Y * model->N);
  copy_vector(R->Wim, R->Wi, model->N * model->S);
  copy_vector(R->Wcm, R->Wc, model->N * model->S);
  copy_vector(R->Wom, R->Wo, model->N * model->S);
  copy_vector(R->Wfm, R->Wf, model->N * model->S);

  copy_vector(R->bym, R->by, model->Y);
  copy_vector(R->bim, R->bi, model->N);
  copy_vector(R->bcm, R->bc, model->N);
  copy_vector(R->bom, R->bo, model->N);
  copy_vector(R->bfm, R->bf, model->N);

  vectors_mutliply_scalar(R->Wym, beta2t, model->Y * model->N);
  vectors_mutliply_scalar(R->Wim, beta2t, model->N * model->S);
  vectors_mutliply_scalar(R->Wcm, beta2t, model->N * model->S);
  vectors_mutliply_scalar(R->Wom, beta2t, model->N * model->S);
  vectors_mutliply_scalar(R->Wfm, beta2t, model->N * model->S);

  vectors_mutliply_scalar(R->bym, beta2t, model->Y);
  vectors_mutliply_scalar(R->bim, beta2t, model->N);
  vectors_mutliply_scalar(R->bcm, beta2t, model->N);
  vectors_mutliply_scalar(R->bom, beta2t, model->N);
  vectors_mutliply_scalar(R->bfm, beta2t, model->N);

  // R hat done!

  vector_sqrt(R->Wym, model->Y * model->N);
  vector_sqrt(R->Wim, model->N * model->S);
  vector_sqrt(R->Wcm, model->N * model->S);
  vector_sqrt(R->Wom, model->N * model->S);
  vector_sqrt(R->Wfm, model->N * model->S);

  vector_sqrt(R->bym, model->Y);
  vector_sqrt(R->bim, model->N);
  vector_sqrt(R->bcm, model->N);
  vector_sqrt(R->bom, model->N);
  vector_sqrt(R->bfm, model->N);

  vectors_add_scalar(R->Wym, 1e-7, model->Y * model->N);
  vectors_add_scalar(R->Wim, 1e-7, model->N * model->S);
  vectors_add_scalar(R->Wcm, 1e-7, model->N * model->S);
  vectors_add_scalar(R->Wom, 1e-7, model->N * model->S);
  vectors_add_scalar(R->Wfm, 1e-7, model->N * model->S);

  vectors_add_scalar(R->bym, 1e-7, model->Y);
  vectors_add_scalar(R->bim, 1e-7, model->N);
  vectors_add_scalar(R->bcm, 1e-7, model->N);
  vectors_add_scalar(R->bom, 1e-7, model->N);
  vectors_add_scalar(R->bfm, 1e-7, model->N);

  copy_vector(gradients->Wym, M->Wym, model->Y * model->N);
  copy_vector(gradients->Wim, M->Wim, model->N * model->S);
  copy_vector(gradients->Wcm, M->Wcm, model->N * model->S);
  copy_vector(gradients->Wom, M->Wom, model->N * model->S);
  copy_vector(gradients->Wfm, M->Wfm, model->N * model->S);

  copy_vector(gradients->bym, M->bym, model->Y);
  copy_vector(gradients->bim, M->bim, model->N);
  copy_vector(gradients->bcm, M->bcm, model->N);
  copy_vector(gradients->bom, M->bom, model->N);
  copy_vector(gradients->bfm, M->bfm, model->N);	

  vectors_scalar_multiply(gradients->Wym, model->params->learning_rate, model->Y * model->N);
  vectors_scalar_multiply(gradients->Wim, model->params->learning_rate, model->N * model->S);
  vectors_scalar_multiply(gradients->Wcm, model->params->learning_rate, model->N * model->S);
  vectors_scalar_multiply(gradients->Wom, model->params->learning_rate, model->N * model->S);
  vectors_scalar_multiply(gradients->Wfm, model->params->learning_rate, model->N * model->S);

  vectors_scalar_multiply(gradients->bym, model->params->learning_rate, model->Y);
  vectors_scalar_multiply(gradients->bim, model->params->learning_rate, model->N);
  vectors_scalar_multiply(gradients->bcm, model->params->learning_rate, model->N);
  vectors_scalar_multiply(gradients->bom, model->params->learning_rate, model->N);
  vectors_scalar_multiply(gradients->bfm, model->params->learning_rate, model->N);	

  vectors_div(gradients->Wym, R->Wym, model->Y * model->N);
  vectors_div(gradients->Wim, R->Wim, model->N * model->S);
  vectors_div(gradients->Wcm, R->Wcm, model->N * model->S);
  vectors_div(gradients->Wom, R->Wom, model->N * model->S);
  vectors_div(gradients->Wfm, R->Wfm, model->N * model->S);

  vectors_div(gradients->bym, R->bym, model->Y);
  vectors_div(gradients->bim, R->bim, model->N);
  vectors_div(gradients->bcm, R->bcm, model->N);
  vectors_div(gradients->bom, R->bom, model->N);
  vectors_div(gradients->bfm, R->bfm, model->N);

  vectors_substract(model->Wy, gradients->Wym, model->Y * model->N);
  vectors_substract(model->Wi, gradients->Wim, model->N * model->S);
  vectors_substract(model->Wc, gradients->Wcm, model->N * model->S);
  vectors_substract(model->Wo, gradients->Wom, model->N * model->S);
  vectors_substract(model->Wf, gradients->Wfm, model->N * model->S);

  vectors_substract(model->by, gradients->bym, model->Y);
  vectors_substract(model->bi, gradients->bim, model->N);
  vectors_substract(model->bc, gradients->bcm, model->N);
  vectors_substract(model->bo, gradients->bom, model->N);
  vectors_substract(model->bf, gradients->bfm, model->N);	
	
} 


// A = A - alpha * m, m = momentum * m + ( 1 - momentum ) * dldA
void gradients_decend(lstm_model_t* model, lstm_model_t* gradients) {

  // Computing momumentum * m
  vectors_mutliply_scalar(gradients->Wym, model->params->momentum, model->Y * model->N);
  vectors_mutliply_scalar(gradients->Wim, model->params->momentum, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wcm, model->params->momentum, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wom, model->params->momentum, model->N * model->S);
  vectors_mutliply_scalar(gradients->Wfm, model->params->momentum, model->N * model->S);

  vectors_mutliply_scalar(gradients->bym, model->params->momentum, model->Y);
  vectors_mutliply_scalar(gradients->bim, model->params->momentum, model->N);
  vectors_mutliply_scalar(gradients->bcm, model->params->momentum, model->N);
  vectors_mutliply_scalar(gradients->bom, model->params->momentum, model->N);
  vectors_mutliply_scalar(gradients->bfm, model->params->momentum, model->N);

  // Computing m = momentum * m + (1 - momentum) * dldA
  vectors_add_scalar_multiply(gradients->Wym, gradients->Wy, model->Y * model->N, 1.0 - model->params->momentum);
  vectors_add_scalar_multiply(gradients->Wim, gradients->Wi, model->N * model->S, 1.0 - model->params->momentum);
  vectors_add_scalar_multiply(gradients->Wcm, gradients->Wc, model->N * model->S, 1.0 - model->params->momentum);
  vectors_add_scalar_multiply(gradients->Wom, gradients->Wo, model->N * model->S, 1.0 - model->params->momentum);
  vectors_add_scalar_multiply(gradients->Wfm, gradients->Wf, model->N * model->S, 1.0 - model->params->momentum);

  vectors_add_scalar_multiply(gradients->bym, gradients->by, model->Y, 1.0 - model->params->momentum);
  vectors_add_scalar_multiply(gradients->bim, gradients->bi, model->N, 1.0 - model->params->momentum);
  vectors_add_scalar_multiply(gradients->bcm, gradients->bc, model->N, 1.0 - model->params->momentum);
  vectors_add_scalar_multiply(gradients->bom, gradients->bo, model->N, 1.0 - model->params->momentum);
  vectors_add_scalar_multiply(gradients->bfm, gradients->bf, model->N, 1.0 - model->params->momentum);

  // Computing A = A - alpha * m
  vectors_substract_scalar_multiply(model->Wy, gradients->Wym, model->Y * model->N, model->params->learning_rate);
  vectors_substract_scalar_multiply(model->Wi, gradients->Wim, model->N * model->S, model->params->learning_rate);
  vectors_substract_scalar_multiply(model->Wc, gradients->Wcm, model->N * model->S, model->params->learning_rate);
  vectors_substract_scalar_multiply(model->Wo, gradients->Wom, model->N * model->S, model->params->learning_rate);
  vectors_substract_scalar_multiply(model->Wf, gradients->Wfm, model->N * model->S, model->params->learning_rate);

  vectors_substract_scalar_multiply(model->by, gradients->bym, model->Y, model->params->learning_rate);
  vectors_substract_scalar_multiply(model->bi, gradients->bim, model->N, model->params->learning_rate);
  vectors_substract_scalar_multiply(model->bc, gradients->bcm, model->N, model->params->learning_rate);
  vectors_substract_scalar_multiply(model->bf, gradients->bfm, model->N, model->params->learning_rate);
  vectors_substract_scalar_multiply(model->bo, gradients->bom, model->N, model->params->learning_rate);
}

void lstm_values_next_cache_init(lstm_values_next_cache_t** d_next_to_set, int N, int X)
{
  lstm_values_next_cache_t * d_next = e_calloc(1, sizeof(lstm_values_next_cache_t));

  init_zero_vector(&d_next->dldh_next, N);
  init_zero_vector(&d_next->dldc_next, N);
  init_zero_vector(&d_next->dldY_pass, X);
  *d_next_to_set = d_next;
}
void lstm_values_next_cache_free(lstm_values_next_cache_t* d_next)
{
  free_vector(&d_next->dldc_next);
  free_vector(&d_next->dldh_next);
  free_vector(&d_next->dldY_pass);
  free(d_next);
}

void lstm_values_next_state_free(lstm_values_state_t* d_next)
{
  free_vector(&d_next->h);
  free_vector(&d_next->c);
  free(d_next);
}

// model, input, state and cache values, &probs, whether or not to apply softmax
void lstm_forward_propagate(lstm_model_t* model, float *input,
  lstm_values_cache_t* cache_in, lstm_values_cache_t* cache_out,
  int softmax)
{
  int N, Y, S, i = 0;
  float *h_old, *c_old, *X_one_hot;

  h_old = cache_in->h;
  c_old = cache_in->c;

  N = model->N;
  Y = model->Y;
  S = model->S;

  float tmp[N]; // VLA must be supported.. May cause portability problems.. If so use init_zero_vector (will be slower).

  copy_vector(cache_out->h_old, h_old, N);
  copy_vector(cache_out->c_old, c_old, N);

  X_one_hot = cache_out->X;

  while ( i < S ) {
    if ( i < N ) {
      X_one_hot[i] = h_old[i];
    } else {
      X_one_hot[i] = input[i - N];
    }
    ++i;
  }

  // Fully connected + sigmoid layers 
  fully_connected_forward(cache_out->hf, model->Wf, X_one_hot, model->bf, N, S);
  sigmoid_forward(cache_out->hf, cache_out->hf, N);

  fully_connected_forward(cache_out->hi, model->Wi, X_one_hot, model->bi, N, S);
  sigmoid_forward(cache_out->hi, cache_out->hi, N);

  fully_connected_forward(cache_out->ho, model->Wo, X_one_hot, model->bo, N, S);
  sigmoid_forward(cache_out->ho, cache_out->ho, N);

  fully_connected_forward(cache_out->hc, model->Wc, X_one_hot, model->bc, N, S);
  tanh_forward(cache_out->hc, cache_out->hc, N);

  // c = hf * c_old + hi * hc
  copy_vector(cache_out->c, cache_out->hf, N);
  vectors_multiply(cache_out->c, c_old, N);
  copy_vector(tmp, cache_out->hi, N);
  vectors_multiply(tmp, cache_out->hc, N);

  vectors_add(cache_out->c, tmp, N);

  // h = ho * tanh_c_cache
  tanh_forward(cache_out->tanh_c_cache, cache_out->c, N);
  copy_vector(cache_out->h, cache_out->ho, N);
  vectors_multiply(cache_out->h, cache_out->tanh_c_cache, N);

  // probs = softmax ( Wy*h + by )
  fully_connected_forward(cache_out->probs, model->Wy, cache_out->h, model->by, Y, N);
  if ( softmax > 0 ) {
    softmax_layers_forward(cache_out->probs, cache_out->probs, Y, model->params->softmax_temp);
  } 
#ifdef INTERLAYER_SIGMOID_ACTIVATION
  if ( softmax <= 0 ) {
    sigmoid_forward(cache_out->probs, cache_out->probs, Y);
    copy_vector(cache_out->probs_before_sigma, cache_out->probs, Y);
  }
#endif
  copy_vector(cache_out->X, X_one_hot, S);
}
//							model, y_probabilities, y_correct, the next deltas, state and cache values, &gradients, &the next deltas
void lstm_backward_propagate(lstm_model_t* model, float* y_probabilities, int y_correct, 
  lstm_values_next_cache_t* d_next, lstm_values_cache_t* cache_in, 
  lstm_model_t* gradients, lstm_values_next_cache_t* cache_out)
{
  float *h,*dldh_next,*dldc_next, *dldy, *dldh, *dldho, *dldhf, *dldhi, *dldhc, *dldc;
  int N, Y, S;

  N = model->N;
  Y = model->Y;
  S = model->S;

  // model cache
  dldh = model->dldh;
  dldc = model->dldc;
  dldho = model->dldho;
  dldhi = model->dldhi;
  dldhf = model->dldhf;
  dldhc = model->dldhc;

  h = cache_in->h;

  dldh_next = d_next->dldh_next;
  dldc_next = d_next->dldc_next;

  dldy = y_probabilities;

  if ( y_correct >= 0 ) {
    dldy[y_correct] -= 1.0;
  }
#ifdef INTERLAYER_SIGMOID_ACTIVATION
  if ( y_correct < 0 ) {
    sigmoid_backward(dldy, cache_in->probs_before_sigma, dldy, Y);
  }
#endif

  fully_connected_backward(dldy, model->Wy, h, gradients->Wy, dldh, gradients->by, Y, N);
  vectors_add(dldh, dldh_next, N);

  copy_vector(dldho, dldh, N);
  vectors_multiply(dldho, cache_in->tanh_c_cache, N);
  sigmoid_backward(dldho, cache_in->ho, dldho, N);

  copy_vector(dldc, dldh, N);
  vectors_multiply(dldc, cache_in->ho, N);
  tanh_backward(dldc, cache_in->tanh_c_cache, dldc, N);
  vectors_add(dldc, dldc_next, N);

  copy_vector(dldhf, dldc, N);
  vectors_multiply(dldhf, cache_in->c_old, N);
  sigmoid_backward(dldhf, cache_in->hf, dldhf, N);

  copy_vector(dldhi, cache_in->hc, N);
  vectors_multiply(dldhi, dldc, N);
  sigmoid_backward(dldhi, cache_in->hi, dldhi, N);

  copy_vector(dldhc, cache_in->hi, N);
  vectors_multiply(dldhc, dldc, N);
  tanh_backward(dldhc, cache_in->hc, dldhc, N);

  fully_connected_backward(dldhi, model->Wi, cache_in->X, gradients->Wi, gradients->dldXi, gradients->bi, N, S);
  fully_connected_backward(dldhc, model->Wc, cache_in->X, gradients->Wc, gradients->dldXc, gradients->bc, N, S);
  fully_connected_backward(dldho, model->Wo, cache_in->X, gradients->Wo, gradients->dldXo, gradients->bo, N, S);
  fully_connected_backward(dldhf, model->Wf, cache_in->X, gradients->Wf, gradients->dldXf, gradients->bf, N, S);

  // dldXi will work as a temporary substitute for dldX (where we get extract dh_next from!)
  vectors_add(gradients->dldXi, gradients->dldXc, S);
  vectors_add(gradients->dldXi, gradients->dldXo, S);
  vectors_add(gradients->dldXi, gradients->dldXf, S);

  copy_vector(cache_out->dldh_next, gradients->dldXi, N);
  copy_vector(cache_out->dldc_next, cache_in->hf, N);
  vectors_multiply(cache_out->dldc_next, dldc, N);

  // To pass on to next layer
  copy_vector(cache_out->dldY_pass, &gradients->dldXi[N], model->X);
}

void lstm_zero_the_model(lstm_model_t * model)
{
  vector_set_to_zero(model->Wy, model->Y * model->N);
  vector_set_to_zero(model->Wi, model->N * model->S);
  vector_set_to_zero(model->Wc, model->N * model->S);
  vector_set_to_zero(model->Wo, model->N * model->S);
  vector_set_to_zero(model->Wf, model->N * model->S);

  vector_set_to_zero(model->by, model->Y);
  vector_set_to_zero(model->bi, model->N);
  vector_set_to_zero(model->bc, model->N);
  vector_set_to_zero(model->bf, model->N);
  vector_set_to_zero(model->bo, model->N);

  vector_set_to_zero(model->Wym, model->Y * model->N);
  vector_set_to_zero(model->Wim, model->N * model->S);
  vector_set_to_zero(model->Wcm, model->N * model->S);
  vector_set_to_zero(model->Wom, model->N * model->S);
  vector_set_to_zero(model->Wfm, model->N * model->S);

  vector_set_to_zero(model->bym, model->Y);
  vector_set_to_zero(model->bim, model->N);
  vector_set_to_zero(model->bcm, model->N);
  vector_set_to_zero(model->bfm, model->N);
  vector_set_to_zero(model->bom, model->N);

  vector_set_to_zero(model->dldhf, model->N);
  vector_set_to_zero(model->dldhi, model->N);
  vector_set_to_zero(model->dldhc, model->N);
  vector_set_to_zero(model->dldho, model->N);
  vector_set_to_zero(model->dldc, model->N);
  vector_set_to_zero(model->dldh, model->N);

  vector_set_to_zero(model->dldXc, model->S);
  vector_set_to_zero(model->dldXo, model->S);
  vector_set_to_zero(model->dldXi, model->S);
  vector_set_to_zero(model->dldXf, model->S);
}

void lstm_zero_d_next(lstm_values_next_cache_t * d_next, 
  int inputs, int neurons)
{
  vector_set_to_zero(d_next->dldh_next, neurons);
  vector_set_to_zero(d_next->dldc_next, neurons);
  vector_set_to_zero(d_next->dldY_pass, inputs);
}

void lstm_next_state_copy(lstm_values_state_t * state, lstm_values_cache_t * cache, int neurons, int write)
{
  if ( write ) {
    // Write to the state carrying unit
    copy_vector(state->h, cache->h, neurons);
    copy_vector(state->c, cache->c, neurons);
  } else {
    // Withdraw from the state carrying unit
    copy_vector(cache->h, state->h, neurons);
    copy_vector(cache->c, state->c, neurons);
  }

}

void lstm_cache_container_set_start(lstm_values_cache_t * cache, int neurons)
{
  // State variables set to zero
  vector_set_to_zero(cache->h, neurons);
  vector_set_to_zero(cache->c, neurons);

}

void lstm_store_net_layers(lstm_model_t** model, FILE *fp, unsigned int layers)
{
  unsigned int p = 0;

  while ( p < layers ) {

#ifdef STORE_NET_AS_ASCII
    vector_store_ascii(model[p]->Wy, model[p]->Y * model[p]->N, fp);
    vector_store_ascii(model[p]->Wi, model[p]->N * model[p]->S, fp);
    vector_store_ascii(model[p]->Wc, model[p]->N * model[p]->S, fp);
    vector_store_ascii(model[p]->Wo, model[p]->N * model[p]->S, fp);
    vector_store_ascii(model[p]->Wf, model[p]->N * model[p]->S, fp);

    vector_store_ascii(model[p]->by, model[p]->Y, fp);
    vector_store_ascii(model[p]->bi, model[p]->N, fp);
    vector_store_ascii(model[p]->bc, model[p]->N, fp);
    vector_store_ascii(model[p]->bf, model[p]->N, fp);
    vector_store_ascii(model[p]->bo, model[p]->N, fp);
#else
    vector_store(model[p]->Wy, model[p]->Y * model[p]->N, fp);
    vector_store(model[p]->Wi, model[p]->N * model[p]->S, fp);
    vector_store(model[p]->Wc, model[p]->N * model[p]->S, fp);
    vector_store(model[p]->Wo, model[p]->N * model[p]->S, fp);
    vector_store(model[p]->Wf, model[p]->N * model[p]->S, fp);

    vector_store(model[p]->by, model[p]->Y, fp);
    vector_store(model[p]->bi, model[p]->N, fp);
    vector_store(model[p]->bc, model[p]->N, fp);
    vector_store(model[p]->bf, model[p]->N, fp);
    vector_store(model[p]->bo, model[p]->N, fp);
#endif

    ++p;
  }
}

void lstm_store_net_layers_as_json(lstm_model_t** model, const char * filename, 
  const char *set_name, set_t *set, unsigned int layers)
{
  FILE * fp;
  unsigned int p = 0;

  fp = fopen(filename, "w");

  if ( fp == NULL ) {
    printf("Failed to open file: %s for writing.\n", filename);
    return;
  }

  fprintf(fp, "{\n\"%s\": ", set_name);
  set_store_as_json(set, fp);

  fprintf(fp, ",\n\"LSTM layers\": %d,\n", layers);

  while ( p < layers ) {

    if ( p > 0 ) 
      fprintf(fp, ",\n");

    fprintf(fp, "\"Layer %d\": {\n", p+1);

    fprintf(fp, "\t\"Wy\": ");
    vector_store_as_matrix_json(model[p]->Wy, model[p]->Y, model[p]->N, fp);
    fprintf(fp, ",\n\t\"Wi\": ");
    vector_store_as_matrix_json(model[p]->Wi, model[p]->N, model[p]->S, fp);
    fprintf(fp, ",\n\t\"Wc\": ");
    vector_store_as_matrix_json(model[p]->Wc, model[p]->N, model[p]->S, fp);
    fprintf(fp, ",\n\t\"Wo\": ");
    vector_store_as_matrix_json(model[p]->Wo, model[p]->N, model[p]->S, fp);
    fprintf(fp, ",\n\t\"Wf\": ");
    vector_store_as_matrix_json(model[p]->Wf, model[p]->N, model[p]->S, fp);

    fprintf(fp, ",\n\t\"by\": ");
    vector_store_json(model[p]->by, model[p]->Y, fp);
    fprintf(fp, ",\n\t\"bi\": ");
    vector_store_json(model[p]->bi, model[p]->N, fp);
    fprintf(fp, ",\n\t\"bc\": ");
    vector_store_json(model[p]->bc, model[p]->N, fp);
    fprintf(fp, ",\n\t\"bf\": ");
    vector_store_json(model[p]->bf, model[p]->N, fp);
    fprintf(fp, ",\n\t\"bo\": ");
    vector_store_json(model[p]->bo, model[p]->N, fp);

    fprintf(fp, "}\n");

    ++p;
  }

  fprintf(fp, "}\n");

  fclose(fp);

}

// Exits the program if EOF is encountered
static void e_lstm_fgets(char *str, size_t n, FILE *fp)
{
  if ( fgets(str,n,fp) == NULL ) {
    fprintf(stderr, "lstm_read error: unexpected EOF. \
Net-file incompatible with current version.\n"); 
    fflush(stderr);
    exit(1);
  }
}

void lstm_load(const char *path, set_t *set,
  lstm_model_parameters_t *params, lstm_model_t ***model)
{
  FILE * fp;
  char intContainer[10];
  int f;
  int F;
  int L;
  int l;
  int layerInputs[LSTM_MAX_LAYERS];
  int layerNodes[LSTM_MAX_LAYERS];
  int layerOutputs[LSTM_MAX_LAYERS];
  int FileVersion;

  fp = fopen(path, "r");

  if ( fp == NULL ) {
    printf("%s error: Failed to open file: %s for reading.\n", 
      __func__, path);
    exit(1);
  }

  initialize_set(set);

  /*
  * LSTM net file structure
  * File version   BINARY_FILE_VERSION
  * NbrFeatures    (F)
  * NbrLayers      (L)
  * Nodes in layer 1 (output layer)
  * Nodes in layer 2
  * ...
  * Nodes in layer L (input layer)
  * Feature Value 1 (int in ASCII [0-255])
  * Feature Value 2
  * ...
  * Feature Value F
  * --- From here on it is a blob of bytes ---
  * Layer 1: Wy
  * Layer 1: Wi
  * Layer 1: Wc
  * Layer 1: Wo
  * Layer 1: Wf
  * Layer 1: by
  * Layer 1: bi
  * Layer 1: bc
  * Layer 1: bf
  * Layer 1: bo
  * ...
  * Layer L: Wy
  * Layer L: Wi
  * ...
  */

  // Read file version
  e_lstm_fgets(intContainer, sizeof(intContainer), fp);
  FileVersion = atoi(intContainer);
  (void) FileVersion; // Not used yet, in this early stage
  // Read NbrFeatures
  e_lstm_fgets(intContainer, sizeof(intContainer), fp);
  F = atoi(intContainer);

  // Read NbrLayers
  e_lstm_fgets(intContainer, sizeof(intContainer), fp);
  L = atoi(intContainer);

  if ( L > LSTM_MAX_LAYERS ) {
    // This is too many layers
    fprintf(stderr, "%s error: Failed to load network, too many layers.\n", 
      __func__);
    exit(1);
  }

  // Setting the number of layers among the parameters
  params->layers = L;

  l = 0;
  while ( l < L ) {
    // Read number of inputs, nodes and ouputs in this layer
    e_lstm_fgets(intContainer, sizeof(intContainer), fp);
    layerInputs[l] = atoi(intContainer);
    e_lstm_fgets(intContainer, sizeof(intContainer), fp);
    layerNodes[l] = atoi(intContainer);
    e_lstm_fgets(intContainer, sizeof(intContainer), fp);
    layerOutputs[l] = atoi(intContainer);
    ++l;
  }

  // Setting the number of neurons
  // NOTE: it is the same for each layer (for now)
  params->neurons = layerNodes[0];

  // Import feature set
  f = 0;
  while ( f < F ) {
    e_lstm_fgets(intContainer, sizeof(intContainer), fp);
    set->values[f] = (char)atoi(intContainer);
    set->free[f] = 0;
    ++f;
  }

  assert(set_get_features(set) == layerInputs[L-1]);

  *model = (lstm_model_t**) malloc(L*sizeof(lstm_model_t*));
  if ( *model == NULL )
    lstm_init_fail("Failed to allocate resources for the net read\n");

  l = 0;
  while ( l < L ) {
    lstm_init_model(
      layerInputs[l],
      layerNodes[l],
      layerOutputs[l],
      &(*model)[l], 0, params);
    ++l;
  }

  lstm_read_net_layers(*model, fp, L);

  fclose(fp);
}

void lstm_store(const char *path, set_t *set,
  lstm_model_t **model, unsigned int layers)
{
  FILE * fp;
  int f;
  int F = set_get_features(set);
  unsigned int l;
  unsigned int L = layers;

  fp = fopen(path, "w");

  if ( fp == NULL ) {
    printf("%s error: Failed to open file: %s for writing.\n", 
      __func__, path);
    exit(1);
  }

  /*
  * LSTM net file structure
  * File version   BINARY_FILE_VERSION
  * NbrFeatures    (F)
  * NbrLayers      (L)
  * Inputs  layer  1 (output layer)
  * Nodes   layer  1
  * outputs layer  1
  * Inputs  layer  2
  * Nodes   layer  2
  * outputs layer  2
  * ...
  * Inputs  layer  L (input layer)
  * Nodes   layer  L
  * outputs layer  L
  * Feature Value  1 (int in ASCII [0-255])
  * Feature Value  2
  * ...
  * Feature Value  F
  * --- From here on it is a blob of bytes ---
  * Layer 1: Wy
  * Layer 1: Wi
  * Layer 1: Wc
  * Layer 1: Wo
  * Layer 1: Wf
  * Layer 1: by
  * Layer 1: bi
  * Layer 1: bc
  * Layer 1: bf
  * Layer 1: bo
  * ...
  * Layer L: Wy
  * Layer L: Wi
  * ...
  */

  // Write file version
  fprintf(fp, "%d\r\n", BINARY_FILE_VERSION);

  // Write NbrFeatures
  fprintf(fp, "%d\r\n", F);

  // Write NbrLayers
  fprintf(fp, "%d\r\n", L);

  l = 0;
  while ( l < L ) {
    // write number of inputs, nodes and outputs in this layer
    fprintf(fp, "%d\r\n%d\r\n%d\r\n",
      model[l]->X, model[l]->N, model[l]->Y);
    ++l;
  }

  // Write feature set
  f = 0;
  while ( f < F ) {
    fprintf(fp, "%d\r\n", set->values[f]);
    ++f;
  }

  // Write the network weights
  lstm_store_net_layers(model, fp, L);

  fclose(fp);
}

// Function to reinitialize an LSTM model by expanding its input and output layers.
// This is done by adding new features to the first and last layers of the model,
// with the weights initialized randomly for the new features.
int lstm_reinit_model(
  lstm_model_t** model,           // Pointer to the LSTM model layers
  unsigned int layers,            // Number of LSTM layers
  unsigned int previousNbrFeatures, // Previous number of features (input size)
  unsigned int newNbrFeatures      // New number of features (expanded input size)
) 
{
  // Local variables to hold LSTM model layer pointers
  lstm_model_t* modelInputs;   // Pointer to the last layer (input-related layer)
  lstm_model_t* modelOutputs;  // Pointer to the first layer (output-related layer)

  // Calculate old and new input sizes (S) and output dimensions (Y)
  int Sold = model[layers - 1]->S;   // Old input size (S)
  int Snew = newNbrFeatures + model[layers - 1]->N; // New input size (S) after adding new features
  int Nin = model[layers - 1]->N;    // Number of neurons in the input layer
  int Nout;                          // Number of neurons in the output layer
  int Yold = previousNbrFeatures;    // Previous output size
  int Ynew = newNbrFeatures;         // New output size (expanded)

  // Variables to hold the new randomly initialized weight vectors
  float *newVectorWf;  // Forget gate weights
  float *newVectorWi;  // Input gate weights
  float *newVectorWc;  // Cell state gate weights
  float *newVectorWo;  // Output gate weights
  float *newVectorWy;  // Output weight matrix

  int i, n;  // Loop counters

  // Sanity checks: If there are no layers, return with an error (-1)
  if (layers == 0)
    return -1;

  // Additional sanity checks: Ensure the new number of features is greater than the previous number
  if (previousNbrFeatures == newNbrFeatures || previousNbrFeatures > newNbrFeatures)
    return -1;

  // Use assertions to guarantee that the input conditions are valid
  assert(previousNbrFeatures < newNbrFeatures);  // Ensure new features are greater than old
  assert(Sold < Snew);  // Ensure the new input size is greater than the old

  // Initialize the number of neurons in the output layer
  Nout = model[0]->N;

  // Set the pointers for the output and input layers
  modelOutputs = model[0];         // First layer (output-related layer)
  modelInputs = model[layers - 1]; // Last layer (input-related layer)

  // Step 1: Reallocate the weight vectors that depend on the input size (Snew)
  // Allocate new random weights for the gates (Forget, Input, Cell state, Output)
  newVectorWf = get_random_vector(Nin * Snew, Snew * 5);  // Forget gate weights
  newVectorWi = get_random_vector(Nin * Snew, Snew * 5);  // Input gate weights
  newVectorWc = get_random_vector(Nin * Snew, Snew * 5);  // Cell state gate weights
  newVectorWo = get_random_vector(Nin * Snew, Snew * 5);  // Output gate weights

  // Copy old weights to the new expanded vectors for each gate
  n = 0;
  while (n < Nin) {  // Iterate through each neuron
    i = 0;
    while (i < Sold) {  // Copy the old weights (Sold) into the new vector (Snew)
      newVectorWf[n * Snew + i] = modelInputs->Wf[n * Sold + i];
      newVectorWi[n * Snew + i] = modelInputs->Wi[n * Sold + i];
      newVectorWc[n * Snew + i] = modelInputs->Wc[n * Sold + i];
      newVectorWo[n * Snew + i] = modelInputs->Wo[n * Sold + i];
      ++i;
    }
    ++n;
  }

  // Free the memory of the old weight vectors and gradient variables
  free(modelInputs->Wf);
  free(modelInputs->Wi);
  free(modelInputs->Wc);
  free(modelInputs->Wo);
  free(modelInputs->dldXc);
  free(modelInputs->dldXo);
  free(modelInputs->dldXi);
  free(modelInputs->dldXf);
  free(modelInputs->Wfm);
  free(modelInputs->Wim);
  free(modelInputs->Wcm);
  free(modelInputs->Wom);

  // Assign the new weight vectors to the input layer (modelInputs)
  modelInputs->Wf = newVectorWf;
  modelInputs->Wi = newVectorWi;
  modelInputs->Wc = newVectorWc;
  modelInputs->Wo = newVectorWo;

  // Initialize new zero vectors for gradients (dldX) and momentum (Wm) based on the new input size
  modelInputs->dldXc = get_zero_vector(Snew);
  modelInputs->dldXo = get_zero_vector(Snew);
  modelInputs->dldXi = get_zero_vector(Snew);
  modelInputs->dldXf = get_zero_vector(Snew);

  modelInputs->Wfm = get_zero_vector(Nin * Snew);
  modelInputs->Wim = get_zero_vector(Nin * Snew);
  modelInputs->Wcm = get_zero_vector(Nin * Snew);
  modelInputs->Wom = get_zero_vector(Nin * Snew);

  // Step 2: Reallocate the weight vector that depends on the output size (Ynew)
  newVectorWy = get_random_vector(Ynew * Nout, Nout);  // New output weight matrix

  // Copy the old output weights into the new expanded output weight matrix
  n = 0;
  while (n < Yold) {  // Iterate through the old output size (Yold)
    i = 0;
    while (i < Nout) {  // Copy the existing weights from the old matrix
      newVectorWy[n * Nout + i] = modelOutputs->Wy[n * Nout + i];
      ++i;
    }
    ++n;
  }

  // Free the memory of the old output weight matrix and its associated variables
  free(modelOutputs->Wy);
  free(modelOutputs->by);
  free(modelOutputs->Wym);
  free(modelOutputs->bym);

  // Assign the new output weight matrix and reinitialize biases and momentum
  modelOutputs->Wy = newVectorWy;
  modelOutputs->by = get_zero_vector(Ynew);  // Initialize output bias
  modelOutputs->Wym = get_zero_vector(Ynew * Nout);  // Initialize output weight momentum
  modelOutputs->bym = get_zero_vector(Ynew);  // Initialize output bias momentum

  // Step 3: Update the input and output layer dimensions
  modelInputs->X = newNbrFeatures;  // Update the input size to the new number of features
  modelInputs->S = Snew;  // Update the input size including new features
  modelOutputs->Y = newNbrFeatures;  // Update the output size to the new number of features

  return 0;  // Return success
}

// Function to read LSTM model parameters (weights and biases) from a file for each layer.
// This function reads either ASCII or binary formats depending on the `STORE_NET_AS_ASCII` flag.
void lstm_read_net_layers(lstm_model_t** model, FILE *fp, unsigned int layers)
{
  // Initialize the layer counter to zero
  unsigned int p = 0;

  // Loop through all the LSTM layers to read their weights and biases
  while (p < layers) {

    // If the model is stored in ASCII format
    #ifdef STORE_NET_AS_ASCII
      // Read weights for each gate (Wy, Wi, Wc, Wo, Wf) and the biases from the file in ASCII
      vector_read_ascii(model[p]->Wy, model[p]->Y * model[p]->N, fp); // Output weights
      vector_read_ascii(model[p]->Wi, model[p]->N * model[p]->S, fp); // Input gate weights
      vector_read_ascii(model[p]->Wc, model[p]->N * model[p]->S, fp); // Cell state weights
      vector_read_ascii(model[p]->Wo, model[p]->N * model[p]->S, fp); // Output gate weights
      vector_read_ascii(model[p]->Wf, model[p]->N * model[p]->S, fp); // Forget gate weights

      // Read biases for each gate
      vector_read_ascii(model[p]->by, model[p]->Y, fp); // Output bias
      vector_read_ascii(model[p]->bi, model[p]->N, fp); // Input gate bias
      vector_read_ascii(model[p]->bc, model[p]->N, fp); // Cell state bias
      vector_read_ascii(model[p]->bf, model[p]->N, fp); // Forget gate bias
      vector_read_ascii(model[p]->bo, model[p]->N, fp); // Output gate bias
    #else
      // If the model is stored in binary format, read the weights and biases
      vector_read(model[p]->Wy, model[p]->Y * model[p]->N, fp);
      vector_read(model[p]->Wi, model[p]->N * model[p]->S, fp);
      vector_read(model[p]->Wc, model[p]->N * model[p]->S, fp);
      vector_read(model[p]->Wo, model[p]->N * model[p]->S, fp);
      vector_read(model[p]->Wf, model[p]->N * model[p]->S, fp);

      // Read biases for each gate in binary format
      vector_read(model[p]->by, model[p]->Y, fp);
      vector_read(model[p]->bi, model[p]->N, fp);
      vector_read(model[p]->bc, model[p]->N, fp);
      vector_read(model[p]->bf, model[p]->N, fp);
      vector_read(model[p]->bo, model[p]->N, fp);
    #endif

    // Move to the next layer
    ++p;
  }
}


// Function to run the LSTM forward propagation and output a sequence of characters to a file.
// It generates `numbers_to_display` outputs starting with the character `first`.
void lstm_output_string_layers_to_file(FILE * fp, lstm_model_t ** model_layers, 
  set_t* char_index_mapping, int first, int numbers_to_display, int layers)
{
  lstm_values_cache_t ***caches_layer;  // 3D array to store the cache for each LSTM layer
  int i = 0, count, index, p = 0, b = 0; // Loop counters
  int input = set_indx_to_char(char_index_mapping, first);  // Get the first input character's index
  int Y = model_layers[0]->Y;  // Number of output classes (characters)
  int N = model_layers[0]->N;  // Number of neurons in the first layer

#ifdef WINDOWS
  float *first_layer_input;  // Input buffer for the first layer (for Windows systems)
#else
  float first_layer_input[Y];  // Input buffer for other systems
#endif

  // If the file pointer is NULL, return early (no file to write to)
  if (fp == NULL)
    return;

#ifdef WINDOWS
  // For Windows systems, allocate memory dynamically for the first layer input
  first_layer_input = malloc(Y * sizeof(float));

  if (first_layer_input == NULL) {
    fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n", 
    __FILE__, __func__, __LINE__, Y * sizeof(float));
    exit(1);  // Exit if memory allocation fails
  }
#endif

  // Allocate memory for the cache for each LSTM layer
  caches_layer = e_calloc(layers, sizeof(lstm_values_cache_t**));

  // Initialize cache for each layer and for both the current and next time steps (float buffering)
  p = 0;
  while (p < layers) {
    caches_layer[p] = e_calloc(2, sizeof(lstm_values_cache_t*));  // Allocate memory for the cache

    b = 0;
    while (b < 2) {
      // Initialize the cache for the input, hidden state, and output at each layer
      caches_layer[p][b] = lstm_cache_container_init(
        model_layers[p]->X, 
        model_layers[p]->N,
        model_layers[p]->Y);
      ++b;
    }
    ++p;
  }

  // Set the initial state for the first cache (both are set to the same starting state)
  lstm_cache_container_set_start(caches_layer[0][0], N);
  lstm_cache_container_set_start(caches_layer[0][0], N);

  // Generate and output `numbers_to_display` characters to the file
  while (i < numbers_to_display) {

    // Convert the character to an index in the character mapping
    index = set_char_to_indx(char_index_mapping, input);

    // Initialize the first layer's input with zeros
    count = 0;
    while (count < Y) {
      first_layer_input[count] = 0.0;
      ++count;
    }

    // Set the one-hot vector for the input character
    first_layer_input[index] = 1.0;

    // Forward propagate through all the layers, starting from the last
    p = layers - 1;
    lstm_forward_propagate_cuda(model_layers[p], first_layer_input, 
      caches_layer[p][i % 2], caches_layer[p][(i+1)%2], p == 0);

    // Propagate forward through remaining layers
    if (p > 0) {
      --p;
      while (p >= 0) {
        lstm_forward_propagate_cuda(model_layers[p], caches_layer[p+1][(i+1)%2]->probs, 
          caches_layer[p][i % 2], caches_layer[p][(i+1)%2], p == 0);
        --p;
      }
      p = 0;
    }

    // Select the next input character based on probabilities
    input = set_probability_choice(char_index_mapping, caches_layer[p][(i+1)%2]->probs);
    fprintf(fp, "%c", input);  // Write the character to the file

    ++i;
  }

  // Free memory for the cache after forward propagation
  p = 0;
  while (p < layers) {
    b = 0;
    while (b < 2) {
      lstm_cache_container_free(caches_layer[p][b]);  // Free each cache container
      free(caches_layer[p][b]);
      ++b;
    }
    free(caches_layer[p]);
    ++p;
  }

  free(caches_layer);  // Free the main cache array
#ifdef WINDOWS
  free(first_layer_input);  // Free the input buffer for Windows systems
#endif

}

// Function to run the LSTM model forward propagation and output a sequence of characters
// This function generates a sequence of `numbers_to_display` characters starting with the character `first`
// The sequence is printed to the console.

void lstm_output_string_layers(lstm_model_t ** model_layers, set_t* char_index_mapping, 
  int first, int numbers_to_display, int layers)
{
  lstm_values_cache_t ***caches_layer;  // A 3D array to store the cache for each LSTM layer
  int i = 0, count, index, p = 0, b = 0; // Loop counters
  int input = set_indx_to_char(char_index_mapping, first);  // Get the index of the first input character
  int Y = model_layers[0]->Y;  // The output size (number of possible characters)
  int N = model_layers[0]->N;  // The number of neurons in the first layer

#ifdef WINDOWS
  float *first_layer_input;  // Input buffer for the first layer (for Windows systems)
#else
  float first_layer_input[Y];  // Input buffer for non-Windows systems
#endif

#ifdef WINDOWS
  // For Windows systems, allocate memory dynamically for the first layer input
  first_layer_input = malloc(Y * sizeof(float));

  // Check if memory allocation was successful
  if (first_layer_input == NULL) {
    fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n", 
      __FILE__, __func__, __LINE__, Y * sizeof(float));
    exit(1);  // Exit if memory allocation fails
  }
#endif

  // Allocate memory for the cache of each LSTM layer, storing two time steps for forward propagation
  caches_layer = e_calloc(layers, sizeof(lstm_values_cache_t**));

  // Initialize cache for each layer and for both the current and next time steps (float buffering)
  p = 0;
  while (p < layers) {
    caches_layer[p] = e_calloc(2, sizeof(lstm_values_cache_t*));  // Allocate memory for two time steps

    b = 0;
    while (b < 2) {
      // Initialize the cache for the input, hidden state, and output for each layer
      caches_layer[p][b] = lstm_cache_container_init(
        model_layers[p]->X, model_layers[p]->N, model_layers[p]->Y); 
      ++b;
    }
    ++p;
  }

  // Set the initial state for the first cache
  lstm_cache_container_set_start(caches_layer[0][0], N);
  lstm_cache_container_set_start(caches_layer[0][0], N);

  // Loop to generate and output `numbers_to_display` characters
  while (i < numbers_to_display) {

    // Convert the character to an index in the character mapping
    index = set_char_to_indx(char_index_mapping, input);

    // Initialize the first layer's input with zeros (one-hot encoding)
    count = 0;
    while (count < Y) {
      first_layer_input[count] = 0.0;
      ++count;
    }

    // Error handling: If the character index is invalid, set it to 0 and print an error message
    if (index < 0) {
      index = 0;
      printf("%s.%s unexpected input char: '%c', (%d)\r\n", 
        __FILE__, __func__, input, input);
    }

    // Set the input for the current character (one-hot encoding)
    first_layer_input[index] = 1.0;

    // Forward propagate the input through the last layer of the LSTM
    p = layers - 1;
    lstm_forward_propagate(model_layers[p], first_layer_input, 
      caches_layer[p][i % 2], caches_layer[p][(i + 1) % 2], p == 0);

    // Propagate forward through the remaining layers if there are multiple layers
    if (p > 0) {
      --p;
      while (p >= 0) {
        // Propagate through the current layer using the probabilities from the previous layer
        lstm_forward_propagate(model_layers[p], 
          caches_layer[p + 1][(i + 1) % 2]->probs, 
          caches_layer[p][i % 2], caches_layer[p][(i + 1) % 2], 
          p == 0);	
        --p;
      }
      p = 0;
    }

    // Select the next character based on the probabilities generated by the last layer
    input = set_probability_choice(char_index_mapping, 
      caches_layer[p][(i + 1) % 2]->probs);
    
    // Print the generated character to the console
    printf("%c", input);

    ++i;  // Increment the character count
  }

  // Free memory for the cache after the forward propagation is completed
  p = 0;
  while (p < layers) {
    b = 0;
    while (b < 2) {
      // Free each cache container
      lstm_cache_container_free(caches_layer[p][b]);
      free(caches_layer[p][b]);
      ++b;
    }
    free(caches_layer[p]);
    ++p;
  }

  // Free the main cache array
  free(caches_layer);

#ifdef WINDOWS
  // Free the input buffer for Windows systems
  free(first_layer_input);
#endif
}

// Function to perform LSTM forward propagation based on an input string
// and generate a sequence of output characters. The function processes
// the `input_string` through the LSTM layers and then continues generating 
// additional characters to a specified `out_length`.

void lstm_output_string_from_string(lstm_model_t **model_layers, set_t* char_index_mapping,
  char * input_string, int layers, int out_length)
{
  lstm_values_cache_t ***caches_layers;  // 3D array to store the cache for each LSTM layer
  int i = 0, count, index, in_len;  // Variables for loop counters and index manipulation
  char input;  // Holds the current input character for the LSTM model
  int Y = model_layers[0]->Y;  // Number of output units (possible characters)

  int p = 0;  // Layer loop counter

  // For non-Windows systems, use a statically sized array for the input vector
  float first_layer_input[Y];

  // Allocate memory for the cache for each layer (with float buffering for two time steps)
  caches_layers = e_calloc(layers, sizeof(lstm_values_cache_t**));

  // Initialize caches for each LSTM layer and for both current and next time steps
  while (p < layers) {
    caches_layers[p] = e_calloc(2, sizeof(lstm_values_cache_t*));  // float buffer for each layer

    i = 0; 
    while (i < 2) {
      // Initialize the cache for the input, hidden state, and output at each layer
      caches_layers[p][i] = lstm_cache_container_init(
        model_layers[p]->X, model_layers[0]->N, model_layers[0]->Y);
      ++i;
    }

    ++p;
  }

  // Step 1: Process the input string
  in_len = strlen(input_string);  // Get the length of the input string
  i = 0;

  // Loop through each character of the input string
  while (i < in_len) {
    // Print the current input character
    printf("%c", input_string[i]);

    // Convert the input character to its corresponding index in the character mapping
    index = set_char_to_indx(char_index_mapping, input_string[i]);

    // Create a one-hot encoding for the current input character
    count = 0;
    while (count < Y) {
      first_layer_input[count] = (count == index) ? 1.0 : 0.0;  // One-hot encoding
      ++count;
    }

    // Perform forward propagation through the last LSTM layer
    p = layers - 1;
    lstm_forward_propagate_cuda(model_layers[p], first_layer_input, 
      caches_layers[p][i % 2], caches_layers[p][(i + 1) % 2], p == 0);

    // Propagate through the remaining layers, if any
    if (p > 0) {
      --p;
      while (p >= 0) {
        lstm_forward_propagate_cuda(model_layers[p], 
          caches_layers[p+1][(i+1)%2]->probs, 
          caches_layers[p][i % 2], caches_layers[p][(i+1)%2], 
          p == 0);
        --p;
      }
      p = 0;
    }

    ++i;  // Move to the next character in the input string
  }

  // Step 2: Generate additional characters based on the LSTM model
  input = set_probability_choice(char_index_mapping, caches_layers[0][i % 2]->probs);  // First generated character
  printf("%c", input);  // Print the first generated character
  i = 0;

  // Generate `out_length` characters after processing the input string
  while (i < out_length) {
    // Convert the character to its index in the character mapping
    index = set_char_to_indx(char_index_mapping, input);

    // Create a one-hot encoding for the current character
    count = 0;
    while (count < Y) {
      first_layer_input[count] = (count == index) ? 1.0 : 0.0;
      ++count;
    }

    // Forward propagate through the last LSTM layer
    p = layers - 1;
    lstm_forward_propagate_cuda(model_layers[p], first_layer_input, 
      caches_layers[p][i % 2], caches_layers[p][(i + 1) % 2], p == 0);

    // Forward propagate through the remaining layers
    if (p > 0) {
      --p;
      while (p >= 0) {
        lstm_forward_propagate_cuda(model_layers[p], 
          caches_layers[p + 1][(i + 1) % 2]->probs, 
          caches_layers[p][i % 2], caches_layers[p][(i + 1) % 2], 
          p == 0);
        --p;
      }
      p = 0;
    }

    // Choose the next character based on the output probabilities and print it
    input = set_probability_choice(char_index_mapping, caches_layers[p][(i+1) % 2]->probs);
    printf("%c", input);  // Print the next character

    ++i;  // Move to the next output character
  }

  // Print a newline after the generated sequence
  printf("\n");

  // Step 3: Clean up allocated memory
  p = 0;
  while (p < layers) {
    i = 0; 
    while (i < 2) {
      // Free the cache for each time step in each layer
      lstm_cache_container_free(caches_layers[p][i]); 
      free(caches_layers[p][i]);
      ++i;
    }
    free(caches_layers[p]);
    ++p;
  }

  // Free the main cache array
  free(caches_layers);

#ifdef WINDOWS
  // Free the input buffer for Windows systems
  free(first_layer_input);
#endif
}

// Function to store the training progress of an LSTM model by appending the
// current iteration number and loss value to a file. The data is written in CSV format.
void lstm_store_progress(const char* filename, unsigned int n, float loss)
{
  FILE *fp;  // File pointer to handle file operations

  // Open the file in append mode ("a"), meaning that new data will be added
  // to the end of the file without overwriting any existing data.
  fp = fopen(filename, "a");

  // Check if the file was successfully opened
  if (fp != NULL) {
    // If the file is open, write the current iteration (n) and loss value (loss)
    // to the file in CSV format, where the iteration number and loss are separated by a comma.
    // Example: "100,0.345678\n" will be written for iteration 100 with a loss of 0.345678.
    fprintf(fp, "%u,%lf\n", n, loss);

    // After writing to the file, close it to ensure that the data is saved and
    // the file is no longer being accessed.
    fclose(fp);
  }
  // If the file cannot be opened, the function does nothing and exits.
}


// Function to apply L2 regularization to an LSTM model's gradients.
// Regularization helps to prevent overfitting by penalizing large weight values.
void lstm_model_regularization(lstm_model_t* model, lstm_model_t* gradients)
{
  // Get the regularization parameter lambda from the model's parameters.
  // Lambda controls the strength of the regularization; a higher lambda means stronger regularization.
  float lambda = model->params->lambda;

  // For each weight matrix (Wy, Wi, Wc, Wo, Wf) and bias vector (by, bi, bc, bo, bf),
  // apply the L2 regularization by adding the scaled model weights to the gradients.
  // L2 regularization is done by adding lambda * weights to the corresponding gradient.

  // Regularizing the output weight matrix Wy
  vectors_add_scalar_multiply(gradients->Wy, model->Wy, model->Y * model->N, lambda);
  
  // Regularizing the input gate weight matrix Wi
  vectors_add_scalar_multiply(gradients->Wi, model->Wi, model->N * model->S, lambda);
  
  // Regularizing the cell state gate weight matrix Wc
  vectors_add_scalar_multiply(gradients->Wc, model->Wc, model->N * model->S, lambda);
  
  // Regularizing the output gate weight matrix Wo
  vectors_add_scalar_multiply(gradients->Wo, model->Wo, model->N * model->S, lambda);
  
  // Regularizing the forget gate weight matrix Wf
  vectors_add_scalar_multiply(gradients->Wf, model->Wf, model->N * model->S, lambda);

  // Regularizing the output bias by
  vectors_add_scalar_multiply(gradients->by, model->by, model->Y, lambda);
  
  // Regularizing the input gate bias bi
  vectors_add_scalar_multiply(gradients->bi, model->bi, model->N, lambda);
  
  // Regularizing the cell state bias bc
  vectors_add_scalar_multiply(gradients->bc, model->bc, model->N, lambda);
  
  // Regularizing the output gate bias bo
  vectors_add_scalar_multiply(gradients->bo, model->bo, model->N, lambda);
  
  // Regularizing the forget gate bias bf
  vectors_add_scalar_multiply(gradients->bf, model->bf, model->N, lambda);
}

//						model, number of training points, X_train, Y_train
// Function to train an LSTM model using the provided training data and parameters.
void lstm_train(lstm_model_t** model_layers, lstm_model_parameters_t *params,
  set_t* char_index_mapping, unsigned int training_points,
  int* X_train, int* Y_train, unsigned int layers, float *loss_out)
{
  // Variable declarations for tracking progress, loss, iterations, etc.
  unsigned int p, i = 0, b = 0, q = 0, e1 = 0, e2 = 0, e3, record_iteration = 0, tmp_count, trailing;
  unsigned long n = 0, epoch = 0;
  float loss = -1, loss_tmp = 0.0, record_keeper = 0.0;
  float initial_learning_rate = params->learning_rate;
  time_t time_iter;  // Variable for tracking time for progress output
  char time_buffer[40];  // Buffer for time-related output
  unsigned long iterations = params->iterations;  // Total number of iterations
  unsigned long epochs = params->epochs;  // Total number of epochs
  int stateful = params->stateful, decrease_lr = params->decrease_lr;

  // Configurations for output printing during training
  int print_progress = params->print_progress;
  int print_progress_iterations = params->print_progress_iterations;
  int print_progress_sample_output = params->print_progress_sample_output;
  int print_progress_to_file = params->print_progress_to_file;
  int print_progress_number_of_chars = params->print_progress_number_of_chars;
  char *print_progress_to_file_name = params->print_sample_output_to_file_name;
  char *print_progress_to_file_arg = params->print_sample_output_to_file_arg;
  int store_progress_every_x_iterations = params->store_progress_every_x_iterations;
  char *store_progress_file_name = params->store_progress_file_name;
  int store_network_every = params->store_network_every;

  // Variables to store LSTM state, gradients, and caches for forward/backward passes
  lstm_values_state_t **stateful_d_next = NULL;
  lstm_values_cache_t ***cache_layers;
  lstm_values_next_cache_t **d_next_layers;
  lstm_model_t **gradient_layers, **gradient_layers_entry, **M_layers = NULL, **R_layers = NULL;
  lstm_model_t **d_model_layers;
  
  cudaMallocManaged((void **)&d_model_layers, layers * sizeof(lstm_model_t*), cudaMemAttachGlobal);

  for(unsigned int i = 0; i < layers; i++){
    int S = model_layers[i]->N + model_layers[i]->X;
    cudaMallocManaged((void **)&d_model_layers[i], sizeof(lstm_model_t), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wf, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wi, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wc, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wo, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wy, model_layers[i]->N * model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bf, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bi, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bo, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->by, model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->params, sizeof(lstm_model_parameters_t), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldhf, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldhi, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldhc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldho, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldh, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldXc, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldXo, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldXi, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->dldXf, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wfm, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wim, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wcm, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wom, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->Wym, S * model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bfm, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bim, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bcm, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bom, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_model_layers[i]->bym, model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMemcpy(d_model_layers[i]->Wf, model_layers[i]->Wf, model_layers[i]->N * S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wi, model_layers[i]->Wi, model_layers[i]->N * S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wc, model_layers[i]->Wc, model_layers[i]->N * S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wo, model_layers[i]->Wo, model_layers[i]->N * S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wy, model_layers[i]->Wy, model_layers[i]->N * model_layers[i]->Y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bf, model_layers[i]->bf, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bi, model_layers[i]->bi, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bc, model_layers[i]->bc, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bo, model_layers[i]->bo, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->by, model_layers[i]->by, model_layers[i]->Y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->params, model_layers[i]->params, sizeof(lstm_model_parameters_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldhf, model_layers[i]->dldhf, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldhi, model_layers[i]->dldhi, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldhc, model_layers[i]->dldhc, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldho, model_layers[i]->dldho, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldc, model_layers[i]->dldc, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldh, model_layers[i]->dldh, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldXc, model_layers[i]->dldXc, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldXo, model_layers[i]->dldXo, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldXi, model_layers[i]->dldXi, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->dldXf, model_layers[i]->dldXf, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wfm, model_layers[i]->Wfm, model_layers[i]->N * S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wim, model_layers[i]->Wim, model_layers[i]->N * S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wcm, model_layers[i]->Wcm, model_layers[i]->N * S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wom, model_layers[i]->Wom, model_layers[i]->N * S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->Wym, model_layers[i]->Wym, S * model_layers[i]->Y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bfm, model_layers[i]->bfm, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bim, model_layers[i]->bim, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bcm, model_layers[i]->bcm, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bom, model_layers[i]->bom, model_layers[i]->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_model_layers[i]->bym, model_layers[i]->bym, model_layers[i]->Y * sizeof(float), cudaMemcpyHostToDevice);
    d_model_layers[i]->X = model_layers[i]->X;
    d_model_layers[i]->N = model_layers[i]->N;
    d_model_layers[i]->Y = model_layers[i]->Y;
    d_model_layers[i]->S = model_layers[i]->S;
  }

  // For non-Windows systems, use a statically sized array for the first layer's input
  //float first_layer_input[model_layers[0]->Y];
  float *first_layer_input;
  cudaMallocManaged((void **)&first_layer_input, model_layers[0]->Y * sizeof(float), cudaMemAttachGlobal);

  // Initialize stateful LSTM (if enabled), where states persist across mini-batches
  if (stateful) {
    //stateful_d_next = e_calloc(layers, sizeof(lstm_values_state_t*));
    cudaMallocManaged((void **)&stateful_d_next, layers * sizeof(lstm_values_state_t*), cudaMemAttachGlobal);

    i = 0;
    while (i < layers) {
      // Allocate space for LSTM states for each layer and initialize the state
      //stateful_d_next[i] = e_calloc(training_points / params->mini_batch_size + 1, sizeof(lstm_values_state_t));
      cudaMallocManaged((void **)&stateful_d_next[i], training_points / params->mini_batch_size + 1 * sizeof(lstm_values_state_t), cudaMemAttachGlobal);
      //lstm_values_state_init(&stateful_d_next[i], model_layers[i]->N);  // Initialize the state for each layer
      cudaMallocManaged((void **)&stateful_d_next[i]->c, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&stateful_d_next[i]->h, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMemset(stateful_d_next[i]->c, 0, model_layers[i]->N);
      cudaMemset(stateful_d_next[i]->h, 0, model_layers[i]->N);
      ++i;
    }
  }

  // Allocate and initialize cache layers for storing forward propagation values
  i = 0;
  //cache_layers = e_calloc(layers, sizeof(lstm_values_cache_t**));
  cudaMallocManaged((void **)&cache_layers, layers * sizeof(lstm_values_cache_t**), cudaMemAttachGlobal);
  while (i < layers) {
    // Each layer has cache memory for the forward pass values of the entire mini-batch
    //cache_layers[i] = e_calloc(params->mini_batch_size + 1, sizeof(lstm_values_cache_t*));
    cudaMallocManaged((void **)&cache_layers[i], (params->mini_batch_size + 1) * sizeof(lstm_values_cache_t*), cudaMemAttachGlobal);

    p = 0;
    while (p < params->mini_batch_size + 1) {
      // Initialize cache for input, hidden states, and output values of the LSTM layers
      //cache_layers[i][p] = lstm_cache_container_init(model_layers[i]->X, model_layers[i]->N, model_layers[i]->Y);
      cudaMallocManaged((void **)&cache_layers[i][p], sizeof(lstm_values_cache_t), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->probs, model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->probs_before_sigma, model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->c, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->h, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->c_old, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->h_old, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->X, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->hf, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->hi, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->ho, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->hc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&cache_layers[i][p]->tanh_c_cache, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMemset(cache_layers[i][p]->probs, 0, model_layers[i]->Y);
      cudaMemset(cache_layers[i][p]->probs_before_sigma, 0, model_layers[i]->Y);
      cudaMemset(cache_layers[i][p]->c, 0, model_layers[i]->N);
      cudaMemset(cache_layers[i][p]->h, 0, model_layers[i]->N);
      cudaMemset(cache_layers[i][p]->c_old, 0, model_layers[i]->N);
      cudaMemset(cache_layers[i][p]->h_old, 0, model_layers[i]->N);
      cudaMemset(cache_layers[i][p]->X, 0, (model_layers[i]->N+model_layers[i]->X));
      cudaMemset(cache_layers[i][p]->hf, 0, model_layers[i]->N);
      cudaMemset(cache_layers[i][p]->hi, 0, model_layers[i]->N);
      cudaMemset(cache_layers[i][p]->ho, 0, model_layers[i]->N);
      cudaMemset(cache_layers[i][p]->hc, 0, model_layers[i]->N);
      cudaMemset(cache_layers[i][p]->tanh_c_cache, 0, model_layers[i]->N);
      if (cache_layers[i][p] == NULL)
        lstm_init_fail("Failed to allocate memory for the caches\n");
      ++p;
    }
    ++i;
  }

  // Allocate memory for gradients
  //gradient_layers = e_calloc(layers, sizeof(lstm_model_t*));
  //gradient_layers_entry = e_calloc(layers, sizeof(lstm_model_t*));
  //d_next_layers = e_calloc(layers, sizeof(lstm_values_next_cache_t*));
  cudaMallocManaged((void **)&gradient_layers, layers * sizeof(lstm_model_t*), cudaMemAttachGlobal);
  cudaMallocManaged((void **)&gradient_layers_entry, layers * sizeof(lstm_model_t*), cudaMemAttachGlobal);
  cudaMallocManaged((void **)&d_next_layers, layers * sizeof(lstm_values_next_cache_t*), cudaMemAttachGlobal);

  // If Adam optimizer is selected, allocate memory for momentum and RMSProp caches (M and R)
  if (params->optimizer == OPTIMIZE_ADAM) {
    //M_layers = e_calloc(layers, sizeof(lstm_model_t*));
    //R_layers = e_calloc(layers, sizeof(lstm_model_t*));
    cudaMallocManaged((void **)&M_layers, layers * sizeof(lstm_model_t*), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&R_layers, layers * sizeof(lstm_model_t*), cudaMemAttachGlobal);
  }

  // Initialize LSTM models and caches for each layer
  i = 0;
  while (i < layers) {
    // Initialize gradient layers for each LSTM model
    int S = model_layers[i]->N + model_layers[i]->X;
    //lstm_init_model(model_layers[i]->X, model_layers[i]->N, model_layers[i]->Y, &gradient_layers[i], 1, params);
    cudaMallocManaged((void **)&gradient_layers[i], sizeof(lstm_model_t), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wf, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wi, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wc, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wo, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wy, model_layers[i]->N * model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bf, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bi, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bo, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->by, model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->params, sizeof(lstm_model_parameters_t), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldhf, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldhi, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldhc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldho, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldh, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldXc, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldXo, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldXi, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->dldXf, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wfm, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wim, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wcm, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wom, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->Wym, S * model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bfm, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bim, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bcm, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bom, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers[i]->bym, model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMemset(gradient_layers[i]->Wf, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers[i]->Wi, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers[i]->Wc, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers[i]->Wo, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers[i]->Wy, 0, model_layers[i]->N * model_layers[i]->Y);
    cudaMemset(gradient_layers[i]->bf, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->bi, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->bc, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->bo, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->by, 0, model_layers[i]->Y);
    cudaMemset(gradient_layers[i]->dldhf, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->dldhi, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->dldhc, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->dldho, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->dldc, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->dldh, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->dldXc, 0, (model_layers[i]->N+model_layers[i]->X));
    cudaMemset(gradient_layers[i]->dldXo, 0, (model_layers[i]->N+model_layers[i]->X));
    cudaMemset(gradient_layers[i]->dldXi, 0, (model_layers[i]->N+model_layers[i]->X));
    cudaMemset(gradient_layers[i]->dldXf, 0, (model_layers[i]->N+model_layers[i]->X));
    cudaMemset(gradient_layers[i]->Wfm, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers[i]->Wim, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers[i]->Wcm, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers[i]->Wom, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers[i]->Wym, 0, S * model_layers[i]->Y);
    cudaMemset(gradient_layers[i]->bfm, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->bim, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->bcm, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->bom, 0, model_layers[i]->N);
    cudaMemset(gradient_layers[i]->bym, 0, model_layers[i]->Y);
    gradient_layers[i]->X = model_layers[i]->X;
    gradient_layers[i]->N = model_layers[i]->N;
    gradient_layers[i]->Y = model_layers[i]->Y;
    gradient_layers[i]->S = model_layers[i]->N + model_layers[i]->X;
    cudaMemcpy(gradient_layers[i]->params, params, sizeof(lstm_model_parameters_t), cudaMemcpyHostToDevice);
    //lstm_init_model(model_layers[i]->X, model_layers[i]->N, model_layers[i]->Y, &gradient_layers_entry[i], 1, params);
    cudaMallocManaged((void **)&gradient_layers_entry[i], sizeof(lstm_model_t), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wf, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wi, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wc, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wo, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wy, model_layers[i]->N * model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bf, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bi, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bo, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->by, model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->params, sizeof(lstm_model_parameters_t), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldhf, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldhi, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldhc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldho, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldc, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldh, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldXc, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldXo, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldXi, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->dldXf, (model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wfm, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wim, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wcm, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wom, model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->Wym, S * model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bfm, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bim, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bcm, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bom, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&gradient_layers_entry[i]->bym, model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
    cudaMemset(gradient_layers_entry[i]->Wf, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers_entry[i]->Wi, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers_entry[i]->Wc, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers_entry[i]->Wo, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers_entry[i]->Wy, 0, model_layers[i]->N * model_layers[i]->Y);
    cudaMemset(gradient_layers_entry[i]->bf, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->bi, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->bc, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->bo, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->by, 0, model_layers[i]->Y);
    cudaMemset(gradient_layers_entry[i]->dldhf, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->dldhi, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->dldhc, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->dldho, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->dldc, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->dldh, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->dldXc, 0, (model_layers[i]->N+model_layers[i]->X));
    cudaMemset(gradient_layers_entry[i]->dldXo, 0, (model_layers[i]->N+model_layers[i]->X));
    cudaMemset(gradient_layers_entry[i]->dldXi, 0, (model_layers[i]->N+model_layers[i]->X));
    cudaMemset(gradient_layers_entry[i]->dldXf, 0, (model_layers[i]->N+model_layers[i]->X));
    cudaMemset(gradient_layers_entry[i]->Wfm, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers_entry[i]->Wim, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers_entry[i]->Wcm, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers_entry[i]->Wom, 0, model_layers[i]->N * S);
    cudaMemset(gradient_layers_entry[i]->Wym, 0, S * model_layers[i]->Y);
    cudaMemset(gradient_layers_entry[i]->bfm, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->bim, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->bcm, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->bom, 0, model_layers[i]->N);
    cudaMemset(gradient_layers_entry[i]->bym, 0, model_layers[i]->Y);
    gradient_layers_entry[i]->X = model_layers[i]->X;
    gradient_layers_entry[i]->N = model_layers[i]->N;
    gradient_layers_entry[i]->Y = model_layers[i]->Y;
    gradient_layers_entry[i]->S = model_layers[i]->N + model_layers[i]->X;
    cudaMemcpy(gradient_layers_entry[i]->params, params, sizeof(lstm_model_parameters_t), cudaMemcpyHostToDevice);

    // Initialize the next layer cache for backpropagation gradients
    //lstm_values_next_cache_init(&d_next_layers[i], model_layers[i]->N, model_layers[i]->X);
    cudaMallocManaged((void **)&d_next_layers[i], sizeof(lstm_values_next_cache_t), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_next_layers[i]->dldh_next, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_next_layers[i]->dldc_next, model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&d_next_layers[i]->dldY_pass, model_layers[i]->X * sizeof(float), cudaMemAttachGlobal);
    cudaMemset(d_next_layers[i]->dldh_next, 0, model_layers[i]->N);
    cudaMemset(d_next_layers[i]->dldc_next, 0, model_layers[i]->N);
    cudaMemset(d_next_layers[i]->dldY_pass, 0, model_layers[i]->X);

    // If using Adam optimizer, initialize M and R caches for each layer
    if (params->optimizer == OPTIMIZE_ADAM) {
      //lstm_init_model(model_layers[i]->X, model_layers[i]->N, model_layers[i]->Y, &M_layers[i], 1, params);
      cudaMallocManaged((void **)&M_layers[i], sizeof(lstm_model_t), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wf, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wi, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wc, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wo, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wy, d_model_layers[i]->N * model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bf, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bi, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bc, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bo, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->by, d_model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->params, sizeof(lstm_model_parameters_t), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldhf, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldhi, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldhc, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldho, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldc, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldh, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldXc, (d_model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldXo, (d_model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldXi, (d_model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->dldXf, (d_model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wfm, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wim, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wcm, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wom, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->Wym, S * d_model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bfm, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bim, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bcm, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bom, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&M_layers[i]->bym, d_model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMemset(M_layers[i]->Wf, 0, d_model_layers[i]->N * S);
      cudaMemset(M_layers[i]->Wi, 0, d_model_layers[i]->N * S);
      cudaMemset(M_layers[i]->Wc, 0, d_model_layers[i]->N * S);
      cudaMemset(M_layers[i]->Wo, 0, d_model_layers[i]->N * S);
      cudaMemset(M_layers[i]->Wy, 0, d_model_layers[i]->N * d_model_layers[i]->Y);
      cudaMemset(M_layers[i]->bf, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->bi, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->bc, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->bo, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->by, 0, d_model_layers[i]->Y);
      cudaMemset(M_layers[i]->dldhf, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->dldhi, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->dldhc, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->dldho, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->dldc, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->dldh, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->dldXc, 0, (d_model_layers[i]->N+d_model_layers[i]->X));
      cudaMemset(M_layers[i]->dldXo, 0, (d_model_layers[i]->N+d_model_layers[i]->X));
      cudaMemset(M_layers[i]->dldXi, 0, (d_model_layers[i]->N+d_model_layers[i]->X));
      cudaMemset(M_layers[i]->dldXf, 0, (d_model_layers[i]->N+d_model_layers[i]->X));
      cudaMemset(M_layers[i]->Wfm, 0, d_model_layers[i]->N * S);
      cudaMemset(M_layers[i]->Wim, 0, d_model_layers[i]->N * S);
      cudaMemset(M_layers[i]->Wcm, 0, d_model_layers[i]->N * S);
      cudaMemset(M_layers[i]->Wom, 0, d_model_layers[i]->N * S);
      cudaMemset(M_layers[i]->Wym, 0, S * d_model_layers[i]->Y);
      cudaMemset(M_layers[i]->bfm, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->bim, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->bcm, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->bom, 0, d_model_layers[i]->N);
      cudaMemset(M_layers[i]->bym, 0, d_model_layers[i]->Y);
      M_layers[i]->X = d_model_layers[i]->X;
      M_layers[i]->N = d_model_layers[i]->N;
      M_layers[i]->Y = d_model_layers[i]->Y;
      M_layers[i]->S = d_model_layers[i]->N + d_model_layers[i]->X;
      cudaMemcpy(M_layers[i]->params, params, sizeof(lstm_model_parameters_t), cudaMemcpyHostToDevice);

      //lstm_init_model(model_layers[i]->X, model_layers[i]->N, model_layers[i]->Y, &R_layers[i], 1, params);
      cudaMallocManaged((void **)&R_layers[i], sizeof(lstm_model_t), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wf, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wi, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wc, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wo, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wy, d_model_layers[i]->N * d_model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bf, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bi, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bc, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bo, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->by, d_model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->params, sizeof(lstm_model_parameters_t), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldhf, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldhi, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldhc, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldho, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldc, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldh, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldXc, (d_model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldXo, (d_model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldXi, (d_model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->dldXf, (d_model_layers[i]->N+model_layers[i]->X) * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wfm, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wim, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wcm, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wom, d_model_layers[i]->N * S * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->Wym, S * d_model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bfm, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bim, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bcm, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bom, d_model_layers[i]->N * sizeof(float), cudaMemAttachGlobal);
      cudaMallocManaged((void **)&R_layers[i]->bym, d_model_layers[i]->Y * sizeof(float), cudaMemAttachGlobal);
      cudaMemset(R_layers[i]->Wf, 0, d_model_layers[i]->N * S);
      cudaMemset(R_layers[i]->Wi, 0, d_model_layers[i]->N * S);
      cudaMemset(R_layers[i]->Wc, 0, d_model_layers[i]->N * S);
      cudaMemset(R_layers[i]->Wo, 0, d_model_layers[i]->N * S);
      cudaMemset(R_layers[i]->Wy, 0, d_model_layers[i]->N * d_model_layers[i]->Y);
      cudaMemset(R_layers[i]->bf, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->bi, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->bc, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->bo, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->by, 0, d_model_layers[i]->Y);
      cudaMemset(R_layers[i]->dldhf, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->dldhi, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->dldhc, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->dldho, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->dldc, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->dldh, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->dldXc, 0, (d_model_layers[i]->N+d_model_layers[i]->X));
      cudaMemset(R_layers[i]->dldXo, 0, (d_model_layers[i]->N+d_model_layers[i]->X));
      cudaMemset(R_layers[i]->dldXi, 0, (d_model_layers[i]->N+d_model_layers[i]->X));
      cudaMemset(R_layers[i]->dldXf, 0, (d_model_layers[i]->N+d_model_layers[i]->X));
      cudaMemset(R_layers[i]->Wfm, 0, d_model_layers[i]->N * S);
      cudaMemset(R_layers[i]->Wim, 0, d_model_layers[i]->N * S);
      cudaMemset(R_layers[i]->Wcm, 0, d_model_layers[i]->N * S);
      cudaMemset(R_layers[i]->Wom, 0, d_model_layers[i]->N * S);
      cudaMemset(R_layers[i]->Wym, 0, S * d_model_layers[i]->Y);
      cudaMemset(R_layers[i]->bfm, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->bim, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->bcm, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->bom, 0, d_model_layers[i]->N);
      cudaMemset(R_layers[i]->bym, 0, d_model_layers[i]->Y);
      R_layers[i]->X = d_model_layers[i]->X;
      R_layers[i]->N = d_model_layers[i]->N;
      R_layers[i]->Y = d_model_layers[i]->Y;
      R_layers[i]->S = d_model_layers[i]->N + d_model_layers[i]->X;
      cudaMemcpy(R_layers[i]->params, params, sizeof(lstm_model_parameters_t), cudaMemcpyHostToDevice);
    }
    ++i;
  }


  i = 0; b = 0;
  // Main training loop: continues until the total number of iterations is reached
  while (n < iterations) {

    // Check if the desired number of epochs is reached
    if (epochs && epoch >= epochs) {
      // Stop the training if the total number of epochs has been reached
      break;
    }

    b = i;  // Track the starting index of the mini-batch

    loss_tmp = 0.0;  // Reset temporary loss for the mini-batch

    // Loop through all LSTM layers
    q = 0;
    while (q < layers) {
      if (stateful) {
        // For stateful LSTM: initialize or copy the previous state
        if (q == 0){
          //lstm_cache_container_set_start(cache_layers[q][0], model_layers[q]->N);
          cudaMemset(cache_layers[q][0]->h, 0, d_model_layers[q]->N * sizeof(float));
          cudaMemset(cache_layers[q][0]->c, 0, d_model_layers[q]->N * sizeof(float));
        }
        else{
          //lstm_next_state_copy(stateful_d_next[q], cache_layers[q][0], model_layers[q]->N, 0);
          cudaMemcpy(cache_layers[q][0]->h, stateful_d_next[q]->h, d_model_layers[q]->N * sizeof(float), cudaMemcpyDeviceToDevice);
          cudaMemcpy(cache_layers[q][0]->c, stateful_d_next[q]->c, d_model_layers[q]->N * sizeof(float), cudaMemcpyDeviceToDevice);
        }
      } else {
        // For stateless LSTM: just reset the cache
        //lstm_cache_container_set_start(cache_layers[q][0], model_layers[q]->N);
        cudaMemset(cache_layers[q][0]->h, 0, d_model_layers[q]->N * sizeof(float));
        cudaMemset(cache_layers[q][0]->c, 0, d_model_layers[q]->N * sizeof(float));
      }
      ++q;
    }

    unsigned int check = i % training_points;  // Ensure we don't exceed training points

    trailing = params->mini_batch_size;  // Set the size of the mini-batch

    // Check for boundary conditions to avoid overflow
    if (i + params->mini_batch_size >= training_points) {
      trailing = training_points - i;
    }

    // Loop through each sample in the mini-batch
    q = 0;
    while (q < trailing) {
      e1 = q;
      e2 = q + 1;  // Set indices for caches used in forward propagation

      e3 = i % training_points;  // Get the sample index for the mini-batch

      // Zero the first layer input for one-hot encoding
      tmp_count = 0;
      while (tmp_count < model_layers[0]->Y) {
        first_layer_input[tmp_count] = 0.0;
        ++tmp_count;
      }

      // One-hot encode the input
      first_layer_input[X_train[e3]] = 1.0;

      /* Start forward propagation from the last layer */
      p = layers - 1;
      // get time from time.h
      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      lstm_forward_propagate(d_model_layers[p], first_layer_input, cache_layers[p][e1], cache_layers[p][e2], p == 0);  // Forward propagate for the top layer
      clock_gettime(CLOCK_MONOTONIC, &end);
      double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
      time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
      printf("Time taken for forward propagation: %f\n", time_taken);
      // If there are more layers, propagate downwards
      if (p > 0) {
        --p;
        while (p <= layers - 1) {
          lstm_forward_propagate_cuda(d_model_layers[p], cache_layers[p+1][e2]->probs,  cache_layers[p][e1], cache_layers[p][e2], p == 0);
          --p;
        }
        p = 0;
      }

      // Calculate the loss using cross-entropy
      loss_tmp += cross_entropy(cache_layers[p][e2]->probs, Y_train[e3]);
      ++i;  // Increment mini-batch sample counter
      ++q;
    }

    // Average the loss over the mini-batch
    loss_tmp /= (q + 1);

    // Initialize loss for the first iteration
    if (loss < 0)
      loss = loss_tmp;

    // Apply moving average to smooth the loss
    loss = loss_tmp * params->loss_moving_avg + (1 - params->loss_moving_avg) * loss;

    // Record the lowest loss for tracking the best iteration
    if (n == 0)
      record_keeper = loss;

    if (loss < record_keeper) {
      record_keeper = loss;
      record_iteration = n;
    }

    // For stateful LSTM: Copy the states for the next iteration
    if (stateful) {
      p = 0;
      while (p < layers) {
        //lstm_next_state_copy(stateful_d_next[p], cache_layers[p][e2], model_layers[p]->N, 1);
        cudaMemcpy(stateful_d_next[p]->h, cache_layers[p][e2]->h, d_model_layers[p]->N * sizeof(float), cudaMemcpyDeviceToDevice);
        ++p;
      }
    }

    // Reset gradients and caches for the next backward pass
    p = 0;
    while (p < layers) {
      lstm_zero_the_model(gradient_layers[p]);
      //cudaMemset(gradient_layers[p]->Wf, 0, gradient_layers[p]->N * gradient_layers[p]->S * sizeof(float));
      //cudaMemset(gradient_layers[p]->Wf, 0, gradient_layers[p]->N * gradient_layers[p]->S);
      //cudaMemset(gradient_layers[p]->Wi, 0, gradient_layers[p]->N * gradient_layers[p]->S);
      //cudaMemset(gradient_layers[p]->Wc, 0, gradient_layers[p]->N * gradient_layers[p]->S);
      //cudaMemset(gradient_layers[p]->Wo, 0, gradient_layers[p]->N * gradient_layers[p]->S);
      //cudaMemset(gradient_layers[p]->Wy, 0, gradient_layers[p]->N * gradient_layers[p]->Y);
      //cudaMemset(gradient_layers[p]->bf, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->bi, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->bc, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->bo, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->by, 0, gradient_layers[p]->Y);
      //cudaMemset(gradient_layers[p]->dldhf, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->dldhi, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->dldhc, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->dldho, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->dldc, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->dldh, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->dldXc, 0, (gradient_layers[p]->N+gradient_layers[p]->X));
      //cudaMemset(gradient_layers[p]->dldXo, 0, (gradient_layers[p]->N+gradient_layers[p]->X));
      //cudaMemset(gradient_layers[p]->dldXi, 0, (gradient_layers[p]->N+gradient_layers[p]->X));
      //cudaMemset(gradient_layers[p]->dldXf, 0, (gradient_layers[p]->N+gradient_layers[p]->X));
      //cudaMemset(gradient_layers[p]->Wfm, 0, gradient_layers[p]->N * gradient_layers[p]->S);
      //cudaMemset(gradient_layers[p]->Wim, 0, gradient_layers[p]->N * gradient_layers[p]->S);
      //cudaMemset(gradient_layers[p]->Wcm, 0, gradient_layers[p]->N * gradient_layers[p]->S);
      //cudaMemset(gradient_layers[p]->Wom, 0, gradient_layers[p]->N * gradient_layers[p]->S);
      //cudaMemset(gradient_layers[p]->Wym, 0, gradient_layers[p]->S * gradient_layers[p]->Y);
      //cudaMemset(gradient_layers[p]->bfm, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->bim, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->bcm, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->bom, 0, gradient_layers[p]->N);
      //cudaMemset(gradient_layers[p]->bym, 0, gradient_layers[p]->Y);

      lstm_zero_d_next(d_next_layers[p], model_layers[p]->X, model_layers[p]->N);
      //cudaMemset(d_next_layers[p]->dldh_next, 0, gradient_layers[p]->N);
      //cudaMemset(d_next_layers[p]->dldc_next, 0, gradient_layers[p]->N);
      //cudaMemset(d_next_layers[p]->dldY_pass, 0, gradient_layers[p]->X);
      ++p;
    }

    // Perform backpropagation through time
    while (q > 0) {
      e1 = q;
      e2 = q - 1;

      e3 = (training_points + i - 1) % training_points;

      // Zero the gradients for the next step
      p = 0;
      while (p < layers) {
        lstm_zero_the_model(gradient_layers_entry[p]);
        //cudaMemset(gradient_layers_entry[p]->Wf, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->S);
        //cudaMemset(gradient_layers_entry[p]->Wi, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->S);
        //cudaMemset(gradient_layers_entry[p]->Wc, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->S);
        //cudaMemset(gradient_layers_entry[p]->Wo, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->S);
        //cudaMemset(gradient_layers_entry[p]->Wy, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->Y);
        //cudaMemset(gradient_layers_entry[p]->bf, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->bi, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->bc, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->bo, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->by, 0, gradient_layers_entry[p]->Y);
        //cudaMemset(gradient_layers_entry[p]->dldhf, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->dldhi, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->dldhc, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->dldho, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->dldc, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->dldh, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->dldXc, 0, (gradient_layers_entry[p]->N+gradient_layers_entry[p]->X));
        //cudaMemset(gradient_layers_entry[p]->dldXo, 0, (gradient_layers_entry[p]->N+gradient_layers_entry[p]->X));
        //cudaMemset(gradient_layers_entry[p]->dldXi, 0, (gradient_layers_entry[p]->N+gradient_layers_entry[p]->X));
        //cudaMemset(gradient_layers_entry[p]->dldXf, 0, (gradient_layers_entry[p]->N+gradient_layers_entry[p]->X));
        //cudaMemset(gradient_layers_entry[p]->Wfm, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->S);
        //cudaMemset(gradient_layers_entry[p]->Wim, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->S);
        //cudaMemset(gradient_layers_entry[p]->Wcm, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->S);
        //cudaMemset(gradient_layers_entry[p]->Wom, 0, gradient_layers_entry[p]->N * gradient_layers_entry[p]->S);
        //cudaMemset(gradient_layers_entry[p]->Wym, 0, gradient_layers_entry[p]->S * gradient_layers_entry[p]->Y);
        //cudaMemset(gradient_layers_entry[p]->bfm, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->bim, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->bcm, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->bom, 0, gradient_layers_entry[p]->N);
        //cudaMemset(gradient_layers_entry[p]->bym, 0, gradient_layers_entry[p]->Y);

        ++p;
      }
      // Backpropagate starting from the first layer
      p = 0;
      lstm_backward_propagate(d_model_layers[p], cache_layers[p][e1]->probs, Y_train[e3],  d_next_layers[p], cache_layers[p][e1], gradient_layers_entry[0], d_next_layers[p]);

      // Propagate through the rest of the layers
      if ( p < layers ) {
        ++p;
        while ( p < layers ) {
          lstm_backward_propagate(d_model_layers[p], d_next_layers[p-1]->dldY_pass, -1, d_next_layers[p], cache_layers[p][e1], gradient_layers_entry[p], d_next_layers[p]);
          ++p;
        }
      }

      p = 0; 
      // Sum the gradients across layers
      while ( p < layers ) {
        sum_gradients(gradient_layers[p], gradient_layers_entry[p]);
        ++p;
      }

      i--; q--; // Decrement mini-batch counters for backpropagation
    }

    // Ensure correct sample index after backpropagation
    assert(check == e3);

    // Clip and fit gradients if necessary
    p = 0;
    while ( p < layers ) {

      if ( params->gradient_clip )
        gradients_clip(gradient_layers[p], params->gradient_clip_limit);

      if ( params->gradient_fit )
        gradients_fit(gradient_layers[p], params->gradient_clip_limit);

      ++p;
    }

    // Apply the selected optimization algorithm (Adam or Gradient Descent)
    p = 0;
    switch ( params->optimizer ) {
    case OPTIMIZE_ADAM:
      while ( p < layers ) {
        gradients_adam_optimizer( d_model_layers[p], gradient_layers[p], M_layers[p], R_layers[p], n);
        ++p;
      }
      break;
    case OPTIMIZE_GRADIENT_DESCENT:
      while ( p < layers ) {
        gradients_decend(d_model_layers[p], gradient_layers[p]);
        ++p;
      }
      break;
    default:
      fprintf( stderr,
        "Failed to update gradients, no acceptible optimization algorithm provided.\n\
        lstm_model_parameters_t has a field called 'optimizer'. Set this value to:\n\
        %d: Adam gradients optimizer algorithm\n\
        %d: Gradients descent algorithm.\n",
        OPTIMIZE_ADAM,
        OPTIMIZE_GRADIENT_DESCENT
      );
      exit(1);
      break;
    }

    // Optionally print progress and output sample during training
    if ( print_progress && !( n % print_progress_iterations )) {

      memset(time_buffer, '\0', sizeof time_buffer);
      time(&time_iter);
      strftime(time_buffer, sizeof time_buffer, "%X", localtime(&time_iter));

      printf("%s Iteration: %lu (epoch: %lu), Loss: %lf, record: %lf (iteration: %d), LR: %lf\n",
        time_buffer, n, epoch, loss, record_keeper, record_iteration, params->learning_rate);

      if ( print_progress_sample_output ) {
        printf("=====================================================\n");
        lstm_output_string_layers(d_model_layers, char_index_mapping, X_train[b],
          print_progress_number_of_chars, layers);
        printf("\n=====================================================\n");
      }

      if ( print_progress_to_file ) {
        FILE * fp_progress_output = fopen(print_progress_to_file_name,
          print_progress_to_file_arg);
        if ( fp_progress_output != NULL ) {
          fprintf(fp_progress_output, "%s====== Iteration: %lu, loss: %.5lf ======\n", n==0 ? "" : "\n", n, loss);
          lstm_output_string_layers_to_file(fp_progress_output, d_model_layers, char_index_mapping, X_train[b], print_progress_number_of_chars, layers);
          fclose(fp_progress_output);
        }
      }
			
      // Flushing stdout
      fflush(stdout);
		}

    // Optionally store progress in a file
    if ( store_progress_every_x_iterations && !(n % store_progress_every_x_iterations ) && 0)
      lstm_store_progress(store_progress_file_name, n, loss);

    // Optionally store the network model periodically
    if ( store_network_every && !(n % store_network_every) && 0 ) {
      lstm_store(
        params->store_network_name_raw,
        char_index_mapping,
        model_layers,
        layers);
      lstm_store_net_layers_as_json(model_layers, params->store_network_name_json,
        params->store_char_indx_map_name, char_index_mapping, layers);
    }

    // Check if we have reached the end of the training points
    if (b + params->mini_batch_size >= training_points)
      epoch++;  // Increment the epoch counter if a full pass over the training data is completed

    // Update the index `i` for the next mini-batch
    i = (b + params->mini_batch_size) % training_points;

    // If `i` becomes smaller than the mini-batch size, reset it to 0 for the next iteration
    if (i < params->mini_batch_size) {
      i = 0;
    }

    // Learning rate schedule: decrease learning rate over time if specified
    if ( decrease_lr ) {
      params->learning_rate = initial_learning_rate / ( 1.0 + n / params->learning_rate_decrease );
      //printf("learning rate: %lf\n", model->params->learning_rate);
    }
    // Increment the iteration counter
    ++n;
  }

  // After the training loop, report the final loss value
  *loss_out = loss;

  // Free memory for caches, gradients, and models for each layer
  p = 0;
  while (p < layers) {
    // Free the memory used for the cache for each layer in backpropagation
    lstm_values_next_cache_free(d_next_layers[p]);

    // Free the cache containers for each mini-batch in each layer
    i = 0;
    while (i < params->mini_batch_size) {
      lstm_cache_container_free(cache_layers[p][i]);
      lstm_cache_container_free(cache_layers[p][i]);
      ++i;
    }

    // Free the Adam optimizer-specific parameters if using Adam optimizer
    if (params->optimizer == OPTIMIZE_ADAM) {
      lstm_free_model(M_layers[p]);  // Free the momentum term (M)
      lstm_free_model(R_layers[p]);  // Free the RMSProp term (R)
    }

    // Free the gradient layers
    lstm_free_model(gradient_layers_entry[p]);
    lstm_free_model(gradient_layers[p]);

    ++p;
  }

  // If using stateful LSTM, free the stateful cache
  if (stateful && stateful_d_next != NULL) {
    i = 0;
    while (i < layers) {
      free(stateful_d_next[i]);  // Free memory for each layer's stateful cache
      ++i;
    }
    free(stateful_d_next);  // Free the array holding stateful cache pointers
  }

  // Free the allocated memory for caches and gradient layers
  free(cache_layers);
  free(gradient_layers);
  if (M_layers != NULL)
    free(M_layers);  // Free memory for the Adam optimizer's momentum cache if used
  if (R_layers != NULL)
    free(R_layers);  // Free memory for the Adam optimizer's RMSProp cache if used
#ifdef WINDOWS
  free(first_layer_input);
#endif
}
