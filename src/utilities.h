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
#ifndef LSTM_UTILITIES_H
#define LSTM_UTILITIES_H

/*! \file utilities.h
    \brief Some utility functions used in the LSTM program
    
    Here are some functions that help produce the LSTM network.
*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

// used on contigous vectors
//		A = A + B		A,		B,    l
void 	vectors_add(float*, float*, int);
void 	vectors_substract(float*, float*, int);
void 	vectors_add_scalar_multiply(float*, float*, int, float);
void 	vectors_scalar_multiply(float*, float, int);
void 	vectors_substract_scalar_multiply(float*, float*, int, float);
void 	vectors_add_scalar(float*, float, int );
void 	vectors_div(float*, float*, int);
void 	vector_sqrt(float*, int);
void 	vector_store_json(float*, int, FILE *);
void 	vector_store_as_matrix_json(float*, int, int, FILE *);
//		A = A + B		A,		B,    R, C
void 	matrix_add(float**, float**, int, int);
void 	matrix_substract(float**, float**, int, int);
//		A = A*b		A,		b,    R, C
void 	matrix_scalar_multiply(float**, float, int, int);

//		A = A * B		A,		B,    l
void 	vectors_multiply(float*, float*, int);
//		A = A * b		A,		b,    l
void 	vectors_mutliply_scalar(float*, float, int);
//		A = random( (R, C) ) / sqrt(R / 2), &A, R, C
int 	init_random_matrix(float***, int, int);
//		A = 0.0s, &A, R, C
int 	init_zero_matrix(float***, int, int);
int 	free_matrix(float**, int);
//						 V to be set, Length
int 	init_zero_vector(float**, int);
int 	free_vector(float**);
//		A = B       A,		B,		length
void 	copy_vector(float*, float*, int);
float* 	get_zero_vector(int); 
float** 	get_zero_matrix(int, int);
float** 	get_random_matrix(int, int);
float* 	get_random_vector(int,int);

void 	matrix_set_to_zero(float**, int, int);
void 	vector_set_to_zero(float*, int);

float sample_normal(void);
float randn(float, float);

float one_norm(float*, int);

void matrix_clip(float**, float, int, int);
int vectors_fit(float*, float, int);
int vectors_clip(float*, float, int);

// I/O
void 	vector_print_min_max(char *, float *, int);
void 	vector_read(float *, int, FILE *);
void 	vector_store(float *, int, FILE *);
void 	matrix_store(float **, int, int, FILE *);  
void 	matrix_read(float **, int, int, FILE *);
void 	vector_read_ascii(float *, int, FILE *);
void 	vector_store_ascii(float *, int, FILE *);

// Memory
void*   e_calloc(size_t count, size_t size);
size_t  e_alloc_total();
#endif

