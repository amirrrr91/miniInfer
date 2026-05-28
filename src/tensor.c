
#include "tensor.h"
#include<stdio.h>
#include<stdlib.h>

void ArenaInit(Arena *a , size_t arena_size)
{
   a->base = malloc(arena_size);
   a->current = a->base;
   a->capacity = arena_size ;

}

float* ArenaAlloc(Arena *a , float *current , size_t size)
{
   a->current += size ;
  return current ;
}

Tensor* make_tensor(Arena *a , int rows, int cols)  
{
   Tensor* ptensor = ArenaAlloc(a , a->current , sizeof(Tensor));
   
    ptensor->cols = cols;
    ptensor->rows = rows;
    ptensor->data = malloc ( rows * cols * sizeof(float));  
    return ptensor;              
}


Tensor* copy_tensor(float* data, int rows, int cols) {
    Tensor* ptensor = make_tensor(rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
           TENSOR_AT(ptensor,i,j)= data[linr_rprsnt(cols,i,j)];
        }
    }
    return ptensor;    
}


void free_tensor(Tensor *t){
    free(t->data);
    free(t);
}

