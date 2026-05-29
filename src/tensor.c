
#include "tensor.h"
#include<stdio.h>
#include<stdlib.h>

void ArenaInit(Arena *a , size_t arena_size)
{
   a->base = malloc(arena_size);
   a->current = a->base;
   a->capacity = arena_size ;

}

void* ArenaAlloc(Arena *a , size_t size)
{
    if ((a->current - a->base) + size > a->capacity) {
        return NULL; // Out of memory
    }
   void *current = a->current;
   a->current += size ;
  return current;
}

void ArenaReset (Arena *a )
{
    a->current = a->base ;
}

void   ArenaFree (Arena * a )
{
    free(a->base);
}

Tensor* make_tensor(Arena *a , int rows, int cols)  
{
   Tensor* ptensor = ArenaAlloc(a , sizeof(Tensor));
   
    ptensor->cols = cols;
    ptensor->rows = rows;
    ptensor->data = ArenaAlloc(a , rows * cols * sizeof(float));  
    return ptensor;              
}


Tensor* copy_tensor(Arena *a ,float* data, int rows, int cols) 
{
    Tensor* ptensor = make_tensor(a, rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
           TENSOR_AT(ptensor,i,j)= data[linr_rprsnt(cols,i,j)];
        }
    }
    return ptensor;    
}


void free_tensor(Tensor *t)
{
   (void)t;
}

