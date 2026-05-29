
#include "tensor.h"
#include<stdio.h>
#include<stdlib.h>

void ArenaInit(Arena *a , size_t arena_size)
{
    /* malloc always returns aligned memory */
      a->base = malloc(arena_size);
   if(a->base == NULL)
   {
      fprintf(stderr,"Fatal Error: Could not allocate Arena of size %zu\n", arena_size);
      exit(1);
   }
   a->current = a->base;
   a->capacity = arena_size ;

}

void* ArenaAlloc(Arena *a , size_t size)
{
    // align size up to nearest multiple of 8
    size_t aligned = (size + 7) & ~7;

    if ((a->current - a->base) + size > a->capacity) 
    {
        fprintf(stderr, "Fatal Error: Arena Out of Memory!\n");
        return NULL; // Out of memory
    }
/* cant use the bitwise op to pointers dicrictrly so cant do an alignment to the pointer */
   void *current = a->current;
   a->current +=  aligned  ;
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


// Tensors are arena-managed. Call ArenaFree() to release all memory.
// This function exists only to satisfy any existing callers.
void free_tensor(Tensor *t) { (void)t; }

