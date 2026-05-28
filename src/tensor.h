#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define linr_rprsnt(cols,i,j)      ( (i) * (cols) + (j) )
#define TENSOR_AT(struct_t,i,j)    (struct_t)->data[(struct_t)->cols * (i) + (j) ]

typedef struct
{
   float  *base;      // the fist address of the arena 
   float  *current;   // the pointer to the cuttent possion  and its type is float bc it points to float that is the data type of tensor's data 
   size_t  capacity ; // the total size that the arena allocate when initialization,... why size_t instead of int ? that bc memory sizes can be veeery large but int just up to 2milions 
} Arena  ;

typedef struct
{
    int rows;
    int cols;
    float *data; // float bc double kills SIMD throughput and doubles memory ,SIMD uses 4byte ,.... and it's float bc i deal with a float wieght for ex...e
} Tensor ;

void ArenaInit    (Arena * a ,size_t arena_size);
float* ArenaAlloc   (Arena * a , float *current,size_t size);
float* ArenaReset   (Arena * a );
void   ArenaFree    (Arena * a );
/*****************************/
Tensor* make_tensor(int rows, int cols);
Tensor* copy_tensor(float *data, int rows, int cols);
void    free_tensor(Tensor *t);


#endif
