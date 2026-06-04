
#ifndef LOADER_H
#define LOADER_H

#include"tensor.h"
#include <stdint.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>


typedef struct{
   Tensor *weights;
   Tensor *bias;
}Layer; 

typedef struct {
   Layer *layer;
   int   n_layers;
}Network;

Network* load_network (const char *path , Arena *a );


#endif 
