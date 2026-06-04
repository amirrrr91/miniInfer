/**************************** بسم الله الرحمن الرحيم ********************/

#ifndef OPS_H
#define OPS_H
#define _USE_MATH_DEFINES
#include "loader.h"
#include<math.h>
/************************* */
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void matmul(Tensor *inputs , Tensor *weights , Tensor *res , Arena *a);
void add_bias(Tensor *bias , Tensor *fc_op) ;
void ReLU(Tensor *fc_op) ;
void softmax(Tensor *output);



#endif