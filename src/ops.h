/**************************** بسم الله الرحمن الرحيم ********************/

#ifndef OPS_H
#define OPS_H
#define _USE_MATH_DEFINES
#include "loader.h"
#include<math.h>
/************************* */
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//#define M_E		2.7182818284590452354


Tensor* matmul(Tensor *inputs , Tensor *weights , Tensor *res , Arena *a);
Tensor* add_bias(Tensor *bias , Tensor *wa);
void connecionEQ(Layer *FC );
void ReLU();
void softmax();
void argma();


#endif