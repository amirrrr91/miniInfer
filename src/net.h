#ifndef NET_H
#define NET_H
#include "ops.h"
/******************************************** */

int forward_pass ( Network *network ,Tensor *inputs , Arena *a  );
Tensor *load_sample(const char *path ,int *out, Arena *a);
#endif