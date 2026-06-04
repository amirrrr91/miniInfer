#include"net.h"

int forward_pass ( Network *network ,Tensor *inputs , Arena *a  )
{
    int out =0 ;
    Tensor *fc1 , *fc2;
       fc1 = matmul (inputs ,network->layers[0].weights ,a);
        add_bias (network->layers[0].bias , fc1) ;
        ReLU(fc1);
        fc2 = matmul (fc1,network->layers[1].weights ,a );
        add_bias (network->layers[1].bias , fc2) ;
        softmax(fc2);
        float max = TENSOR_AT(fc2, 0, 0) ;
        for (int j = 0; j < fc2->cols; j++)
        {
            if (TENSOR_AT(fc2, 0, j) > max) 
            {
                max = TENSOR_AT(fc2, 0, j);
                out = j;
            }
        }
    return out ; 
}

  
