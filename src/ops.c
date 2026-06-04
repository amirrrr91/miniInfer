/*[i*inputs->cols + k] = [i][k]**/
#include"ops.h"

void matmul(Tensor *inputs , Tensor *weights , Tensor *res , Arena *a)
{
    /*input:    (1 × 784)
fc1 weights:    (128 × 784)"by default by pytorch" → transposed for multiply: (784 × 128)
fc1 output:     (1 × 128)
relu output:    (1 × 128)    → same shape, just negatives zeroed
fc2 weights:    (10 × 128)"by default by pytorch"  → transposed: (128 × 10)
fc2 output:     (1 × 10)     → your 10 scores
softmax output: (1 × 10)     → same shape, now probabilities
argmax:         1 digit     
 */
// after transposing we بنضرب صف في صف عادي فا مستتغربش 
   /* res     = make_tensor(a , 1 , weights->rows ); 
    for (int i = 0; i < inputs->rows; i++)       // i → inputs rows  1
    {
       for (int j = 0; j < weights->rows; j++)    // j → weights cols  = 128 
       {
           float sum = 0.0f;
           for (int k = 0; k < inputs->cols; k++) // k → inputs cols = weights rows but because the transposing it (k → weights cols  = 784) 
           {
              sum += inputs->data[i*inputs->cols + k]  * weights->data[j*weights->cols + k] ;  // b is already transposed in memory
           }
           res->data[i*res->cols + j]  = sum;
       }
    }*/
   /** // this is wrong because copy dont transpose just in its implementation  this is data[linr_rprsnt(cols,i,j)] 
    inputs  = copy_tensor(a , inputs->data , 1 , weights->cols);                 //[1][784]     // u ask it subuse to be weights->rows  buuuutttt  the "PyTorch stores nn.Linear(784, 128) as a (128×784) matrix — output rows, input cols. This is called row-major weight storage. When PyTorch does the actual multiply internally it transposes on the fly."
    weights = copy_tensor(a , weights->data , weights->cols , weights->rows);    //[128][784] for first layer fc1 "by default by pytorch" so transpose it 
    res     = make_tensor(a , 1 , weights->rows );                               //[1][128]
    float r = 0 ;
        for (int i=0; i < inputs->rows; i++)    //1 
    {
        for (int k=0; k < weights->rows; k++)    //784
        {
            r = inputs->data[i*inputs->cols + k];
            for (int j=0; j < weights->cols; j++)  //128
            {
                res->data[i* res->cols + j] += r * weights->data[k* weights->cols + j];
            }
        }
    }*/
res     = make_tensor(a , 1 , weights->rows );                               //[1][128]
    for (int i = 0; i < inputs->rows; i++)        // 1
    {
        for (int j = 0; j < weights->rows; j++)   // 128 ← swapped
        {
            for (int k = 0; k < weights->cols; k++) // shared size  ← swapped   784
            {
                TENSOR_AT(res, i, j) += TENSOR_AT(inputs, i, k) * TENSOR_AT(weights, j, k);
                //                                                             ↑
                //                                       j,k because B is transposed
            }
        }
    }
}

void add_bias(Tensor *bias , Tensor *fc_op)    // fc_op = full connected network output
{
    for (int i = 0; i < fc_op->rows; i++)   //1
    {
        for (int j = 0; j < fc_op->cols; j++) //128
        {
            TENSOR_AT(fc_op, i, j) += bias->data[j];

        }
    }    
}
void ReLU(Tensor *fc_op)
{
    for (int i = 0; i < fc_op->rows; i++)   //1
    {
        for (int j = 0; j < fc_op->cols; j++) //128
        {   
            TENSOR_AT(fc_op, i, j) = MAX(0.0f,TENSOR_AT(fc_op, i, j));
        }
    }

}
void softmax(Tensor *output)
{
    for (int i = 0; i < output->rows; i++)   //1
    {
        float max = TENSOR_AT(output, i, 0) ;
        for (int j = 0; j < output->cols; j++)
            {
                max = MAX(TENSOR_AT(output, i, j),max);
            }

        float sum =0.0f;
        for (int j = 0; j < output->cols; j++)
            {
                sum += expf(TENSOR_AT(output, i, j) - max);
            }

        for (int j = 0; j < output->cols; j++) 
        {   
            TENSOR_AT(output, i, j) = expf(TENSOR_AT(output, i, j)- max)/sum;
        }
    }
}
void argma();