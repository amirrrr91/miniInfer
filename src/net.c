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

Tensor *load_sample(const char *path ,int *out, Arena *a){

   FILE *file = fopen(path ,"rb");
   if (file == NULL){
      fprintf(stderr,"Error opening'%s'\n",path);
      return NULL ;
      }

  /* ── verify header ─────────────────────────────────────────── */
   uint32_t label ;  
   Tensor *pixels = make_tensor(a , 1 , 784);

   fread(&label, sizeof(uint32_t), 1, file);
   fread(pixels->data ,784* sizeof(float), 1 ,file);

   *out = (int)label;   // add this line
   fclose(file);

   return pixels ;
 }


  
