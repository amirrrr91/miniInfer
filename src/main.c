#include "loader.h"
#include <stdio.h>
/*
vboxuser@ubuntu-24:~/miniInfer$  ls -la weights/model.bin
-rw-rw-r-- 1 vboxuser vboxuser 407116 Mar 30 20:47 weights/model.bin
407116 / 4 = 101779 floats
*/
#define ARENA_SIZE (110000 * sizeof(float))
 Tensor *load_sample(const char *path ,int *out, Arena *a);
int main(void) {
    Arena a;
    int out ;
    ArenaInit(&a ,ARENA_SIZE ) ;
    Network *network = load_network("weights/model.bin", &a);
    Tensor  *inputs  = load_sample ("weights/sample.bin",&out,&a);

    int prediction = forward_pass(network, inputs, &a);
    printf("True label:  %d\n", out);
    printf("Predicted:   %d\n", prediction);
    printf("%s\n", prediction == out ? "CORRECT" : "WRONG");
    
    return 0;
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
