#include"loader.h"
#include <stdint.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>

 Network *load_network(const char *path){

   FILE *file = fopen(path ,"rb");
   if (file == NULL){
      fprintf(stderr,"Error opening'%s'\n",path);
      return NULL ;
      }

  /* ── verify header ─────────────────────────────────────────── */

   char magic[4]; // every file has a magic code like PDF files start with %PDF 
   uint32_t version, n_layers;   // version means Right now our format is version 1

   fread(magic, 1, 4, file);
   if(memcmp(magic ,"MINF" , 4) != 0){
      fprintf(stderr,"Error : invalid magic bytes - not a MINF \n");
      fclose(file);
      return NULL ;
   }

   fread(&version , sizeof(uint32_t), 1 ,file);
   fread(&n_layers, sizeof(uint32_t), 1 ,file);

  /* ── allocate network ──────────────────────────────────────── */

   Network *net = malloc(sizeof(Network));
   net->layer = malloc(n_layers * sizeof(Layer));

   for (int i = 0 ; i < n_layers ; i++){

       uint32_t rows , cols , bias_len ;

      fread(&rows , sizeof(uint32_t), 1 , file);
      fread(&cols , sizeof(uint32_t), 1 , file);

      net->layer[i].weights = make_tensor((int)rows ,(int) cols);
      fread(net->layer[i].weights->data , sizeof(float),rows * cols , file );

      fread(&bias_len , sizeof(uint32_t), 1 , file);
      net->layer[i].bias = make_tensor((int)bias_len , 1);
      fread(net->layer[i].bias->data , sizeof(float), bias_len , file );

  /* verification print — remove after confirming correctness */

    printf("Layer %d: W[0][0] = %f\n", i , net->layer[i].weights->data[0]);
    }

    fclose(file);
    return net;
}
