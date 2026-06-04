#include "loader.h"
#include"net.h"
#include"ops.h"
#include"tensor.h"
#include <stdio.h>
/*
vboxuser@ubuntu-24:~/miniInfer$  ls -la weights/model.bin
-rw-rw-r-- 1 vboxuser vboxuser 407116 Mar 30 20:47 weights/model.bin
407116 / 4 = 101779 floats
*/
#define ARENA_SIZE (10 * 1024 * 1024)  // 10 MB

int main(void) {
    Arena a;
    int out ;
    char path[64];
    ArenaInit(&a ,ARENA_SIZE ) ;
    Network *network = load_network("weights/model.bin", &a);
    for (int i =0 ; i< 10 ;i++)
    {

    sprintf(path, "weights/sample_%d.bin", i);
    Tensor  *inputs  = load_sample (path,&out,&a);

    int prediction = forward_pass(network, inputs, &a);
    printf("True label:  %d\n", out);
    printf("Predicted:   %d\n", prediction);
    printf("%s\n", prediction == out ? "CORRECT" : "WRONG");
    }
    ArenaReset(&a);   
    return 0;
}
