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
    
    int out ;
    char path[64];
    Arena permanent, temp;
ArenaInit(&permanent, ARENA_SIZE);
ArenaInit(&temp,ARENA_SIZE);  // small, just for inference tensors

Network *network = load_network("weights/model.bin", &permanent);

// load all samples once before the timed loop
Tensor *samples[100];
int labels[100];
for (int j = 0; j < 100; j++) {
    sprintf(path, "weights/sample_%d.bin", j);
    samples[j] = load_sample(path, &labels[j], &permanent);
}

// now time pure compute
for (int i = 0; i < 1000; i++)
    for (int j = 0; j < 100; j++) {
        ArenaReset(&temp);
        forward_pass(network, samples[j], &temp);
    }
    ArenaReset(&permanent);
    ArenaReset(&temp);
    return 0;
}
