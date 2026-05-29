#include "loader.h"
#include <stdio.h>
/*
vboxuser@ubuntu-24:~/miniInfer$  ls -la weights/model.bin
-rw-rw-r-- 1 vboxuser vboxuser 407116 Mar 30 20:47 weights/model.bin
407116 / 4 = 101779 floats
*/
#define ARENA_SIZE (110000 * sizeof(float))
int main(void) {
    Arena a;
    ArenaInit(&a ,ARENA_SIZE ) ;
    Network *network = load_network("weights/model.bin", &a);
    return 0;
}
