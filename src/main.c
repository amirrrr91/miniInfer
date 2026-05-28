#include "loader.h"
#include <stdio.h>

int main(void) {
    Network *network = load_network("weights/model.bin");
    return 0;
}
