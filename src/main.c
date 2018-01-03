#include <stdio.h>
#include "include/jordan-elman.h"

int main(int argc, char **argv) {
    if(argc < 1){
        printf("%s: usage: %s ", argv[0], argv[0]);
        return 1;
    }

    RNN_model *rnn_model = NULL;
    int ret_val = 0;

    rnn_model = malloc(sizeof(RNN_model));

    ret_val = RNN_load
            (
                    rnn_model,
                    4, 3,
                    0.1, 0.1,
                    1000000,
                    false, false, false, false
            );
    if(MEM_ERR == ret_val) {
        fprintf(stderr, "internal memory error\n");
        exit(1);
    }

    RNN_destroy(rnn_model);

    return 0;
}