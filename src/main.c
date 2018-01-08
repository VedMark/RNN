#include <stdio.h>
#include "include/jordan-elman.h"

int main(int argc, char **argv) {
    if(argc < 1){
        printf("%s: usage: %s ", argv[0], argv[0]);
        return 1;
    }

    float arr[] = {.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0};

    RNN_model *rnn_model = NULL;
    int ret_val = 0;

    rnn_model = malloc(sizeof(RNN_model));

    ret_val = RNN_load
            (
                    rnn_model,
                    6, 3,
                    0.1, 0.1,
                    1000000,
                    false, false, false, false
            );
    if(MEM_ERR == ret_val) {
        fprintf(stderr, "internal memory error!\n");
        exit(1);
    }

    ret_val = RNN_train(rnn_model, arr, 10);
    if(MEM_ERR == ret_val) {
        fprintf(stderr, "internal memory error!\n");
        exit(1);
    }
    if(ALG_ERR == ret_val) {
        fprintf(stderr, "linear algebra error!\n");
        exit(1);
    }
    if(PAR_ERR == ret_val) {
        fprintf(stderr, "parameter(s) has wrong value!\n");
        exit(1);
    }

    RNN_destroy(rnn_model);

    return 0;
}