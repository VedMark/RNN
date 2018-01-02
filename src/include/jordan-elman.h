#ifndef ICOMPRESSOR_COMPRESSOR_H
#define ICOMPRESSOR_COMPRESSOR_H


#include <stdbool.h>
#include "algebra.h"

typedef struct RNN_model {
    matrix_type W1;
    matrix_type W2;
    unsigned p;
    unsigned L;
    double E_max;
    double alpha_max;
    double epoch_max;
    char bZero_train : 1;
    char bZero_predict : 1;
    char bAuto_predict : 1;
    char bVerbose : 1;
} RNN_model;


RNN_model *RNN_load(unsigned p, unsigned L, double E_max,
                    double alpha_max, double epoch_max, bool zero_train,
                    bool zero_pred, bool auto_pred, bool verbose);
void RNN_destroy(RNN_model *model);
int RNN_train(RNN_model *model, float *array, int n);
float RNN_predict(RNN_model *model, float *array, int n);

#endif //ICOMPRESSOR_COMPRESSOR_H
