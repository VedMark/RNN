#ifndef ICOMPRESSOR_COMPRESSOR_H
#define ICOMPRESSOR_COMPRESSOR_H


#include <stdbool.h>
#include <gsl/gsl_blas.h>

#define MEM_ERR (1)
#define ALG_ERR (-1)
#define PAR_ERR (-2)
#define SUCCESS (0)

typedef struct RNN_model {
    gsl_vector *input;
    gsl_matrix *X;
    gsl_vector *x_;
    gsl_vector *x_c;
    gsl_vector *y;
    gsl_vector *y_c;
    gsl_vector *delta_x_;
    gsl_vector *delta_y;
    gsl_vector *e;
    gsl_matrix *W_x;
    gsl_vector *v;
    gsl_matrix *W_x_;
    gsl_vector *W_y;
    unsigned n;
    unsigned p;
    size_t m;
    double E_max;
    size_t epoch_max;
    size_t numPredict;
    char bVerbose : 1;
    char bNullify : 1;
} RNN_model;


int RNN_load(RNN_model *model, unsigned n, size_t m, size_t L, double E_max, size_t epoch_max, size_t pred, bool verbose,
             bool nullify);
void RNN_destroy(RNN_model *model);
int RNN_train(RNN_model *model, gsl_vector *array);
gsl_vector *RNN_predict(RNN_model *model);

#endif //ICOMPRESSOR_COMPRESSOR_H
