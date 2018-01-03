#include <stdlib.h>
#include <time.h>
#include "include/jordan-elman.h"


void init_uniform_dist(gsl_matrix *matrix);


int RNN_load(RNN_model *model,
             unsigned n, unsigned p,
             double E_max,
             double alpha_max,
             double epoch_max,
             bool zero_train, bool zero_pred,
             bool auto_pred, bool verbose)
{
    model->W1 = gsl_matrix_alloc(n + 1, p);
    if(NULL == model->W1) return MEM_ERR;

    model->W2 = gsl_matrix_alloc(p + 1, 1);
    if(NULL == model->W1) return MEM_ERR;

    init_uniform_dist(model->W1);
    init_uniform_dist(model->W2);

    model->y = gsl_vector_alloc(1);
    if(NULL == model->y) return MEM_ERR;

    model->n = n + 1;
    model->p = p + 1;
    model->m = 1;
    model->E_max = E_max;
    model->alpha_max = alpha_max;
    model->epoch_max = epoch_max;
    model->bZero_train = zero_train;
    model->bZero_predict = zero_pred;
    model->bAuto_predict = auto_pred;
    model->bVerbose = verbose;

    return SUCCESS;
}

void RNN_destroy(RNN_model *model) {
    gsl_matrix_free(model->W1);
    gsl_matrix_free(model->W2);
}

int RNN_train(RNN_model *model, float *array, int n) {


    return SUCCESS;
}

int train_step(RNN_model *model, int row) {


    return SUCCESS;
}

float RNN_predict(RNN_model *model, float *array, int n) {


    return SUCCESS;
}

void init_uniform_dist(gsl_matrix *matrix) {
    srand((unsigned int) time(NULL));

    for(size_t i = 0; i < matrix->size1; ++i) {
        for(size_t j = 0; j < matrix->size2; ++j) {
            gsl_matrix_set(matrix, i, j, 1. * rand() / RAND_MAX * 2 - 1);
        }
    }
}
