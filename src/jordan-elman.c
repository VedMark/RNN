#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include "include/jordan-elman.h"

#define PRINT_MATRIX(M) {                               \
    printf("MxN: %lu %lu\n", (M)->size1, (M)->size2);     \
    for(int i = 0; i < (M)->size1; ++i) {               \
        for(int j = 0; j < (M)->size2; ++j) {           \
            printf("%lf ", gsl_matrix_get((M), i, j));    \
        }                                               \
    printf("\n");                                       \
    }                                                   \
}

#define PRINT_VECTOR(V) {                           \
    printf("M: %lu\n", (V)->size);                  \
    for(int i = 0; i < (V)->size; ++i) {            \
        printf("%lf ", gsl_vector_get((V), i));     \
    }                                               \
    printf("\n");                                   \
}

double adaptive_step(RNN_model *model);
void normalize_m(gsl_matrix *matrix);
void normalize_v(gsl_vector *vector);
void gamma_last_layer(const RNN_model *model, size_t row);
void gamma_hidden_layer(const RNN_model *model);
double E(const gsl_matrix const *Y, const gsl_vector const *e);
double F(double S);
double F_(double S);
double S(const RNN_model const *model, size_t row, size_t i);
double model_out(const RNN_model const *model);

void back_propagation(const RNN_model const *model, size_t row);
void forward_propagation(const RNN_model const *model, gsl_vector *p, size_t row);
void init_sample(RNN_model *model, float *array);
void init_uniform_dist_m(gsl_matrix *matrix);
void init_uniform_dist_v(gsl_vector *vector);
int train_epoch(RNN_model *model, gsl_matrix *Y, gsl_vector *p, double *alpha, double *error);
bool verify_sample(RNN_model *model, int szArray);
void update_values(RNN_model *model, gsl_vector *p_1, size_t row, double alpha);


double adaptive_step(RNN_model *model) {
    double alpha = 0;
    double y_i = 0;
    double sum = 1;

    for(size_t i = 0; i < model->p; ++i) {
        y_i = gsl_vector_get(model->y, i);

        sum += pow(y_i, 2);
    }

    alpha = 1 / sum;

    return alpha > 1 ? 1 : alpha == 0 ? .00001 : alpha;
}

void normalize_m(gsl_matrix *matrix) {
    float sum = 0;

    for(size_t j = 0; j < matrix->size2; ++j) {
        sum = 0;
        for(size_t i = 0; i < matrix->size1; ++i) {
            sum += pow(gsl_matrix_get(matrix, i, j), 2);
        }
        sum = sqrtf(sum);
        if(0 == sum) continue;

        for(size_t i = 0; i < matrix->size1; ++i) {
            gsl_matrix_set(matrix, i, j, gsl_matrix_get(matrix, i, j) / sum);
        }
    }
}

void normalize_v(gsl_vector *vector) {
    float sum = 0;

    for(size_t j = 0; j < vector->size; ++j) {
        sum += pow(gsl_vector_get(vector, j), 2);
    }

    if(0 == sum) return;

    sum = 1 / sqrtf(sum);

    gsl_vector_scale(vector, sum);
}


void gamma_last_layer(const RNN_model *model, size_t row) {
    double diff = 0;

    for(size_t i = 0; i < model->p; ++i) {
        double y = gsl_vector_get(model->y, i);
        double e = gsl_vector_get(model->e, row);
        diff = e - y;
        gsl_vector_set(model->gamma_y, i, diff * F_(y));
    }
}

void gamma_hidden_layer(const RNN_model *model) {
    double sum = 0;
    double gamma_j = 0;
    double v_i = 0;
    double w_ij = 0;

    for(size_t i = 0; i < model->m; ++i) {
        sum = 0;
        w_ij = gsl_vector_get(model->v, i);
        v_i = gsl_vector_get(model->x_, i);

        for(size_t j = 0; j < model->p; ++j) {
            gamma_j = gsl_vector_get(model->gamma_y, j);
            sum += gamma_j * w_ij * F_(v_i) ;
        }
        gsl_vector_set(model->gamma_x_, i, sum);
    }
}

double E(const gsl_matrix const *Y, const gsl_vector const *e) {
    double error = 0;
    double y_j = 0;
    double e_j = 0;

    for(size_t i = 0; i < Y->size1; ++i) {
        for(size_t j = 0; j < Y->size2; ++j) {
            y_j = gsl_matrix_get(Y, i, j);
            e_j = gsl_vector_get(e, j);

            error += pow(y_j - e_j, 2);
        }
    }

    return error / 2;
}

inline double F(double S) {
    return tanh(S); //log(fabs(cosh(S)));
}

inline double F_(double S) {
   return 1 - pow(S, 2); //tanh(S);
}

double S(const RNN_model const *model, size_t row, size_t i) {
    double sum = 0;

    for(size_t j = 0; j < model->n; ++j) {
        double x = gsl_matrix_get(model->X, row, j);
        double w_x = gsl_matrix_get(model->W_x, j, i);
        sum += x * w_x;
    }

    for(size_t l = 0; l < model->m; ++l) {
        sum += gsl_vector_get(model->x_, l) * gsl_vector_get(model->v, l);
    }

    for(size_t k = 0; k < model->p; ++k) {
        sum += gsl_vector_get(model->y, k) * gsl_matrix_get(model->W_y, k, i);
    }

    return sum;
}

double model_out(const RNN_model const *model) {
    double v_ij = 0;
    double p_i = 0;
    double sum = 0;

    for(size_t i = 0; i < model->m; ++i) {
        v_ij = gsl_vector_get(model->v, i);
        p_i = gsl_vector_get(model->x_, i);
        sum += v_ij * p_i;
    }

    return F(sum);
}

int RNN_load(RNN_model *model,
             unsigned n, unsigned m,
             double E_max,
             double alpha_max,
             double epoch_max,
             bool zero_train, bool zero_pred,
             bool auto_pred, bool verbose)
{
    model->X = gsl_matrix_alloc(m, n);
    if(NULL == model->X) return MEM_ERR;

    model->x_ = gsl_vector_calloc(m);
    if(NULL == model->x_) return MEM_ERR;

    model->y = gsl_vector_calloc(1);
    if(NULL == model->y) return MEM_ERR;

    model->gamma_x_ = gsl_vector_alloc(m);
    if(NULL == model->gamma_x_) return MEM_ERR;

    model->gamma_y = gsl_vector_alloc(1);
    if(NULL == model->gamma_y) return MEM_ERR;

    model->e = gsl_vector_alloc(m);
    if(NULL == model->e) return MEM_ERR;

    model->W_x = gsl_matrix_alloc(n, m);
    if(NULL == model->W_x) return MEM_ERR;

    model->v = gsl_vector_alloc(m);
    if(NULL == model->v) return MEM_ERR;

    model->W_y = gsl_matrix_alloc(1, m);
    if(NULL == model->W_x) return MEM_ERR;

    init_uniform_dist_m(model->W_x);
    init_uniform_dist_v(model->v);
    init_uniform_dist_m(model->W_y);

    model->n = n;
    model->m = m;
    model->p = 1;
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
    gsl_matrix_free(model->X);
    gsl_vector_free(model->x_);
    gsl_vector_free(model->y);
    gsl_vector_free(model->gamma_x_);
    gsl_vector_free(model->gamma_y);
    gsl_vector_free(model->e);
    gsl_matrix_free(model->W_x);
    gsl_vector_free(model->v);
    gsl_matrix_free(model->W_y);
}

int RNN_train(RNN_model *model, float *array, int n) {
    gsl_matrix *Y = NULL;
    gsl_vector *p = NULL;

    int ret_val = 0;
    double alpha = 0;
    double error = 0;
    int epoch = 1;

    p = gsl_vector_alloc(model->m);
    Y = gsl_matrix_alloc(model->m, model->p);
    if(NULL == Y) return MEM_ERR;

    if(!verify_sample(model, n)) return PAR_ERR;
    init_sample(model, array);

    printf("-- training the model\n");

    normalize_m(model->W_x);
    normalize_v(model->v);
    normalize_m(model->W_y);

    do {
        ret_val = train_epoch(model, Y, p, &alpha, &error);
        if(MEM_ERR == ret_val) return MEM_ERR;
        if(ALG_ERR == ret_val) return ALG_ERR;

        printf("epoch: %d; alpha: %.6lf; error: %.6lf\n",
               epoch++, alpha, error);

    } while(error > model->E_max);

    gsl_vector_free(p);
    gsl_matrix_free(Y);

    return SUCCESS;
}

void init_sample(RNN_model *model, float *array) {
    for(size_t i = 0; i < model->m; ++i) {
        for(size_t j = 0; j < model->n; ++j) {
            gsl_matrix_set(model->X, i, j, array[i + j]);
        }
        gsl_vector_set(model->e, i, array[i + model->n]);
    }
}

bool verify_sample(RNN_model *model, int szArray) {
    return szArray >= model->n + model->m;
}

int train_epoch(RNN_model *model, gsl_matrix *Y, gsl_vector *p, double *alpha, double *error) {
    for(size_t i = 0; i < model->m; ++i) {
        forward_propagation(model, p, i);
        back_propagation(model, i);

        gsl_matrix_set_row(Y, i, model->y);

        *alpha = adaptive_step(model);

        update_values(model, p, i, *alpha);

        normalize_m(model->W_x);
        normalize_v(model->v);
        normalize_m(model->W_y);
    }

    *error = E(Y, model->e);

    return SUCCESS;
}

void forward_propagation(const RNN_model const *model, gsl_vector *p, size_t row) {
    double p_i = 0;

    for(size_t i = 0; i < model->m; ++i) {
        p_i = F(S(model, row, i));
        gsl_vector_set(p, i, p_i);
    }

    gsl_vector_swap(model->x_, p);

    for(size_t i = 0; i < model->p; ++i) {
        gsl_vector_set(model->y, i, model_out(model));
    }
}

void back_propagation(const RNN_model const *model, size_t row) {
    gamma_last_layer(model, row);
    gamma_hidden_layer(model);
}

void update_values(RNN_model *model, gsl_vector *p_1, size_t row, double alpha) {
    double y = 0;
    double p_i = 0;
    double x_j = 0;
    double gamma_i = 0;
    double gamma = 0;

    y = gsl_vector_get(model->y, 0);
    gamma = gsl_vector_get(model->gamma_y, 0);


    for(size_t i = 0; i < model->m; ++i) {
        p_i = gsl_vector_get(model->x_, i);
        gamma_i = gsl_vector_get(model->gamma_x_, i);

        for(size_t j = 0; j < model->n; ++j) {
            x_j = gsl_matrix_get(model->X, row, j);

            gsl_matrix_set(model->W_x, j, i,
                           gsl_matrix_get(model->W_x, j, i)
                           + x_j * gamma_i * alpha);
        }

        gsl_vector_set(model->v, i,
                       gsl_vector_get(model->v, i) + p_i * gamma * alpha);

        for(size_t l = 0; l < model->p; ++l) {
            gsl_matrix_set(model->W_y, l, i,
                           gsl_matrix_get(model->W_y, l, i)
                           + y * gamma_i * alpha);
        }
    }
}

float RNN_predict(RNN_model *model, float *array, size_t n) {

    return SUCCESS;
}

void init_uniform_dist_m(gsl_matrix *matrix) {
    srandom((unsigned int) time(NULL));

    for(size_t i = 0; i < matrix->size1; ++i) {
        for(size_t j = 0; j < matrix->size2; ++j) {
            gsl_matrix_set(matrix, i, j, 1. * random() / RAND_MAX * 2 - 1);
        }
    }
}

void init_uniform_dist_v(gsl_vector *vector) {
    srandom((unsigned int) time(NULL));

    for(size_t i = 0; i < vector->size; ++i) {
            gsl_vector_set(vector, i, 1. * random() / RAND_MAX * 2 - 1);
    }
}