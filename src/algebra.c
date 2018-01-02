#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "include/algebra.h"

#define PRINT_MATRIX(m1) {                      \
    printf("NxM = %dx%d\n", (m1)->n, (m1)->m);  \
    for(int i = 0; i < (m1)->n; ++i) {          \
        for(int j = 0; j < (m1)->m; ++j) {      \
            printf("%f ", (m1)->values[i][j]);  \
        }                                       \
        printf("\n");                           \
    }                                           \
    printf("\n");                               \
}                                               \

int new_vector(vector_type *vector, unsigned n) {
    vector->n = n;

    vector->values = malloc(vector->n * sizeof(float));
    if(vector->values == NULL) return MEM_ERR;

    return SUCCESS;
}

void free_vector(vector_type *vector) {
    free(vector->values);
}

int diff_vectors(const vector_type * const v1,
                 const vector_type *const v2,
                 vector_type *const v_res) {

    if(v1->n != v2->n) return ALG_ERR;

    for(int i = 0; i < v1->n; ++i) {
        v_res->values[i] = v1->values[i] - v2->values[i];
    }
    return SUCCESS;
}

int new_matrix(matrix_type *matrix, unsigned n, unsigned m) {
    matrix->n = n;
    matrix->m = m;

    matrix->values = malloc(matrix->n * sizeof(vector_type));
    if(matrix->values == NULL) return MEM_ERR;

    for(int i = 0; i < matrix->n; ++i) {
        if(MEM_ERR == new_vector(&matrix->values[i], m)) return MEM_ERR;
    }
    return SUCCESS;
}

void free_matrix(matrix_type *matrix) {
    for(int i = 0; i < matrix->n; ++i) {
        free_vector(&matrix->values[i]);
    }
    free(matrix->values);
}

void init_uniform_dist(matrix_type *matrix1) {
    srand((unsigned int) time(NULL));

    for(int i = 0; i < matrix1->n; ++i) {
        for(int j = 0; j < matrix1->m; ++j) {
            matrix1->values[i].values[j] = (float)
                    ((1. * rand() / RAND_MAX * 2 - 1 ) * 0.11);
        }
    }
}

float randnorm(float mu, float sigma)
{
    float U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;

    if (call == 1)
    {
        call = !call;
        return (mu + sigma * X2);
    }

    do
    {
        U1 = -1 + ((float) rand () / RAND_MAX) * 2;
        U2 = -1 + ((float) rand () / RAND_MAX) * 2;
        W = powf (U1, 2) + powf(U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrtf ((-2 * logf (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * X1);
}

void init_normal_dist(matrix_type *matrix1) {
    srand((unsigned int) time(NULL));
    float number = 0;

    for(int i = 0; i < matrix1->n; ++i) {
        for(int j = 0; j < matrix1->m; ++j) {
            do {
                number = randnorm(0, 0.1);
            } while(number <= -1 || number >= 1);

            matrix1->values[i].values[j] = number;
        }
    }
}

vector_type *get_col(const matrix_type *const matrix, unsigned col) {
    vector_type *new_v = malloc(sizeof(vector_type));
    if(NULL == new_v) return NULL;

    new_v->values = malloc(matrix->n * sizeof(float));
    if(NULL == new_v->values) return NULL;

    for(unsigned i = 0; i < matrix->n; ++i) {
        new_v->values[i] = matrix->values[i].values[col];
    }

    return new_v;
}

vector_type *get_row(const matrix_type *const matrix, unsigned row) {
    vector_type *new_v = malloc(sizeof(vector_type));
    if(NULL == new_v) return NULL;

    new_v->values = malloc(matrix->values[row].n * sizeof(float));
    if(NULL == new_v->values) return NULL;

    for(unsigned j = 0; j < matrix->values[row].n; ++j) {
        new_v->values[j] = matrix->values[row].values[j];
    }

    return new_v;
}

int mult_matrixes(const matrix_type *const m1,
                  const matrix_type *const m2,
                  matrix_type *m_res) {
    if(m1->m != m2->n) return ALG_ERR;
    float sum = 0;

    for(int i = 0; i < m1->n; ++i) {
        for(int j = 0; j < m2->m; ++j) {
            sum = 0;
            for(int r = 0; r < m1->m; ++r) {
                sum += m1->values[i].values[r] * m2->values[r].values[j];
            }
            m_res->values[i].values[j] = sum;
        }
    }
    return SUCCESS;
}

void normalize(matrix_type *m) {
    float sum = 0;

    for(int j = 0; j < m->m; ++j) {
        sum = 0;
        for(int i = 0; i < m->n; ++i) {
            sum += m->values[i].values[j] * m->values[i].values[j];
        }
        sum = sqrtf(sum);
        for(int i = 0; i < m->n; ++i) {
            m->values[i].values[j] /= sum;
        }
    }
}

void transpose(const matrix_type *const m, matrix_type *m_res) {
    m_res->n = m->m;
    m_res->m = m->n;

    for(int i = 0; i < m_res->n; ++i) {
        for(int j = 0; j < m_res->m; ++j) {
            m_res->values[i].values[j] = m->values[j].values[i];
        }
    }
}
