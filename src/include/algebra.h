#ifndef RNN_ALGEBRA_H
#define RNN_ALGEBRA_H


#define MEM_ERR (1)
#define ALG_ERR (-1)
#define SUCCESS (0)

typedef struct vector_type {
    float *values;
    unsigned n;
} vector_type;

typedef struct matrix_type {
    vector_type *values;
    unsigned n;
    unsigned m;
} matrix_type;

int new_vector(vector_type *vector, unsigned n);
void free_vector(vector_type *vector);
int diff_vectors(const vector_type *v1, const vector_type *v2, vector_type *v_res);

int new_matrix(matrix_type *matrix, unsigned n, unsigned m);
void free_matrix(matrix_type *matrix);
void init_uniform_dist(matrix_type *matrix1);
vector_type *get_col(const matrix_type *matrix, unsigned col);
vector_type *get_row(const matrix_type *matrix, unsigned row);
int mult_matrixes(const matrix_type *m1, const matrix_type *m2, matrix_type *m_res);
void normalize(register matrix_type *m);
void transpose(const matrix_type *m, matrix_type *m_res);

#endif //RNN_ALGEBRA_H
