#include <stdio.h>
#include <string.h>
#include "include/jordan-elman.h"

void print_help() {
    printf("Usage: RNN <input> <p> <m> <E_max> <epoch_max> <pred> <verbose> <nullify>\n"
                   "   or:  RNN --help\n\n"
                   "  input\t\ta file containing sequence\n"
                   "  p\t\tsize of window\n"
                   "  m\t\tnumber of neurons in a hidden layer\n"
                   "  E_max\t\tmaximum standard error. Training ends when\n"
                   "\t\t  standard error on an epoch becomes less than E_max\n"
                   "\t\t  (0 < E_max <= 0.1)\n"
                   "  epoch_max\tmaximum epoches count for learning the network\n"
                   "  pred\tnumber of values to predict\n"
                   "  verbose\tif true, print information about each step during learning\n"
                   "  nullify\tif true, context neurons are nullified after each epoch\n");
}

int main(int argc, char **argv) {
    const char *usage = "Usage: %s <input> <p> <m> <E_max> <epoch_max> <pred> <verbose> <nullify>\n"
            "  or:  %s --help\n";
    if(!(argc == 2 || argc == 9)){
        printf(usage, argv[0], argv[0]);
        return 1;
    }

    if (argc == 2) {
        if (!strcmp(argv[1], "--help")) {
            print_help();
            return 0;
        } else {
            fprintf(stderr, usage, argv[0], argv[0]);
            return 1;
        }
    }

    RNN_model *rnn_model = NULL;
    gsl_vector *input_vector = NULL;
    FILE *input_file = NULL;
    gsl_vector *predictions = NULL;
    int ret_val = 0;
    size_t szMatrix = 0;
    unsigned int n = 0;
    unsigned int m = 0;
    double E_max = 0;
    size_t epoch_max = 0;
    size_t pred = 0;
    bool verbose;
    bool nullify;

    if(NULL == (input_file = fopen(argv[1], "r"))) {
        fprintf(stderr, "could not open file %s!\n", argv[1]);
        return 1;
    }

    rnn_model = malloc(sizeof(RNN_model));

    ret_val = fscanf(input_file, "%zu\n", &szMatrix);
    if(EOF == ret_val) {
        fprintf(stderr, "error reading from file\n");
        return 1;
    }

    input_vector = gsl_vector_alloc(szMatrix);
    gsl_vector_fscanf(input_file, input_vector);

    n = strtoul(argv[2], NULL, 10);
    m = strtoul(argv[3], NULL, 10);
    E_max = strtod(argv[4], NULL);
    epoch_max = strtoull(argv[5], NULL, 10);
    pred = strtoul(argv[6], NULL, 10);
    verbose = !strcmp(argv[7], "true") ? true : false;
    nullify = !strcmp(argv[8], "true") ? true : false;

    if(n >= szMatrix || E_max < 0 || E_max > 0.1 || epoch_max <= 0) {
        fprintf(stderr, "parameter(s) has wrong value!\n");
        return 1;
    }

    ret_val = RNN_load(rnn_model, n, m, szMatrix - n, E_max, epoch_max, pred, verbose, nullify);
    if(MEM_ERR == ret_val) {
        fprintf(stderr, "internal memory error!\n");
        exit(1);
    }

    ret_val = RNN_train(rnn_model, input_vector);
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

    if(NULL == (predictions = RNN_predict(rnn_model))) {
        fprintf(stderr, "internal memory error!\n");
    }

    for(size_t i = 0; i < predictions->size; ++i) {
        printf("%lf ", gsl_vector_get(predictions, i));
    }
    printf("\n");

    RNN_destroy(rnn_model);
    gsl_vector_free(input_vector);
    gsl_vector_free(predictions);
    fclose(input_file);
    free(rnn_model);

    return 0;
}