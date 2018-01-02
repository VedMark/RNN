#include "include/jordan-elman.h"

RNN_model *RNN_load(unsigned p, unsigned L, double E_max,
                    double alpha_max, double epoch_max, bool zero_train,
                    bool zero_pred, bool auto_pred, bool verbose){

}

void RNN_destroy(RNN_model *model) {

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
