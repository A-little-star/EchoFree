#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/model.h"

BottleNeck* create_bottleneck(int hidden_dim) {
    BottleNeck *btnk = (BottleNeck*)malloc(sizeof(BottleNeck));
    btnk->hidden_dim = hidden_dim;
    btnk->rnn = create_gru_layer(hidden_dim, hidden_dim);
    // btnk->rnn = create_lstm_layer(hidden_dim, hidden_dim);
    btnk->fc = create_linear_layer(hidden_dim, hidden_dim);
    return btnk;
}

void free_bottleneck(BottleNeck *btnk) {
    if (btnk) {
        free_gru_layer(btnk->rnn);
        // free_lstm_layer(btnk->rnn);
        free_linear_layer(btnk->fc);
        free(btnk);
    }
}

Parameter* bottleneck_load_params(BottleNeck *btnk, Parameter *params) {
    Parameter *p1 = gru_load_params(btnk->rnn, params);
    // Parameter *p1 = lstm_load_params(btnk->rnn, params);
    Parameter *p2 = linear_load_params(btnk->fc, p1);
    return p2;
}

Tensor *bottleneck_forward(BottleNeck *btnk, Tensor *input, Tensor *hidden_state, Tensor *cell_state) {
    // input shape: [C, T, F]
    Tensor *feats_in1 = permute(input, (int[]){1, 0, 2}); // [T, C, F]
    int T = feats_in1->shape[0], C = feats_in1->shape[1], F = feats_in1->shape[2];
    Tensor *feats_in2 = reshape(feats_in1, (int[]){T, C*F}, 2); // [T, C*F]
    Tensor *gru_out = gru_forward(btnk->rnn, feats_in2, hidden_state);
    // Tensor *lstm_out = lstm_forward(btnk->rnn, feats_in2, hidden_state, cell_state);
    Tensor *fc_out = linear_forward(btnk->fc, gru_out); // [T, C*F]
    Tensor *fc_out1 = reshape(fc_out, (int[]){T, C, F}, 3);
    Tensor *out = permute(fc_out1, (int[]){1, 0, 2});
    delete_tensor(feats_in1);
    delete_tensor(feats_in2);
    delete_tensor(gru_out);
    delete_tensor(fc_out);
    delete_tensor(fc_out1);
    return out;
}