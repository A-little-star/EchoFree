#include <math.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/act_func.h"
#include "../include/module.h"

SigmoidLayer* create_sigmoid_layer() {
    return (SigmoidLayer*)malloc(sizeof(SigmoidLayer));
}

void free_sigmoid_layer(SigmoidLayer* layer) {
    free(layer);
}

// Sigmoid 前向传播
Tensor* sigmoid_forward(SigmoidLayer* layer, Tensor* input) {
    Tensor* output = create_tensor(input->shape, input->ndim);
    for (int i = 0; i < input->size; i++) {
        output->data[i] = sigmoid(input->data[i]);
    }
    return output;
}
