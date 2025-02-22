#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/parser.h"
#include "../include/tensor.h"
#include "../include/module.h"

ResidualBlock* create_residualblock(int hidden_channels, int stream) {
    ResidualBlock* block = (ResidualBlock*)malloc(sizeof(ResidualBlock));
    block->hidden_channels = hidden_channels;
    block->stream = stream;
    block->conv2d = create_conv2d_layer(
        hidden_channels, hidden_channels,
        4, 3,
        1, 1,
        3, 1, 
        1, stream
    );
    block->bn = create_batchnorm_layer(hidden_channels, 1e-5);
    block->act = create_elu_layer(1);
    return block;
}

void residualblock_reset_buffer(ResidualBlock* block) {
    if (block->stream) conv2d_reset_buffer(block->conv2d);
}

void free_residualblock(ResidualBlock* block) {
    if (block) {
        free_conv2d_layer(block->conv2d);
        free_batchnorm_layer(block->bn);
        free_elu_layer(block->act);
        free(block);
    }
}

Parameter* residualblock_load_params(ResidualBlock* block, Parameter* params) {
    Parameter* p1 = conv2d_load_params(block->conv2d, params);
    Parameter* p2 = batchnorm_load_params(block->bn, p1);
    return p2;
}

Tensor* residualblock_forward(ResidualBlock* block, Tensor* input) {
    Tensor* conv_out = conv2d_forward(block->conv2d, input);
    Tensor *mid = tensor_slice(conv_out, (int[]){0, 0, 0}, (int[]){conv_out->shape[0], conv_out->shape[1] - 3, conv_out->shape[2]});
    Tensor *bn_out = batchnorm_forward(block->bn, mid);
    Tensor *elu_out = elu_forward(block->act, bn_out);
    Tensor *out = tensor_add(elu_out, input);
    delete_tensor(conv_out);
    delete_tensor(mid);
    delete_tensor(bn_out);
    delete_tensor(elu_out);
    return out;
}