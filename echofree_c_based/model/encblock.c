#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/model.h"

EncoderBlock *create_encoder_block(int in_channels, int out_channels, int stream) {
    EncoderBlock *block = (EncoderBlock*)malloc(sizeof(EncoderBlock));
    block->in_channels = in_channels;
    block->out_channels = out_channels;
    block->kernel_h = 4;
    block->kernel_w = 3;
    block->stride_h = 1;
    block->stride_w = 2;
    block->stream = stream;
    block->conv2d = create_depthwise_conv2d_layer(
        in_channels, out_channels, 
        4, 3, 
        1, 2, 
        3, 0, stream
    );
    block->bn = create_batchnorm_layer(out_channels, 1e-5);
    block->act = create_elu_layer(1);
    // block->act = create_leaky_relu_layer(0.01);

    return block;
}

void encoder_block_reset_buffer(EncoderBlock* block) {
    if (block->stream) depthwise_conv2d_reset_buffer(block->conv2d);
}

void free_encoder_block(EncoderBlock *block) {
    if (block) {
        free_depthwise_conv2d_layer(block->conv2d);
        free_batchnorm_layer(block->bn);
        free_elu_layer(block->act);
        free(block);
    }
}

Parameter* encoderblock_load_params(EncoderBlock *block, Parameter *params) {
    Parameter *p1 = depthwise_conv2d_load_params(block->conv2d, params);
    Parameter *p2 = batchnorm_load_params(block->bn, p1);
    return p2;
}

Tensor *encoderblock_forward(EncoderBlock *block, Tensor *input) {
    Tensor *mid0 = depthwise_conv2d_forward(block->conv2d, input);
    Tensor *mid1 = tensor_slice(mid0, (int[]){0, 0, 0}, (int[]){mid0->shape[0], mid0->shape[1] - 3, mid0->shape[2]});
    Tensor *mid2 = batchnorm_forward(block->bn, mid1);
    Tensor *output = elu_forward(block->act, mid2);
    delete_tensor(mid0);
    delete_tensor(mid1);
    delete_tensor(mid2);
    return output;
}