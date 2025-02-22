#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/parser.h"
#include "../include/tensor.h"
#include "../include/module.h"

SubPixelConv* create_subpixelconv(int in_channels, int out_channels) {
    SubPixelConv* block = (SubPixelConv*)malloc(sizeof(SubPixelConv));
    block->in_channels = in_channels;
    block->out_channels = out_channels;
    block->conv = create_conv2d_layer(
        in_channels, out_channels * 2,
        1, 1, 
        1, 1, 
        0, 0,
        1, 0
    );
    return block;
}

void free_subpixelconv(SubPixelConv *block) {
    if (block) {
        free_conv2d_layer(block->conv);
        free(block);
    }
}

Parameter* subpixelconv_load_params(SubPixelConv* block, Parameter* params) {
    Parameter* p = conv2d_load_params(block->conv, params);
    return p;
}

Tensor* subpixelconv_forward(SubPixelConv* block, Tensor* input) {
    Tensor* mid1 = conv2d_forward(block->conv, input);
    int c = mid1->shape[0], t = mid1->shape[1], f = mid1->shape[2];
    int new_shape[] = {c / 2, 2, t, f};
    Tensor* mid2 = reshape(mid1, new_shape, 4); // [C/2, 2, T, F]
    int po[] = {0, 2, 3, 1};
    Tensor* mid3 = permute(mid2, po); // [C/2, T, F, 2]
    int new_shape2[] = {c/2, t, f*2};
    Tensor* out = reshape(mid3, new_shape2, 3);
    delete_tensor(mid1);
    delete_tensor(mid2);
    delete_tensor(mid3);
    return out;
}