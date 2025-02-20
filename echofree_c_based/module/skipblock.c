#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/parser.h"
#include "../include/tensor.h"
#include "../include/module.h"

SkipBlock* create_skipblock(int hidden_channels) {
    SkipBlock* block = (SkipBlock*)malloc(sizeof(SkipBlock));
    block->hidden_channels = hidden_channels;
    block->conv = create_conv2d_layer(
        hidden_channels, hidden_channels,
        1, 1, 
        1, 1, 
        0, 0, 
        1, 0
    );
    return block;
}

void free_skipblock(SkipBlock *block) {
    if (block) {
        free_conv2d_layer(block->conv);
        free(block);
    }
}

Parameter* skipblock_load_params(SkipBlock* block, Parameter* params) {
    Parameter* p = conv2d_load_params(block->conv, params);
    return p;
}

Tensor* skipblock_forward(SkipBlock* block, Tensor* encoder_feats, Tensor* decoder_feats) {
    Tensor *feats = conv2d_forward(block->conv, encoder_feats);
    int f_en = encoder_feats->shape[encoder_feats->ndim - 1];
    int f_de = decoder_feats->shape[decoder_feats->ndim - 1];
    Tensor *decoder_feats2;
    if (f_en > f_de) {
        int pad[6] = {0, 0, 0, 0, 0, f_en - f_de};
        decoder_feats2 = tensor_pad(decoder_feats, pad);
    } else {
        decoder_feats2 = decoder_feats;
    }
    // TODO
    Tensor *out = tensor_add(encoder_feats, decoder_feats2);
    delete_tensor(feats);
    if (f_en > f_de) delete_tensor(decoder_feats2);
    return out;
}