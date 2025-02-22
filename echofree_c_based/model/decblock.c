#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/model.h"

DecoderBlock* create_decoder_block(int in_channels, int out_channels, int use_res, int is_last, int stream) {
    DecoderBlock* block = (DecoderBlock*)malloc(sizeof(DecoderBlock));
    block->in_channels = in_channels;
    block->out_channels = out_channels;
    block->use_res = use_res;
    block->is_last = is_last;
    block->stream = stream;
    block->skip = create_skipblock(in_channels);
    if (use_res) {
        block->resblock = create_residualblock(in_channels, stream);
    }
    block->subpixconv = create_subpixelconv(in_channels, out_channels);
    if (!is_last) {
        block->bn = create_batchnorm_layer(out_channels, 1e-5);
        block->act = create_elu_layer(1);
        // block->act = create_leaky_relu_layer(0.01);
    }
    return block;
}

void decoder_block_reset_buffer(DecoderBlock* block) {
    if (block->stream && block->use_res) residualblock_reset_buffer(block->resblock);
}

void free_decoder_block(DecoderBlock* block) {
    if (block) {
        free_skipblock(block->skip);
        if (block->use_res) free_residualblock(block->resblock);
        free_subpixelconv(block->subpixconv);
        if (!(block->is_last)) {
            free_batchnorm_layer(block->bn);
            free_elu_layer(block->act);
        }
        free(block);
    }
}

Parameter* decoderblock_load_params(DecoderBlock* block, Parameter* params) {
    Parameter* p1 = skipblock_load_params(block->skip, params);
    Parameter* p2;
    if (block->use_res) p2 = residualblock_load_params(block->resblock, p1);
    else p2 = p1;
    Parameter* p3 = subpixelconv_load_params(block->subpixconv, p2);
    Parameter *p4;
    if (!(block->is_last)) p4 = batchnorm_load_params(block->bn, p3);
    else p4 = p3;
    return p4;
}

Tensor* decoderblock_forward(DecoderBlock* block, Tensor* en, Tensor* de) {
    Tensor* skip_out = skipblock_forward(block->skip, en, de);
    Tensor* res_out;
    if (block->use_res) res_out = residualblock_forward(block->resblock, skip_out);
    else res_out = skip_out;
    Tensor* subpixconv_out = subpixelconv_forward(block->subpixconv, res_out);
    Tensor* out;
    if (!(block->is_last)) {
        Tensor* bn_out = batchnorm_forward(block->bn, subpixconv_out);
        out = elu_forward(block->act, bn_out);
        delete_tensor(bn_out);
    }
    else out = subpixconv_out;
    delete_tensor(skip_out);
    if (block->use_res) delete_tensor(res_out);
    if (!(block->is_last)) delete_tensor(subpixconv_out);
    return out;
}