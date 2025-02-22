#ifndef MODEL_H
#define MODEL_H
#include "module.h"

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;

    DepthwiseConv2DLayer *conv2d;
    BatchNormLayer *bn;
    ELULayer *act;
    int stream;
} EncoderBlock;

typedef struct {
    int hidden_dim;
    GRULayer *rnn;
    LinearLayer *fc;
} BottleNeck;

typedef struct {
    int in_channels;
    int out_channels;
    int use_res; // 0 1
    int is_last; // 0 1
    SkipBlock* skip;
    ResidualBlock* resblock;
    SubPixelConv* subpixconv;
    BatchNormLayer* bn;
    ELULayer* act;
    int stream;
} DecoderBlock;

typedef struct {
    int hidden_dim;
    EncoderBlock *mic_enc[4];
    EncoderBlock *ref_enc;
    BottleNeck *bottleneck;
    DecoderBlock *dec[4];
    LinearLayer *fc;
    SigmoidLayer *sigmoid;
    int stream;
} EchoFree;

EncoderBlock *create_encoder_block(int in_channels, int out_channels, int stream);
void encoder_block_reset_buffer(EncoderBlock* block);
void free_encoder_block(EncoderBlock *block);
Parameter* encoderblock_load_params(EncoderBlock *block, Parameter *params);
Tensor *encoderblock_forward(EncoderBlock *block, Tensor *input);

BottleNeck* create_bottleneck(int hidden_dim);
void free_bottleneck(BottleNeck *btnk);
Parameter* bottleneck_load_params(BottleNeck *btnk, Parameter *params);
Tensor *bottleneck_forward(BottleNeck *btnk, Tensor *input, Tensor *hidden_state, Tensor *cell_state);

DecoderBlock* create_decoder_block(int in_channels, int out_channels, int use_res, int is_last, int stream);
void decoder_block_reset_buffer(DecoderBlock* block);
void free_decoder_block(DecoderBlock* block);
Parameter* decoderblock_load_params(DecoderBlock* block, Parameter* params);
Tensor* decoderblock_forward(DecoderBlock* block, Tensor* en, Tensor* de);

EchoFree* create_echofree(int stream);
void echofree_reset_buffer(EchoFree* model);
Parameter* echofree_load_params(EchoFree *model, ModelStateDict *sd);
void free_echofree(EchoFree *model);
Tensor *echofree_forward(EchoFree *model, Tensor *mic, Tensor *ref, Tensor* hidden_state, Tensor* cell_state);

#endif