#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../include/tensor.h"
#include "../include/model.h"

EchoFree * create_echofree(int stream) {
    int encoder_mic_channels[] = {1, 8, 16, 24, 32};
    int encoder_ref_channels[] = {1, 8};
    int decoder_channels[] = {32, 24, 16, 8, 1};
    int hidden_dim = 192;
    EchoFree *model = (EchoFree*)malloc(sizeof(EchoFree));
    model->hidden_dim = hidden_dim;
    model->stream = stream;
    model->mic_enc[0] = create_encoder_block(encoder_mic_channels[0], encoder_mic_channels[1], stream);
    model->ref_enc = create_encoder_block(encoder_ref_channels[0], encoder_ref_channels[1], stream);
    model->mic_enc[1] = create_encoder_block(encoder_mic_channels[1] + encoder_ref_channels[1], encoder_mic_channels[2], stream);
    model->mic_enc[2] = create_encoder_block(encoder_mic_channels[2], encoder_mic_channels[3], stream);
    model->mic_enc[3] = create_encoder_block(encoder_mic_channels[3], encoder_mic_channels[4], stream);
    model->bottleneck = create_bottleneck(hidden_dim);
    model->dec[0] = create_decoder_block(decoder_channels[0], decoder_channels[1], 0, 0, stream);
    model->dec[1] = create_decoder_block(decoder_channels[1], decoder_channels[2], 0, 0, stream);
    model->dec[2] = create_decoder_block(decoder_channels[2], decoder_channels[3], 0, 0, stream);
    model->dec[3] = create_decoder_block(decoder_channels[3], decoder_channels[4], 1, 1, stream);
    model->fc = create_linear_layer(110, 100);
    model->sigmoid = create_sigmoid_layer();
    return model;
}

void echofree_reset_buffer(EchoFree* model) {
    if (model->stream) {
        for (int i = 0; i < 4; i ++ ) encoder_block_reset_buffer(model->mic_enc[i]);
        encoder_block_reset_buffer(model->ref_enc);
        for (int i = 0; i < 4; i ++ ) decoder_block_reset_buffer(model->dec[i]);
    }
}

Parameter* echofree_load_params(EchoFree *model, ModelStateDict *sd) {
    Parameter *params = sd->params;
    Parameter *p1 = encoderblock_load_params(model->mic_enc[0], params);
    Parameter *p2 = encoderblock_load_params(model->mic_enc[1], p1);
    Parameter *p3 = encoderblock_load_params(model->mic_enc[2], p2);
    Parameter *p4 = encoderblock_load_params(model->mic_enc[3], p3);
    Parameter *p5 = encoderblock_load_params(model->ref_enc, p4);
    Parameter *p6 = bottleneck_load_params(model->bottleneck, p5);

    Parameter *p7 = decoderblock_load_params(model->dec[0], p6);
    Parameter *p8 = decoderblock_load_params(model->dec[1], p7);
    Parameter *p9 = decoderblock_load_params(model->dec[2], p8);
    Parameter *p10 = decoderblock_load_params(model->dec[3], p9);

    Parameter *p11 = linear_load_params(model->fc, p10);

    // assert(p5 == sd->params + sd->size);
    return p10;
}

void free_echofree(EchoFree *model) {
    if (model) {
        for (int i = 0; i < 4; i ++ ) free_encoder_block(model->mic_enc[i]);
        free_encoder_block(model->ref_enc);
        free_bottleneck(model->bottleneck);
        for (int i = 0; i < 4; i ++ ) free_decoder_block(model->dec[i]);
        free(model);
    }
}

Tensor *echofree_forward(EchoFree *model, Tensor *mic, Tensor *ref, Tensor *hidden_state, Tensor *cell_state) {
    Tensor *mic_enc1_out = encoderblock_forward(model->mic_enc[0], mic);
    Tensor *mic_ref1_out = encoderblock_forward(model->ref_enc, ref);
    Tensor *enc1_out_cat = concatenate(mic_ref1_out, mic_enc1_out, 0);
    Tensor *enc2_out = encoderblock_forward(model->mic_enc[1], enc1_out_cat);
    Tensor *enc3_out = encoderblock_forward(model->mic_enc[2], enc2_out);
    Tensor *enc4_out = encoderblock_forward(model->mic_enc[3], enc3_out);
    
    // Tensor *hidden_state = create_tensor((int[]){model->hidden_dim}, 1);
    // for (int i = 0; i < hidden_state->size; i ++ )
    //     hidden_state->data[i] = 0;
    Tensor *feats_out = bottleneck_forward(model->bottleneck, enc4_out, hidden_state, cell_state);
    
    Tensor *dec1_out = decoderblock_forward(model->dec[0], enc4_out, feats_out);
    Tensor *dec2_out = decoderblock_forward(model->dec[1], enc3_out, dec1_out);
    Tensor *dec3_out = decoderblock_forward(model->dec[2], enc2_out, dec2_out);
    Tensor *dec4_out = decoderblock_forward(model->dec[3], mic_enc1_out, dec3_out);

    Tensor *fc_out = linear_forward(model->fc, dec4_out);
    Tensor *output = sigmoid_forward(model->sigmoid, fc_out);
    
    delete_tensor(mic_enc1_out);
    delete_tensor(mic_ref1_out);
    delete_tensor(enc1_out_cat);
    delete_tensor(enc2_out);
    delete_tensor(enc3_out);
    delete_tensor(enc4_out);
    // delete_tensor(hidden_state);
    delete_tensor(feats_out);
    delete_tensor(dec1_out);
    delete_tensor(dec2_out);
    delete_tensor(dec3_out);
    delete_tensor(dec4_out);
    delete_tensor(fc_out);

    return output;
}