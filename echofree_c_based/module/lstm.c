#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "../include/tensor.h"
#include "../include/module.h"
#include "../include/act_func.h"
#include "../include/parser.h"

// 创建 LSTM 层
LSTMLayer* create_lstm_layer(int input_size, int hidden_size) {
    LSTMLayer* layer = (LSTMLayer*)malloc(sizeof(LSTMLayer));
    layer->input_size = input_size;
    layer->hidden_size = hidden_size;

    int input_hidden_size = input_size * hidden_size;
    int hidden_hidden_size = hidden_size * hidden_size;

    // 分配内存
    layer->W_ii = (float*)malloc(input_hidden_size * sizeof(float));
    layer->W_if = (float*)malloc(input_hidden_size * sizeof(float));
    layer->W_ig = (float*)malloc(input_hidden_size * sizeof(float));
    layer->W_io = (float*)malloc(input_hidden_size * sizeof(float));

    layer->W_hi = (float*)malloc(hidden_hidden_size * sizeof(float));
    layer->W_hf = (float*)malloc(hidden_hidden_size * sizeof(float));
    layer->W_hg = (float*)malloc(hidden_hidden_size * sizeof(float));
    layer->W_ho = (float*)malloc(hidden_hidden_size * sizeof(float));

    layer->b_ii = (float*)calloc(hidden_size, sizeof(float));
    layer->b_if = (float*)calloc(hidden_size, sizeof(float));
    layer->b_ig = (float*)calloc(hidden_size, sizeof(float));
    layer->b_io = (float*)calloc(hidden_size, sizeof(float));

    layer->b_hi = (float*)calloc(hidden_size, sizeof(float));
    layer->b_hf = (float*)calloc(hidden_size, sizeof(float));
    layer->b_hg = (float*)calloc(hidden_size, sizeof(float));
    layer->b_ho = (float*)calloc(hidden_size, sizeof(float));

    // 初始化权重和偏置
    for (int i = 0; i < input_hidden_size; i++) {
        layer->W_ii[i] = layer->W_if[i] = layer->W_ig[i] = layer->W_io[i] = 0.1f;
    }
    for (int i = 0; i < hidden_hidden_size; i++) {
        layer->W_hi[i] = layer->W_hf[i] = layer->W_hg[i] = layer->W_ho[i] = 0.1f;
    }

    return layer;
}

// 释放 LSTM 层的内存
void free_lstm_layer(LSTMLayer* layer) {
    free(layer->W_ii); free(layer->W_if); free(layer->W_ig); free(layer->W_io);
    free(layer->W_hi); free(layer->W_hf); free(layer->W_hg); free(layer->W_ho);
    free(layer->b_ii); free(layer->b_if); free(layer->b_ig); free(layer->b_io);
    free(layer->b_hi); free(layer->b_hf); free(layer->b_hg); free(layer->b_ho);
    free(layer);
}

Parameter* lstm_load_params(LSTMLayer* layer, Parameter* params) {
    int input_size = layer->input_size;
    int hidden_size = layer->hidden_size;
    int input_hidden_size = input_size * hidden_size;
    int hidden_hidden_size = hidden_size * hidden_size;
    if (params[0].size != 4 * input_hidden_size) {
        printf("load parameters error: %s\n", params[0].name);
        assert(params[0].size == 4 * input_hidden_size);
    }
    if (params[1].size != 4 * hidden_hidden_size) {
        printf("load parameters error: %s\n", params[1].name);
        assert(params[1].size == 4 * hidden_hidden_size);
    }
    if (params[2].size != 4 * hidden_size) {
        printf("load parameters error: %s\n", params[2].name);
        assert(params[2].size == 4 * hidden_size);
    }
    if (params[3].size != 4 * hidden_size) {
        printf("load parameters error: %s\n", params[3].name);
        assert(params[3].size == 4 * hidden_size);
    }
    memcpy(layer->W_ii, params[0].values + 0 * input_hidden_size, input_hidden_size * sizeof(float));
    memcpy(layer->W_if, params[0].values + 1 * input_hidden_size, input_hidden_size * sizeof(float));
    memcpy(layer->W_ig, params[0].values + 2 * input_hidden_size, input_hidden_size * sizeof(float));
    memcpy(layer->W_io, params[0].values + 3 * input_hidden_size, input_hidden_size * sizeof(float));

    memcpy(layer->W_hi, params[1].values + 0 * hidden_hidden_size, hidden_hidden_size * sizeof(float));
    memcpy(layer->W_hf, params[1].values + 1 * hidden_hidden_size, hidden_hidden_size * sizeof(float));
    memcpy(layer->W_hg, params[1].values + 2 * hidden_hidden_size, hidden_hidden_size * sizeof(float));
    memcpy(layer->W_ho, params[1].values + 3 * hidden_hidden_size, hidden_hidden_size * sizeof(float));

    memcpy(layer->b_ii, params[2].values + 0 * hidden_size, hidden_size * sizeof(float));
    memcpy(layer->b_if, params[2].values + 1 * hidden_size, hidden_size * sizeof(float));
    memcpy(layer->b_ig, params[2].values + 2 * hidden_size, hidden_size * sizeof(float));
    memcpy(layer->b_io, params[2].values + 3 * hidden_size, hidden_size * sizeof(float));

    memcpy(layer->b_hi, params[3].values + 0 * hidden_size, hidden_size * sizeof(float));
    memcpy(layer->b_hf, params[3].values + 1 * hidden_size, hidden_size * sizeof(float));
    memcpy(layer->b_hg, params[3].values + 2 * hidden_size, hidden_size * sizeof(float));
    memcpy(layer->b_ho, params[3].values + 3 * hidden_size, hidden_size * sizeof(float));

    return params + 4;
}

// LSTM 前向传播
Tensor* lstm_forward(LSTMLayer* layer, Tensor* input, Tensor* hidden_state, Tensor* cell_state) {
    // input shape: [seq_len, input_size]
    // hidden_state shape: [hidden_size]
    // cell_state shape: [hidden_size]
    int input_size = layer->input_size;
    int hidden_size = layer->hidden_size;
    int seq_len = input->shape[0];
    assert(input_size == input->shape[1]);
    assert(hidden_size == hidden_state->shape[0]);
    assert(hidden_size == cell_state->shape[0]);

    Tensor* output = create_tensor((int[]){input->shape[0], hidden_size}, 2);

    // 临时变量
    float *i_gate = (float*)malloc(hidden_size * sizeof(float));
    float *f_gate = (float*)malloc(hidden_size * sizeof(float));
    float *g_gate = (float*)malloc(hidden_size * sizeof(float));
    float *o_gate = (float*)malloc(hidden_size * sizeof(float));
    float *c_new = (float*)malloc(hidden_size * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        float* x_t = input->data + t * input_size; // 当前时间步的输入
        float* o_t = output->data + t * hidden_size; // 当前时间步的输出

        // 输入门 i
        for (int i = 0; i < hidden_size; i++) {
            float Wx_i = 0.0f, Wh_i = 0.0f;
            for (int j = 0; j < input_size; j++) {
                Wx_i += x_t[j] * layer->W_ii[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                Wh_i += hidden_state->data[j] * layer->W_hi[i * hidden_size + j];
            }
            i_gate[i] = sigmoid(Wx_i + Wh_i + layer->b_ii[i] + layer->b_hi[i]);
        }

        // 遗忘门 f
        for (int i = 0; i < hidden_size; i++) {
            float Wx_f = 0.0f, Wh_f = 0.0f;
            for (int j = 0; j < input_size; j++) {
                Wx_f += x_t[j] * layer->W_if[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                Wh_f += hidden_state->data[j] * layer->W_hf[i * hidden_size + j];
            }
            f_gate[i] = sigmoid(Wx_f + Wh_f + layer->b_if[i] + layer->b_hf[i]);
        }

        // 候选记忆 g
        for (int i = 0; i < hidden_size; i++) {
            float Wx_g = 0.0f, Wh_g = 0.0f;
            for (int j = 0; j < input_size; j++) {
                Wx_g += x_t[j] * layer->W_ig[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                Wh_g += hidden_state->data[j] * layer->W_hg[i * hidden_size + j];
            }
            g_gate[i] = tanh_activation(Wx_g + Wh_g + layer->b_ig[i] + layer->b_hg[i]);
        }

        // 输出门 o
        for (int i = 0; i < hidden_size; i++) {
            float Wx_o = 0.0f, Wh_o = 0.0f;
            for (int j = 0; j < input_size; j++) {
                Wx_o += x_t[j] * layer->W_io[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                Wh_o += hidden_state->data[j] * layer->W_ho[i * hidden_size + j];
            }
            o_gate[i] = sigmoid(Wx_o + Wh_o + layer->b_io[i] + layer->b_ho[i]);
        }

        // 更新细胞状态 c
        for (int i = 0; i < hidden_size; i++) {
            c_new[i] = f_gate[i] * cell_state->data[i] + i_gate[i] * g_gate[i];
        }

        // 更新隐藏状态 h
        for (int i = 0; i < hidden_size; i++) {
            hidden_state->data[i] = o_gate[i] * tanh_activation(c_new[i]);
        }

        // 将隐藏状态保存到输出
        memcpy(o_t, hidden_state->data, hidden_size * sizeof(float));
        memcpy(cell_state->data, c_new, hidden_size * sizeof(float));
    }

    // 释放临时内存
    free(i_gate); free(f_gate); free(g_gate); free(o_gate); free(c_new);

    return output;
}
