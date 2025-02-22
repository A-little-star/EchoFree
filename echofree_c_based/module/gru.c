#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "../include/tensor.h"
#include "../include/module.h"
#include "../include/act_func.h"
#include "../include/parser.h"

// 创建 GRU 层
GRULayer* create_gru_layer(int input_size, int hidden_size) {
    GRULayer* layer = (GRULayer*)malloc(sizeof(GRULayer));
    layer->input_size = input_size;
    layer->hidden_size = hidden_size;

    int input_hidden_size = input_size * hidden_size;
    int hidden_hidden_size = hidden_size * hidden_size;

    // 分配内存
    layer->W_ir = (float*)malloc(input_hidden_size * sizeof(float));
    layer->W_iz = (float*)malloc(input_hidden_size * sizeof(float));
    layer->W_in = (float*)malloc(input_hidden_size * sizeof(float));

    layer->W_hr = (float*)malloc(hidden_hidden_size * sizeof(float));
    layer->W_hz = (float*)malloc(hidden_hidden_size * sizeof(float));
    layer->W_hn = (float*)malloc(hidden_hidden_size * sizeof(float));

    layer->b_ir = (float*)calloc(hidden_size, sizeof(float));
    layer->b_iz = (float*)calloc(hidden_size, sizeof(float));
    layer->b_in = (float*)calloc(hidden_size, sizeof(float));

    layer->b_hr = (float*)calloc(hidden_size, sizeof(float));
    layer->b_hz = (float*)calloc(hidden_size, sizeof(float));
    layer->b_hn = (float*)calloc(hidden_size, sizeof(float));

    // 初始化权重和偏置（可以根据需要使用更复杂的初始化方法）
    for (int i = 0; i < input_hidden_size; i++) {
        layer->W_ir[i] = layer->W_iz[i] = layer->W_in[i] = 0.1f; // 示例初始化
    }
    for (int i = 0; i < hidden_hidden_size; i++) {
        layer->W_hr[i] = layer->W_hz[i] = layer->W_hn[i] = 0.1f;
    }

    return layer;
}

// 释放 GRU 层的内存
void free_gru_layer(GRULayer* layer) {
    free(layer->W_ir); free(layer->W_iz); free(layer->W_in);
    free(layer->W_hr); free(layer->W_hz); free(layer->W_hn);
    free(layer->b_ir); free(layer->b_iz); free(layer->b_in);
    free(layer->b_hr); free(layer->b_hz); free(layer->b_hn);
    free(layer);
}

Parameter* gru_load_params(GRULayer* layer, Parameter* params) {
    int input_size = layer->input_size;
    int hidden_size = layer->hidden_size;
    int input_hidden_size = input_size * hidden_size;
    int hidden_hidden_size = hidden_size * hidden_size;
    assert(params[0].size == 3 * input_hidden_size);
    assert(params[1].size == 3 * hidden_hidden_size);
    assert(params[2].size == 3 * hidden_size);
    assert(params[3].size == 3 * hidden_size);

    memcpy(layer->W_ir, params[0].values, input_hidden_size * sizeof(float));
    memcpy(layer->W_iz, params[0].values + input_hidden_size, input_hidden_size * sizeof(float));
    memcpy(layer->W_in, params[0].values + 2 * input_hidden_size, input_hidden_size * sizeof(float));

    memcpy(layer->W_hr, params[1].values, hidden_hidden_size * sizeof(float));
    memcpy(layer->W_hz, params[1].values + hidden_hidden_size, hidden_hidden_size * sizeof(float));
    memcpy(layer->W_hn, params[1].values + 2 * hidden_hidden_size, hidden_hidden_size * sizeof(float));

    memcpy(layer->b_ir, params[2].values, hidden_size * sizeof(float));
    memcpy(layer->b_iz, params[2].values + hidden_size, hidden_size * sizeof(float));
    memcpy(layer->b_in, params[2].values + 2 * hidden_size, hidden_size * sizeof(float));

    memcpy(layer->b_hr, params[3].values, hidden_size * sizeof(float));
    memcpy(layer->b_hz, params[3].values + hidden_size, hidden_size * sizeof(float));
    memcpy(layer->b_hn, params[3].values + 2 * hidden_size, hidden_size * sizeof(float));

    return params + 4;
}

// GRU 前向传播
Tensor* gru_forward(GRULayer* layer, Tensor* input, Tensor* hidden_state) {
    // input shape: [seq_len, input_size]
    // hidden_state shape: [hidden_size]
    int input_size = layer->input_size;
    int hidden_size = layer->hidden_size;
    int seq_len = input->shape[0];
    assert(input_size == input->shape[1]);
    // assert(hidden_size == hidden_state->shape[0]);
    if (hidden_size != hidden_state->shape[0]) {
        printf("In gru_forward: hidden_size = %d, but hidden_state's length is %d\n", hidden_size, hidden_state->shape[0]);
        assert(0);
    }

    Tensor* output = create_tensor((int[]){input->shape[0], hidden_size}, 2);

    // 临时变量
    float *r = (float*)malloc(hidden_size * sizeof(float));
    float *z = (float*)malloc(hidden_size * sizeof(float));
    float *n = (float*)malloc(hidden_size * sizeof(float));
    float *h_new = (float*)malloc(hidden_size * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        float* x_t = input->data + t * input_size; // 当前时间步的输入
        float* o_t = output->data + t * hidden_size; // 当前时间步的输出

        // 计算重置门 r
        for (int i = 0; i < hidden_size; i++) {
            float Wx_r = 0.0f, Wh_r = 0.0f;
            for (int j = 0; j < input_size; j++) {
                Wx_r += x_t[j] * layer->W_ir[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                Wh_r += hidden_state->data[j] * layer->W_hr[i * hidden_size + j];
            }
            r[i] = sigmoid(Wx_r + Wh_r + layer->b_ir[i] + layer->b_hr[i]);
        }

        // 计算更新门 z
        for (int i = 0; i < hidden_size; i++) {
            float Wx_z = 0.0f, Wh_z = 0.0f;
            for (int j = 0; j < input_size; j++) {
                Wx_z += x_t[j] * layer->W_iz[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                Wh_z += hidden_state->data[j] * layer->W_hz[i * hidden_size + j];
            }
            z[i] = sigmoid(Wx_z + Wh_z + layer->b_iz[i] + layer->b_hz[i]);
        }

        // 计算候选隐藏状态 n
        for (int i = 0; i < hidden_size; i++) {
            float Wx_n = 0.0f, Wh_n = 0.0f;
            for (int j = 0; j < input_size; j++) {
                Wx_n += x_t[j] * layer->W_in[i * input_size + j];
            }
            for (int j = 0; j < hidden_size; j++) {
                Wh_n += hidden_state->data[j] * layer->W_hn[i * hidden_size + j];
            }
            Wh_n += layer->b_hn[i];
            Wh_n *= r[i];
            n[i] = tanh_activation(Wx_n + Wh_n + layer->b_in[i]);
        }

        // 更新隐藏状态
        for (int i = 0; i < hidden_size; i++) {
            h_new[i] = (1.0f - z[i]) * n[i] + z[i] * hidden_state->data[i];
        }

        // 将新的隐藏状态保存到输出
        memcpy(o_t, h_new, hidden_size * sizeof(float));
        memcpy(hidden_state->data, h_new, hidden_size * sizeof(float));
    }

    // if (layer->stream) memcpy(layer->hidden_state->data, hidden_state->data, hidden_size * sizeof(float));

    // 释放临时内存
    free(r); free(z); free(n); free(h_new);

    return output;
}
