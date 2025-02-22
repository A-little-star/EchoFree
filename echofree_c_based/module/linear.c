#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/module.h"
#include "../include/tensor.h"
#include "../include/parser.h"

// 创建全连接层
LinearLayer* create_linear_layer(int input_size, int output_size) {
    LinearLayer* layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;

    // 分配权重和偏置
    layer->weight = (float*)malloc(output_size * input_size * sizeof(float));
    layer->bias = (float*)malloc(output_size * sizeof(float));

    // 初始化权重和偏置为随机值或零（示例中随机值）
    for (int i = 0; i < output_size * input_size; ++i) {
        layer->weight[i] = (float)rand() / RAND_MAX - 0.5f; // 随机初始化
    }
    for (int i = 0; i < output_size; ++i) {
        layer->bias[i] = (float)rand() / RAND_MAX - 0.5f; // 随机初始化
    }

    return layer;
}

// 释放全连接层
void free_linear_layer(LinearLayer* layer) {
    if (layer) {
        free(layer->weight);
        free(layer->bias);
        free(layer);
    }
}

Parameter* linear_load_params(LinearLayer* layer, Parameter* params) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    assert(params[0].size == output_size * input_size);
    assert(params[1].size == output_size);
    memcpy(layer->weight, params[0].values, output_size * input_size * sizeof(float));
    memcpy(layer->bias, params[1].values, output_size * sizeof(float));
    return params + 2;
}

// 全连接层前向传播
Tensor* linear_forward(LinearLayer* layer, Tensor* input) {
    // 检查输入尺寸
    if (input->shape[input->ndim - 1] != layer->input_size) {
        fprintf(stderr, "Input size does not match Linear layer input_size.\n");
        return NULL;
    }

    // 计算输出形状
    int batch_size = input->size / layer->input_size;
    int output_size = layer->output_size;

    // 分配输出Tensor
    Tensor* output = (Tensor*)malloc(sizeof(Tensor));
    output->ndim = input->ndim;
    output->shape = (int*)malloc(output->ndim * sizeof(int));
    memcpy(output->shape, input->shape, (input->ndim - 1) * sizeof(int));
    output->shape[input->ndim - 1] = output_size;
    output->size = batch_size * output_size;
    output->data = (float*)malloc(output->size * sizeof(float));

    // 计算全连接层的输出
    for (int b = 0; b < batch_size; ++b) {
        for (int o = 0; o < output_size; ++o) {
            float sum = layer->bias[o];
            for (int i = 0; i < layer->input_size; ++i) {
                sum += input->data[b * layer->input_size + i] * layer->weight[o * layer->input_size + i];
            }
            output->data[b * output_size + o] = sum;
        }
    }

    return output;
}

// int main() {
//     // 定义输入向量
//     float input[3] = {1.0, 2.0, 3.0};

//     // 创建隐藏层和输出层
//     LinearLayer hidden_layer = create_linear_layer(3, 4);
//     LinearLayer output_layer = create_linear_layer(4, 2);

//     // 定义隐藏层和输出层的输出向量
//     float hidden_output[4];
//     float final_output[2];

//     // 前向传播
//     linear_forward(&hidden_layer, input, hidden_output);
//     linear_forward(&output_layer, hidden_output, final_output);

//     // 打印输出
//     printf("Final Output:\n");
//     for (int i = 0; i < 2; i++) {
//         printf("%f ", final_output[i]);
//     }
//     printf("\n");

//     // 释放内存
//     free_linear_layer(&hidden_layer);
//     free_linear_layer(&output_layer);

//     return 0;
// }
