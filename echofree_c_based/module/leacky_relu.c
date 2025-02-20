#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../include/tensor.h"
#include "../include/module.h"

// 创建 LeakyReLU 层
LeakyReLULayer* create_leaky_relu_layer(float negative_slope) {
    LeakyReLULayer* layer = (LeakyReLULayer*)malloc(sizeof(LeakyReLULayer));
    layer->negative_slope = negative_slope;
    return layer;
}

// 释放 LeakyReLU 层的内存
void free_leaky_relu_layer(LeakyReLULayer* layer) {
    free(layer);
}

// LeakyReLU 前向推理
Tensor* leaky_relu_forward(LeakyReLULayer* layer, Tensor* input) {
    assert(input->ndim >= 1); // 确保输入至少有 1 个维度

    // 创建输出 Tensor
    Tensor* output = create_tensor(input->shape, input->ndim);

    // 获取 negative_slope 参数
    float negative_slope = layer->negative_slope;

    // 遍历 Tensor 中的每个元素并应用 LeakyReLU 函数
    for (int i = 0; i < input->size; i++) {
        float x = input->data[i];
        if (x > 0) {
            output->data[i] = x;
        } else {
            output->data[i] = negative_slope * x;
        }
    }

    return output;
}
