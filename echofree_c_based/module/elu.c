#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../include/tensor.h"
#include "../include/module.h"

// 创建 ELU 层
ELULayer* create_elu_layer(float alpha) {
    ELULayer* layer = (ELULayer*)malloc(sizeof(ELULayer));
    layer->alpha = alpha;
    return layer;
}

// 释放 ELU 层的内存
void free_elu_layer(ELULayer* layer) {
    free(layer);
}

// ELU 前向推理
Tensor* elu_forward(ELULayer* layer, Tensor* input) {
    assert(input->ndim >= 1); // 确保输入至少有 1 个维度

    // 创建输出 Tensor
    Tensor* output = create_tensor(input->shape, input->ndim);

    // 获取 alpha 参数
    float alpha = layer->alpha;

    // 遍历 Tensor 中的每个元素并应用 ELU 函数
    for (int i = 0; i < input->size; i++) {
        float x = input->data[i];
        if (x > 0) {
            output->data[i] = x;
        } else {
            output->data[i] = alpha * (expf(x) - 1);
        }
    }

    return output;
}
