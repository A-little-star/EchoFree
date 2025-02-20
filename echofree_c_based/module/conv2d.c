#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/parser.h"
#include "../include/tensor.h"
#include "../include/module.h"

// 创建卷积层
Conv2DLayer* create_conv2d_layer(int in_channels, int out_channels,
                                 int kernel_h, int kernel_w,
                                 int stride_h, int stride_w,
                                 int padding_h, int padding_w,
                                 int group, int stream) {
    Conv2DLayer* layer = (Conv2DLayer*)malloc(sizeof(Conv2DLayer));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_h = kernel_h;
    layer->kernel_w = kernel_w;
    layer->stride_h = stride_h;
    layer->stride_w = stride_w;
    layer->padding_h = padding_h;
    layer->padding_w = padding_w;
    layer->group = group;
    layer->stream = stream;

    // 分配权重和偏置的内存
    int weight_size = out_channels * in_channels * kernel_h * kernel_w / group;
    layer->weights = (float *)malloc(weight_size * sizeof(float));
    layer->bias = (float *)malloc(out_channels * sizeof(float));

    // 初始化权重和偏置为随机值
    for (int i = 0; i < weight_size; i++) {
        layer->weights[i] = (float)1.0f; // 初始化为1.0
    }
    for (int i = 0; i < out_channels; i++) {
        layer->bias[i] = (float)0.0f;  // 偏置初始化为0
    }
    layer->buffer = NULL; // 表示一个尚未确定形状的缓冲区，将在forward中分配空间

    return layer;
}

void conv2d_reset_buffer(Conv2DLayer* layer) {
    if (layer->stream) {
        if (layer->buffer) {
            delete_tensor(layer->buffer);
            layer->buffer = NULL;
        }
    }
}

// 加载卷积层参数
Parameter* conv2d_load_params(Conv2DLayer* layer, Parameter *params) {
    int in_channels = layer->in_channels;
    int out_channels = layer->out_channels;
    int kernel_h = layer->kernel_h;
    int kernel_w = layer->kernel_w;
    int group = layer->group;
    float *weight = params[0].values;
    float *bias = params[1].values;

    int weight_size = out_channels * in_channels * kernel_h * kernel_w / group;
    
    if (weight_size != params[0].size) {
        printf("load parameters error: %s\n", params[0].name);
        assert(weight_size == params[0].size);
    }
    if (out_channels != params[1].size) {
        printf("load parameters error: %s\n", params[1].name);
        assert(out_channels == params[1].size);
    }

    // 初始化权重和偏置为随机值
    for (int i = 0; i < weight_size; i++) {
        layer->weights[i] = weight[i]; // 加载权重
    }
    for (int i = 0; i < out_channels; i++) {
        layer->bias[i] = bias[i];  // 加载偏置
    }
    return params + 2;
}

// 释放卷积层的内存
void free_conv2d_layer(Conv2DLayer *layer) {
    free(layer->weights);
    free(layer->bias);
    if (layer->stream && layer->buffer) delete_tensor(layer->buffer);
    layer->weights = NULL;
    layer->bias = NULL;
    free(layer);
    layer = NULL;
}

// 计算输出的高度和宽度
void compute_output_size(Conv2DLayer *layer, int in_h, int in_w, int *out_h, int *out_w, int stream) {
    if (stream) *out_h = (in_h + layer->padding_h - layer->kernel_h) / layer->stride_h + 1;
    else *out_h = (in_h + 2 * layer->padding_h - layer->kernel_h) / layer->stride_h + 1;
    *out_w = (in_w + 2 * layer->padding_w - layer->kernel_w) / layer->stride_w + 1;
}

// 2D 卷积的推理，支持分组卷积
Tensor* conv2d_forward(Conv2DLayer *layer, Tensor* input) {
    assert(input->ndim == 3);
    int in_channels = layer->in_channels;
    int out_channels = layer->out_channels;
    int kernel_h = layer->kernel_h;
    int kernel_w = layer->kernel_w;
    int stride_h = layer->stride_h;
    int stride_w = layer->stride_w;
    int padding_h = layer->padding_h;
    int padding_w = layer->padding_w;
    int group = layer->group;

    int stream = layer->stream;
    if (stream) {
        padding_h = 0;
        // layer->padding_h = 0;
    }

    // 取出输入张量的形状
    int in_c = input->shape[0], in_h = input->shape[1], in_w = input->shape[2];
    if (in_c != in_channels) {
        fprintf(stderr, "Conv2d layer's input channels doesn't match the input tensor's channels.\n");
        assert(0);
    };

    // 如果是流式推理，那么第一帧创建buffer
    Tensor* padded_input;
    if (stream && (kernel_h > 1)) {
        if (layer->buffer == NULL) {
            int buffer_shape[] = {in_c, kernel_h - 1, in_w};
            layer->buffer = create_tensor(buffer_shape, 3);
            for (int i = 0; i < layer->buffer->size; i ++ ) layer->buffer->data[i] = 0;
        }
        padded_input = concatenate(layer->buffer, input, 1);
        delete_tensor(layer->buffer);
        int start_index[] = {0, padded_input->shape[1] - (kernel_h - 1), 0};
        int end_index[] = {padded_input->shape[0], padded_input->shape[1], padded_input->shape[2]};
        layer->buffer = tensor_slice(padded_input, start_index, end_index);
    }
    else {
        padded_input = input;
    }

    // 重新计算形状参数
    in_c = padded_input->shape[0], in_h = padded_input->shape[1], in_w = padded_input->shape[2];

    // 每个组的输入和输出通道数
    int group_in_channels = in_channels / group;
    int group_out_channels = out_channels / group;

    // 计算输出的高度和宽度
    int out_h, out_w;
    compute_output_size(layer, in_h, in_w, &out_h, &out_w, stream);

    // 创建输出向量
    Tensor* output = create_tensor((int[]){out_channels, out_h, out_w}, 3);
    // Tensor output = create_tensor(out_channels, out_h, out_w);

    // 初始化输出为偏置
    memset(output->data, 0, out_h * out_w * out_channels * sizeof(float));
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                output->data[(oc * out_h + oh) * out_w + ow] = layer->bias[oc];
            }
        }
    }

    // 遍历每个组
    for (int g = 0; g < group; g++) {
        // 计算该组的输入和输出通道范围
        int in_offset = g * group_in_channels;
        int out_offset = g * group_out_channels;

        // 遍历该组的输出通道
        for (int oc = 0; oc < group_out_channels; oc++) {
            int out_channel = out_offset + oc;

            // 遍历输出高度
            for (int oh = 0; oh < out_h; oh++) {
                // 遍历输出宽度
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;

                    // 遍历输入通道
                    for (int ic = 0; ic < group_in_channels; ic++) {
                        int in_channel = in_offset + ic;

                        // 遍历卷积核
                        for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                int in_h_index = oh * stride_h + kh - padding_h;
                                int in_w_index = ow * stride_w + kw - padding_w;

                                // 检查边界
                                if (in_h_index >= 0 && in_h_index < in_h &&
                                    in_w_index >= 0 && in_w_index < in_w) {
                                    sum += padded_input->data[(in_channel * in_h + in_h_index) * in_w + in_w_index] *
                                           layer->weights[((out_channel * group_in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                                }
                            }
                        }
                    }

                    // 将结果存储到输出
                    output->data[(out_channel * out_h + oh) * out_w + ow] += sum;
                }
            }
        }
    }
    if (stream && (kernel_h > 1)) delete_tensor(padded_input);
    return output;
}