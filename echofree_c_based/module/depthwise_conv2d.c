#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/tensor.h"
#include "../include/module.h"
#include "../include/parser.h"

// 注意这里的深度可分离卷积和真正的深度可分离卷积不一样，为了方便进行量化，将分组卷积改为了普通卷积

DepthwiseConv2DLayer* create_depthwise_conv2d_layer(int in_channels, int out_channels,
                                int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int padding_h, int padding_w, int stream) {
    DepthwiseConv2DLayer* layer = (DepthwiseConv2DLayer*)malloc(sizeof(DepthwiseConv2DLayer));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_h = kernel_h;
    layer->kernel_w = kernel_w;
    layer->stride_h = stride_h;
    layer->stride_w = stride_w;
    layer->padding_h = padding_h;
    layer->padding_w = padding_w;
    layer->stream = stream;

    layer->depth_conv = create_conv2d_layer(
        in_channels, in_channels, 
        kernel_h, kernel_w, 
        stride_h, stride_w, 
        padding_h, padding_w, 
        in_channels, stream
    );
    layer->point_conv = create_conv2d_layer(
        in_channels, out_channels,
        1, 1,
        1, 1, 
        0, 0,
        1, stream
    );
    return layer;
};

void depthwise_conv2d_reset_buffer(DepthwiseConv2DLayer* layer) {
    if (layer->stream) conv2d_reset_buffer(layer->depth_conv);
}

Parameter* depthwise_conv2d_load_params(DepthwiseConv2DLayer *layer, Parameter *params) {
    Parameter *p1 = conv2d_load_params(layer->depth_conv, params);
    Parameter *p2 = conv2d_load_params(layer->point_conv, p1);
    return p2;
}

void free_depthwise_conv2d_layer(DepthwiseConv2DLayer *layer) {
    free_conv2d_layer(layer->depth_conv);
    free_conv2d_layer(layer->point_conv);
    free(layer);
    layer = NULL;
}

Tensor* depthwise_conv2d_forward(DepthwiseConv2DLayer *layer, Tensor *input) {
    Tensor* mid = conv2d_forward(layer->depth_conv, input);
    Tensor* out = conv2d_forward(layer->point_conv, mid);
    // 销毁中间变量
    delete_tensor(mid);
    return out;
}