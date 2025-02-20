#ifndef MODULE_H
#define MODULE_H
#include "parser.h"
#include "tensor.h"

// 定义线性层的结构
typedef struct {
    int input_size;
    int output_size;
    float *weight;  // 权重矩阵 (output_size x input_size)
    float *bias;    // 偏置向量 (output_size)
} LinearLayer;

// 定义卷积层的结构体（包括权重、偏置等参数）
typedef struct {
    int in_channels;   // 输入通道数
    int out_channels;  // 输出通道数
    int kernel_h, kernel_w; // 卷积核尺寸
    int stride_h, stride_w; // 步幅
    int padding_h, padding_w; // 填充
    int group; // 分组

    // 权重和偏置
    float *weights; // 权重: [out_channels, in_channels, kernel_h, kernel_w]
    float *bias;    // 偏置: [out_channels]

    int stream;    // 取值0/1，是否流式推理
    Tensor *buffer; // 缓冲区，流式推理时存放过去的帧信息（只在stream=1时有意义）
} Conv2DLayer;

// GRU层定义
typedef struct {
    int input_size;  // 输入特征维度
    int hidden_size; // 隐藏层维度

    // 权重矩阵和偏置向量
    float *W_ir, *W_iz, *W_in; // 输入到各门的权重
    float *W_hr, *W_hz, *W_hn; // 隐藏状态到各门的权重
    float *b_ir, *b_iz, *b_in; // 输入偏置
    float *b_hr, *b_hz, *b_hn; // 隐藏状态偏置
} GRULayer;

// LSTM层定义
typedef struct {
    int input_size;  // 输入特征维度
    int hidden_size; // 隐藏层维度

    // 权重矩阵和偏置向量
    float *W_ii, *W_if, *W_ig, *W_io; // 输入到各门的权重
    float *W_hi, *W_hf, *W_hg, *W_ho; // 隐藏状态到各门的权重
    float *b_ii, *b_if, *b_ig, *b_io; // 输入偏置
    float *b_hi, *b_hf, *b_hg, *b_ho; // 隐藏状态偏置
} LSTMLayer;

// BatchNorm 层定义
typedef struct {
    int num_features;   // 输入通道数
    float* gamma;       // 缩放参数
    float* beta;        // 平移参数
    float* running_mean; // 训练阶段累积的均值
    float* running_var;  // 训练阶段累积的方差
    float eps;      // 防止除零的小值
} BatchNormLayer;

// ELU 层定义
typedef struct {
    float alpha;  // 控制负值部分的参数
} ELULayer;

// LeakyReLU 层定义
typedef struct {
    float negative_slope; // 控制负值部分的斜率
} LeakyReLULayer;

// Sigmoid 层定义
typedef struct {
    // Sigmoid 层不需要额外参数
} SigmoidLayer;

typedef struct {
    int in_channels;   // 输入通道数
    int out_channels;  // 输出通道数
    int kernel_h, kernel_w; // 卷积核尺寸
    int stride_h, stride_w; // 步幅
    int padding_h, padding_w; // 填充

    Conv2DLayer* depth_conv;
    Conv2DLayer* point_conv;
    int stream;
} DepthwiseConv2DLayer;

typedef struct {
    int hidden_channels;
    Conv2DLayer *conv;
} SkipBlock;

typedef struct {
    int in_channels;
    int out_channels;
    Conv2DLayer *conv;
} SubPixelConv;

typedef struct {
    int hidden_channels;
    Conv2DLayer* conv2d;
    BatchNormLayer* bn;
    ELULayer* act;
    int stream;
} ResidualBlock;

LinearLayer* create_linear_layer(int input_size, int output_size);
void free_linear_layer(LinearLayer* layer);
Parameter* linear_load_params(LinearLayer* layer, Parameter* params);
Tensor* linear_forward(LinearLayer* layer, Tensor* input);

// 创建卷积层
Conv2DLayer* create_conv2d_layer(int in_channels, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w,
                                int padding_h, int padding_w, int group, int stream);
void conv2d_reset_buffer(Conv2DLayer* layer);
Tensor* conv2d_forward(Conv2DLayer *layer, Tensor* input);
Parameter* conv2d_load_params(Conv2DLayer* layer, Parameter *params);
void free_conv2d_layer(Conv2DLayer *layer);

GRULayer* create_gru_layer(int input_size, int hidden_size);
void free_gru_layer(GRULayer* layer);
Parameter* gru_load_params(GRULayer* layer, Parameter* params);
Tensor* gru_forward(GRULayer* layer, Tensor* input, Tensor* hidden_state);

LSTMLayer* create_lstm_layer(int input_size, int hidden_size);
void free_lstm_layer(LSTMLayer* layer);
Parameter* lstm_load_params(LSTMLayer* layer, Parameter* params);
Tensor* lstm_forward(LSTMLayer* layer, Tensor* input, Tensor* hidden_state, Tensor* cell_state);

BatchNormLayer* create_batchnorm_layer(int num_features, float epsilon);
void free_batchnorm_layer(BatchNormLayer* layer);
Parameter* batchnorm_load_params(BatchNormLayer *layer, Parameter *params);
Tensor* batchnorm_forward(BatchNormLayer* layer, Tensor* input);

ELULayer* create_elu_layer(float alpha);
void free_elu_layer(ELULayer *layer);
Tensor* elu_forward(ELULayer* layer, Tensor* input);

LeakyReLULayer* create_leaky_relu_layer(float negative_slope);
void free_leaky_relu_layer(LeakyReLULayer* layer);
Tensor* leaky_relu_forward(LeakyReLULayer* layer, Tensor* input);

SigmoidLayer* create_sigmoid_layer();
void free_sigmoid_layer(SigmoidLayer* layer);
Tensor* sigmoid_forward(SigmoidLayer* layer, Tensor* input);

DepthwiseConv2DLayer* create_depthwise_conv2d_layer(
    int in_channels, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int stream);
void depthwise_conv2d_reset_buffer(DepthwiseConv2DLayer* layer);
Parameter* depthwise_conv2d_load_params(DepthwiseConv2DLayer *layer, Parameter *params);
void free_depthwise_conv2d_layer(DepthwiseConv2DLayer *layer);
Tensor* depthwise_conv2d_forward(DepthwiseConv2DLayer *layer, Tensor *input);

SkipBlock* create_skipblock(int hidden_channels);
void free_skipblock(SkipBlock *block);
Parameter* skipblock_load_params(SkipBlock* block, Parameter* params);
Tensor* skipblock_forward(SkipBlock* block, Tensor* encoder_feats, Tensor* decoder_feats);

SubPixelConv* create_subpixelconv(int in_channels, int out_channels);
void free_subpixelconv(SubPixelConv *block);
Parameter* subpixelconv_load_params(SubPixelConv* block, Parameter* params);
Tensor* subpixelconv_forward(SubPixelConv* block, Tensor* input);

ResidualBlock* create_residualblock(int hidden_channels, int stream);
void residualblock_reset_buffer(ResidualBlock* block);
void free_residualblock(ResidualBlock* block);
Parameter* residualblock_load_params(ResidualBlock* block, Parameter* params);
Tensor* residualblock_forward(ResidualBlock* block, Tensor* input);

#endif