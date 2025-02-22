#ifndef MATRIX_OP_H
#define MATRIX_OP_H

// 张量的结构体定义
// 目前只支持float32类型
typedef struct {
    float* data;            // 存储数据的指针
    int* shape;             // 形状数组
    int ndim;               // Tensor的维度
    int size;               // Tensor的总元素个数
} Tensor;

// 创建张量
// Tensor create_tensor(int c, int t, int f);
Tensor* create_tensor(int* shape, int ndim);
// 删除张量
void delete_tensor(Tensor *tensor);
// 初始化张量数据
void init_tensor(Tensor* tensor, float* values);
// 打印张量的形状
void print_tensor_shape(Tensor* tensor, const char *name);
// 打印张量数据
void print_tensor(const Tensor* tensor);
// 修改Tensor某位置的值
void tensor_set(Tensor* tensor, const int* indices, float value);
// 获取Tensor某位置的值
float tensor_get(const Tensor* tensor, const int* indices);
// 矩阵切片
Tensor* tensor_slice(Tensor* tensor, int* start_indices, int* end_indices);
// squeeze()操作，将第dim维度折叠
Tensor* tensor_squeeze(Tensor* tensor, int dim);
// concat拼接操作
Tensor* concatenate(Tensor* tensor1, Tensor* tensor2, int dim);
// permute操作
Tensor* permute(const Tensor* input, const int* permute_order);
// reshape操作
Tensor* reshape(const Tensor* input, const int* new_shape, int new_ndim);
// pad操作 pad: pad 2 dimention: [pad_bottom, pad_top, pad_left, pad_right]
Tensor* tensor_pad(Tensor* input, int* pad);

Tensor *tensor_add(Tensor* a, Tensor *b);
Tensor *tensor_mul(Tensor* a, Tensor *b);

#endif