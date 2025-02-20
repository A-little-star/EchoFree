#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include "../include/tensor.h"

// 创建一个新的张量
Tensor* create_tensor(int* shape, int ndim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));

    // 复制形状并计算步长
    tensor->size = 1;
    for (int i = 0; i < ndim; ++i) {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    // 分配内存
    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    return tensor;
}

// 释放张量所占的内存空间
void delete_tensor(Tensor *tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor);
    }
}

// 初始化张量数据
void init_tensor(Tensor* tensor, float* values) {
    if (tensor == NULL || tensor->data == NULL || values == NULL) {
        fprintf(stderr, "Invalid tensor or values\n");
        return;
    }
    int total_elements = tensor->size;
    for (int i = 0; i < total_elements; i++) {
        tensor->data[i] = values[i];
    }
}

// 打印 Tensor 的形状
void print_tensor_shape(Tensor* tensor, const char* name) {
    printf("%s shape: (", name);
    for (int i = 0; i < tensor->ndim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf(")\n");
}

// 打印张量数据
// 打印Tensor递归函数
void print_tensor_recursive(const Tensor* tensor, int depth, int offset) {
    if (depth == tensor->ndim) {
        // 到达最内层，打印单个元素
        printf("%f", tensor->data[offset]);
        return;
    }

    int depth_size = 1;
    for (int i = 0; i <= depth; i ++ )
        depth_size *= tensor->shape[i];

    // 打印当前维度
    printf("[");
    for (int i = 0; i < tensor->shape[depth]; ++i) {
        if (i > 0) {
            printf(", ");
        }
        // 递归打印下一维度
        print_tensor_recursive(tensor, depth + 1, offset + i * (tensor->size / depth_size));
    }
    printf("]");
}

// 打印Tensor的主函数
void print_tensor(const Tensor* tensor) {
    if (!tensor || !tensor->data || !tensor->shape) {
        printf("Error: Invalid Tensor.\n");
        return;
    }

    print_tensor_recursive(tensor, 0, 0);
    printf("\n");
}
// void print_tensor(const Tensor* tensor) {
//     if (tensor == NULL || tensor->data == NULL) {
//         fprintf(stderr, "Invalid tensor\n");
//         return;
//     }
//     for (int c = 0; c < tensor->C; c++) {
//         printf("Channel %d:\n", c + 1);
//         for (int t = 0; t < tensor->T; t++) {
//             for (int f = 0; f < tensor->F; f++) {
//                 printf("%.2f ", tensor->data[c * tensor->T * tensor->F + t * tensor->F + f]);
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }
// }

// 计算多维索引对应的一维偏移
int compute_offset(const Tensor* tensor, const int* indices) {
    int offset = 0;
    int stride = tensor->size;

    for (int i = 0; i < tensor->ndim; ++i) {
        stride /= tensor->shape[i];  // 计算当前维度的步长
        if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
            printf("Error: Index out of bounds at dimension %d.\n", i);
            return -1;  // 返回无效偏移
        }
        offset += indices[i] * stride;
    }
    return offset;
}

// 修改Tensor某位置的值
void tensor_set(Tensor* tensor, const int* indices, float value) {
    int offset = compute_offset(tensor, indices);
    if (offset == -1) {
        return;  // 出错直接返回
    }
    tensor->data[offset] = value;
}

// 获取Tensor某位置的值
float tensor_get(const Tensor* tensor, const int* indices) {
    int offset = compute_offset(tensor, indices);
    if (offset == -1) {
        return 0.0f;  // 出错返回默认值
    }
    return tensor->data[offset];
}

// 获取切片后的Tensor
Tensor* tensor_slice(Tensor* tensor, int* start_indices, int* end_indices) {
    // 创建一个新的Tensor来存储切片结果
    Tensor* slice = (Tensor*)malloc(sizeof(Tensor));
    slice->ndim = tensor->ndim;
    slice->shape = (int*)malloc(slice->ndim * sizeof(int));

    for (int i = 0; i < tensor->ndim; i ++ ) {
        assert(end_indices[i] <= tensor->shape[i]);
    }
    
    int size = 1;
    for (int i = 0; i < slice->ndim; i++) {
        slice->shape[i] = end_indices[i] - start_indices[i];
        size *= slice->shape[i];
    }
    
    slice->size = size;
    slice->data = (float*)malloc(size * sizeof(float));
    
    // 复制切片数据
    int* indices = (int*)malloc(slice->ndim * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        // 计算当前索引
        int temp_offset = i;
        for (int j = slice->ndim - 1; j >= 0; j--) {
            indices[j] = temp_offset % slice->shape[j] + start_indices[j];
            temp_offset /= slice->shape[j];
        }
        offset = compute_offset(tensor, indices);
        slice->data[i] = tensor->data[offset];
    }

    free(indices);
    return slice;
}

// 获取Tensor的元素总数
int get_tensor_size(Tensor* tensor) {
    int size = 1;
    for (int i = 0; i < tensor->ndim; i++) {
        size *= tensor->shape[i];
    }
    return size;
}

// Tensor的squeeze操作，只移除指定dim维度的大小为1的维度
Tensor* tensor_squeeze(Tensor* tensor, int dim) {
    // 检查dim维度是否为1
    if (tensor->shape[dim] != 1) {
        printf("Error: The dimension %d is not of size 1!\n", dim);
        return tensor;  // 如果该维度不是1，直接返回原Tensor
    }
    // 创建一个新的Tensor来存储squeeze后的结果
    Tensor* squeezed_tensor = (Tensor*)malloc(sizeof(Tensor));
    // 新的Tensor形状数组
    squeezed_tensor->ndim = tensor->ndim - 1;
    squeezed_tensor->shape = (int*)malloc(squeezed_tensor->ndim * sizeof(int));
    // 复制新的维度形状，去掉dim维度
    int j = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        if (i != dim) {
            squeezed_tensor->shape[j++] = tensor->shape[i];
        }
    }
    // 计算新的Tensor大小
    squeezed_tensor->size = get_tensor_size(squeezed_tensor);
    // 分配新的内存空间
    squeezed_tensor->data = (float*)malloc(squeezed_tensor->size * sizeof(float));
    // 使用memcpy进行数据拷贝
    memcpy(squeezed_tensor->data, tensor->data, squeezed_tensor->size * sizeof(float));
    return squeezed_tensor;
}

// concatenate 函数实现
Tensor* concatenate(Tensor* tensor1, Tensor* tensor2, int dim) {
    // 检查维度是否匹配
    if (tensor1->ndim != tensor2->ndim) {
        fprintf(stderr, "Tensors must have the same number of dimensions.\n");
        return NULL;
    }

    for (int i = 0; i < tensor1->ndim; i++) {
        if (i != dim && tensor1->shape[i] != tensor2->shape[i]) {
            fprintf(stderr, "Tensors must have the same shape except for the concatenation dimension.\n");
            return NULL;
        }
    }

    // 计算拼接后的新形状
    int* new_shape = (int*)malloc(tensor1->ndim * sizeof(int));
    for (int i = 0; i < tensor1->ndim; i++) {
        if (i == dim) {
            new_shape[i] = tensor1->shape[i] + tensor2->shape[i]; // 拼接维度的大小
        } else {
            new_shape[i] = tensor1->shape[i]; // 其他维度保持不变
        }
    }

    // 创建拼接后的输出 Tensor
    Tensor* result = create_tensor(new_shape, tensor1->ndim);
    free(new_shape);

    // 拼接数据
    int dim_before_cat = 1, t1_dim_after_cat = 1, t2_dim_after_cat = 1;
    for (int i = 0; i < dim; i ++ ) {
        dim_before_cat *= tensor1->shape[i];
    }
    for (int i = dim; i < tensor1->ndim; i ++ ) {
        t1_dim_after_cat *= tensor1->shape[i];
        t2_dim_after_cat *= tensor2->shape[i];
    }

    int off1 = 0, off2 = 0, offr = 0;
    for (int i = 0; i < dim_before_cat; i ++ ) {
        memcpy(result->data + offr, tensor1->data + off1, t1_dim_after_cat * sizeof(float));
        off1 += t1_dim_after_cat;
        offr += t1_dim_after_cat;
        memcpy(result->data + offr, tensor2->data + off2, t2_dim_after_cat * sizeof(float));
        off2 += t2_dim_after_cat;
        offr += t2_dim_after_cat;
    }

    return result;
}

// permute函数
Tensor* permute(const Tensor* input, const int* permute_order) {
    // 1. 检查输入是否合法
    int ndim = input->ndim;
    for (int i = 0; i < ndim; ++i) {
        if (permute_order[i] < 0 || permute_order[i] >= ndim) {
            fprintf(stderr, "Invalid permute order.\n");
            return NULL;
        }
    }

    // 2. 创建新的shape
    int* new_shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; ++i) {
        new_shape[i] = input->shape[permute_order[i]];
    }
    // 3. 创建新的Tensor
    Tensor* output = (Tensor*)malloc(sizeof(Tensor));
    output->ndim = ndim;
    output->size = input->size;
    output->shape = new_shape;
    output->data = (float*)malloc(output->size * sizeof(float));

    // 4. 对数据进行重新排列
    int* indices = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < output->size; ++i) {
        // 计算目标Tensor的多维索引
        int remaining = i;
        for (int j = ndim - 1; j >= 0; --j) {
            indices[j] = remaining % new_shape[j];
            remaining /= new_shape[j];
        }
        // 根据permute_order找到原始Tensor的多维索引
        int* original_indices = (int*)malloc(ndim * sizeof(int));
        for (int j = 0; j < ndim; ++j) {
            original_indices[permute_order[j]] = indices[j];
        }

        // 找到原始Tensor的线性索引，并复制数据
        int original_index = compute_offset(input, original_indices);
        output->data[i] = input->data[original_index];

        free(original_indices);
    }

    free(indices);
    return output;
}

// 检查新的shape是否合法
int is_valid_shape(const Tensor* input, const int* new_shape, int new_ndim) {
    int new_size = 1;
    for (int i = 0; i < new_ndim; ++i) {
        if (new_shape[i] <= 0) {
            return false; // 所有维度必须为正整数
        }
        new_size *= new_shape[i];
    }
    return new_size == input->size; // 确保元素总数匹配
}

// reshape函数
Tensor* reshape(const Tensor* input, const int* new_shape, int new_ndim) {
    // 检查新形状是否合法
    if (!is_valid_shape(input, new_shape, new_ndim)) {
        fprintf(stderr, "Invalid shape: Total elements do not match.\n");
        assert(0);
        return NULL;
    }

    // 创建新的Tensor
    Tensor* output = (Tensor*)malloc(sizeof(Tensor));
    output->ndim = new_ndim;
    output->size = input->size;
    output->shape = (int*)malloc(new_ndim * sizeof(int));
    for (int i = 0; i < new_ndim; ++i) {
        output->shape[i] = new_shape[i];
    }

    // 数据共享（浅复制）
    // output->data = input->data; // reshape不改变数据存储方式
    // 深复制
    output->data = (float*)malloc(output->size * sizeof(float));
    for (int i = 0; i < input->size; i ++ )
        output->data[i] = input->data[i];

    return output;
}

// Tensor的Pad函数
Tensor* tensor_pad(Tensor* input, int* pad) {
    int ndim = input->ndim;
    // assert(ndim * 2 == sizeof(pad) / sizeof(pad[0]));

    // 计算新形状
    int* new_shape = (int*)malloc(ndim * sizeof(int));
    int new_size = 1;
    for (int i = 0; i < ndim; i++) {
        new_shape[i] = input->shape[i] + pad[2 * i] + pad[2 * i + 1];
        new_size *= new_shape[i];
    }

    // 创建新Tensor
    Tensor* output = create_tensor(new_shape, ndim);
    memset(output->data, 0, output->size * sizeof(float));

    // 填充数据
    int* old_indices = (int*)calloc(ndim, sizeof(int));
    int* new_indices = (int*)calloc(ndim, sizeof(int));
    for (int i = 0; i < input->size; i++) {
        // 计算旧Tensor中的多维索引
        int remaining = i;
        for (int j = ndim - 1; j >= 0; j--) {
            old_indices[j] = remaining % input->shape[j];
            remaining /= input->shape[j];
        }

        // 计算新Tensor中的多维索引
        for (int j = 0; j < ndim; j++) {
            new_indices[j] = old_indices[j] + pad[2 * j];
        }

        // 计算新Tensor中的线性索引
        int new_linear_index = 0;
        int stride = 1;
        for (int j = ndim - 1; j >= 0; j--) {
            new_linear_index += new_indices[j] * stride;
            stride *= new_shape[j];
        }

        // 复制数据
        output->data[new_linear_index] = input->data[i];
    }

    // 释放临时内存
    free(old_indices);
    free(new_indices);
    free(new_shape);

    return output;
}

//
Tensor *tensor_add(Tensor* a, Tensor *b) {
    assert(a->ndim == b->ndim);
    for (int i = 0; i < a->ndim; i ++ ) {
        assert(a->shape[i] == b->shape[i]);
    }
    Tensor *out = create_tensor(a->shape, a->ndim);
    for (int i = 0; i < a->size; i ++ )
        out->data[i] = a->data[i] + b->data[i];
    return out;
}

Tensor *tensor_mul(Tensor* a, Tensor *b) {
    assert(a->ndim == b->ndim);
    for (int i = 0; i < a->ndim; i ++ ) {
        assert(a->shape[i] == b->shape[i]);
    }
    Tensor *out = create_tensor(a->shape, a->ndim);
    for (int i = 0; i < a->size; i ++ )
        out->data[i] = a->data[i] * b->data[i];
    return out;
}