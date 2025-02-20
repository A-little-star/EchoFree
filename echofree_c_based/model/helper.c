#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/wavreader.h"
#include "../include/stft.h"
#include "../include/typedef.h"
#include "../include/pfdkf.h"
#include "../include/parser.h"
#include "../include/model.h"
#include "../include/rnnoise.h"

// 替换字符串中的 "_mic" 为 "_lpb"
char* replace_substring(const char* input, const char* target, const char* replacement) {
    // 获取原字符串和目标字符串的长度
    size_t input_len = strlen(input);
    size_t target_len = strlen(target);
    size_t replacement_len = strlen(replacement);

    // 如果目标字符串为空，直接返回原字符串的副本
    if (target_len == 0) {
        return strdup(input);
    }

    // 计算新字符串的最大可能长度
    size_t max_new_len = input_len + (replacement_len - target_len) * 10; // 假设最多10次替换
    char* result = (char*)malloc(max_new_len + 1);
    if (!result) {
        perror("内存分配失败");
        exit(EXIT_FAILURE);
    }

    // 遍历输入字符串并进行替换
    const char* current = input;
    char* dest = result;

    while (*current) {
        const char* match = strstr(current, target); // 查找目标字符串的位置
        if (match) {
            // 复制目标字符串前的部分
            size_t prefix_len = match - current;
            memcpy(dest, current, prefix_len);
            dest += prefix_len;

            // 复制替换字符串
            memcpy(dest, replacement, replacement_len);
            dest += replacement_len;

            // 跳过目标字符串部分
            current = match + target_len;
        } else {
            // 如果没有匹配，复制剩余部分并退出循环
            strcpy(dest, current);
            break;
        }
    }

    return result;
}

// 提取文件名并去除 "_mic" 后缀的函数
char* extract_filename(const char* path) {
    // 找到路径中最后一个 '/' 的位置
    const char* last_slash = strrchr(path, '/');
    const char* filename = last_slash ? last_slash + 1 : path; // 提取文件名部分

    // 找到文件名中最后一个 '.' 的位置（去掉扩展名）
    char* dot = strrchr(filename, '.');
    size_t len = dot ? (size_t)(dot - filename) : strlen(filename);

    // 动态分配内存存储中间结果
    char* extracted = (char*)malloc(len + 1);
    if (!extracted) {
        perror("内存分配失败");
        exit(EXIT_FAILURE);
    }

    // 复制文件名部分到新字符串
    strncpy(extracted, filename, len);
    extracted[len] = '\0';

    // 查找并去除 "_mic" 后缀
    char* mic_pos = strstr(extracted, "_mic");
    if (mic_pos) {
        *mic_pos = '\0'; // 截断字符串
    }

    return extracted;
}

// 拼接输出路径的函数
char* create_output_path(const char* out_dir, const char* filename) {
    size_t out_dir_len = strlen(out_dir);
    size_t filename_len = strlen(filename);
    size_t ext_len = 4; // ".wav"

    // 分配足够的内存存储拼接结果
    char* output_path = (char*)malloc(out_dir_len + filename_len + ext_len + 2);
    if (!output_path) {
        perror("内存分配失败");
        exit(EXIT_FAILURE);
    }

    // 拼接字符串
    sprintf(output_path, "%s/%s.wav", out_dir, filename);

    return output_path;
}

int infer_one_wav(EchoFree* model, const char* filename_mic, const char* filename_ref, const char* filename_out, int stream) {
    int num_samples;                        // 样本数
    int sample_rate;                        // 采样率
    float *mic_ = read_wav_file(filename_mic, &num_samples, &sample_rate);
    float *ref_ = read_wav_file(filename_ref, &num_samples, &sample_rate);

    float *e_ = (float*)calloc(num_samples, sizeof(float));
    float *y_ = (float*)calloc(num_samples, sizeof(float));

    Tensor* hidden_state = create_tensor((int[]){model->hidden_dim}, 1);
    for (int i = 0; i < hidden_state->size; i ++ ) hidden_state->data[i] = 0;
    Tensor* cell_state = create_tensor((int[]){model->hidden_dim}, 1);
    for (int i = 0; i < cell_state->size; i ++ ) cell_state->data[i] = 0;

    DenoiseState* st_mic = rnnoise_create();
    DenoiseState* st_y = rnnoise_create();
    DenoiseState* st_out = rnnoise_create();

    int N = 10;
    int M = 256;
    float A = 0.999;
    float P_initial = 10.0;
    PFDKF *filter = init_pfdkf(N, M, A, P_initial);

    float* out_wav = (float*)malloc(num_samples*sizeof(float));
    if (stream) {
        int win_len = WINDOW_SIZE;
        int hop_len = FRAME_SIZE;
        Tensor *mic = create_tensor((int[]){win_len}, 1);
        Tensor *y = create_tensor((int[]){win_len}, 1);
        Tensor* cspecs_mic = create_tensor((int[]){2, 1, FREQ_SIZE}, 3);
        Tensor* cspecs_y = create_tensor((int[]){2, 1, FREQ_SIZE}, 3);
        Tensor* features_mic = create_tensor((int[]){1, 1, NB_FEATURES}, 3);
        Tensor* features_y = create_tensor((int[]){1, 1, NB_FEATURES}, 3);
        for (int s = 0, num_frame = 0; s + win_len < num_samples; s += hop_len, num_frame ++ ) {
            if (s == 0) {
                // 先把第一个256的线性滤波做好
                filt(filter, ref_, mic_, e_, y_);
                update(filter);
            }
            float *e_n = (float*)malloc(filter->M * sizeof(float));
            float *y_n = (float*)malloc(filter->M * sizeof(float));
            filt(filter, ref_+s+hop_len, mic_+s+hop_len, e_+s+hop_len, y_+s+hop_len); // 注意线性滤波是先ref后mic，踩了很多次坑了
            update(filter);
            if (s == 0) {
                wav_norm(mic_, hop_len);
                wav_norm(y_, hop_len);
            }
            wav_norm(mic_+s+hop_len, hop_len);
            wav_norm(y_+s+hop_len, hop_len);
            init_tensor(mic, mic_ + s);
            init_tensor(y, y_ + s);
            feature_extract_frame(mic, cspecs_mic, features_mic, st_mic);
            feature_extract_frame(y, cspecs_y, features_y, st_y);
            Tensor* gains_frame = echofree_forward(model, features_mic, features_y, hidden_state, cell_state);
            post_process_frame(cspecs_mic, gains_frame, out_wav + s, st_out);
            wav_invnorm(out_wav+s, hop_len);
        }
        echofree_reset_buffer(model);
        delete_tensor(mic);
        delete_tensor(y);
        delete_tensor(cspecs_mic);
        delete_tensor(cspecs_y);
        delete_tensor(features_mic);
        delete_tensor(features_y);
    }
    else if (!stream) {
        pfdkf(ref_, mic_, e_, y_, num_samples);
        wav_norm(mic_, num_samples);
        wav_norm(ref_, num_samples);
        wav_norm(e_, num_samples);
        wav_norm(y_, num_samples);
        Tensor *mic = create_tensor((int[]){num_samples}, 1);
        Tensor *y = create_tensor((int[]){num_samples}, 1);
        init_tensor(mic, mic_);
        init_tensor(y, y_);
        int length = num_samples / FRAME_SIZE;
        Tensor* cspecs_mic = create_tensor((int[]){2, length, FREQ_SIZE}, 3);
        Tensor* cspecs_y = create_tensor((int[]){2, length, FREQ_SIZE}, 3);
        Tensor* features_mic = create_tensor((int[]){1, length, NB_FEATURES}, 3);
        Tensor* features_y = create_tensor((int[]){1, length, NB_FEATURES}, 3);
        feature_extract(mic, cspecs_mic, features_mic);
        feature_extract(y, cspecs_y, features_y);
        Tensor *gains = echofree_forward(model, features_mic, features_y, hidden_state, cell_state);
        post_process(cspecs_mic, gains, out_wav);
        wav_invnorm(out_wav, num_samples);
        delete_tensor(mic);
        delete_tensor(y);
        delete_tensor(cspecs_mic);
        delete_tensor(cspecs_y);
        delete_tensor(features_mic);
        delete_tensor(features_y);
    }
    
    delete_tensor(hidden_state);
    delete_tensor(cell_state);
    rnnoise_destroy(st_mic);
    rnnoise_destroy(st_y);
    rnnoise_destroy(st_out);
    free_pfdkf(filter);

    if (write_wav_file(filename_out, out_wav, num_samples, sample_rate, 1, 16) == 0) {
        printf("e WAV file written successfully.\n");
    } else {
        printf("Failed to write WAV file.\n");
    }
    free(mic_);
    free(ref_);
    free(e_);
    free(y_);
    free(out_wav);
    return 0;
}

// 测试wavreader.c
// int main() {
//     const char *filename = "example.wav";    // 替换为实际文件名
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *audio_data = read_wav_file(filename, &num_samples, &sample_rate);
//     // wav_norm(audio_data, num_samples);

//     if (write_wav_file("output.wav", audio_data, num_samples, sample_rate, 1, 16) == 0) {
//         printf("WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }

//     free(audio_data);
//     return 0;
// }

// 测试线性滤波
// int main() {
//     // Example usage of PFDKF
//     const char *filename_mic = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav";
//     const char *filename_ref = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav";
//     const char *filename_e = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/e.wav";
//     const char *filename_y = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/y.wav";
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *mic = read_wav_file(filename_mic, &num_samples, &sample_rate);
//     float *ref = read_wav_file(filename_ref, &num_samples, &sample_rate);

//     float *e = (float*)calloc(num_samples, sizeof(float));
//     float *y = (float*)calloc(num_samples, sizeof(float));
//     pfdkf(ref, mic, e, y, num_samples);

//     if (write_wav_file(filename_e, e, num_samples, sample_rate, 1, 16) == 0) {
//         printf("e WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }
//     if (write_wav_file(filename_y, y, num_samples, sample_rate, 1, 16) == 0) {
//         printf("y WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }

//     return 0;
// }

// 测试加载参数
// int main() {
//     const char *filename = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_txt/test_dict.txt";
//     Parameter *params = (Parameter*)malloc(MAX_PARAMS * sizeof(Parameter));
//     parse_ckpt(filename, params);
//     return EXIT_SUCCESS;
// }

// 测试矩阵切片
// int main() {
//     // 输入参数
//     int in_channels = 3, in_h = 4, in_w = 4;
//     float input_data[3 * 4 * 4] = {
//         1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,  13, 14, 15, 16, // 通道 1
//         1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,    1, 1, 1, 1, // 通道 2
//         0, 1, 0, 1,   0, 1, 0, 1,   0, 1, 0, 1,    0, 1, 0, 1  // 通道 3
//     };
//     // Tensor *input = create_tensor((int[]){3, 4, 4}, 3);
//     Tensor *input = create_tensor((int[]){3, 4, 4}, 3);
//     init_tensor(input, input_data);
//     print_tensor(input);
//     printf("\n");

//     Tensor *output = tensor_squeeze(input, 0);
//     print_tensor(output);

//     // Tensor *output = tensor_slice(input, (int[]){0, 0, 0}, (int[]){2, 4, 4});
//     // print_tensor(output);

//     return 0;
// }

// 测试concatenate函数
// int main() {
//     // 输入参数
//     int in_channels = 3, in_h = 4, in_w = 4;
//     float input_data1[3 * 2 * 4] = {
//         1, 2, 3, 4,   5, 6, 7, 8,  // 通道 1
//         1, 1, 1, 1,   1, 1, 1, 1,  // 通道 2
//         0, 1, 0, 1,   0, 1, 0, 1,  // 通道 3
//     };
//     float input_data2[1 * 2 * 4] = {
//         9, 10, 11, 12,  13, 14, 15, 16, // 通道 1
//     };
//     // Tensor *input = create_tensor((int[]){3, 4, 4}, 3);
//     Tensor *input1 = create_tensor((int[]){3, 2, 4}, 3);
//     init_tensor(input1, input_data1);
//     Tensor *input2 = create_tensor((int[]){1, 2, 4}, 3);
//     init_tensor(input2, input_data2);
//     print_tensor(input1);
//     printf("\n");
//     print_tensor(input2);
//     printf("\n");

//     Tensor *output = concatenate(input1, input2, 0);
//     print_tensor(output);
//     printf("\n");
//     return 0;
// }

// 测试GRU
// int main() {
//     Tensor *input = create_tensor((int[]){2, 2}, 2);
//     GRULayer *gru = create_gru_layer(2, 2);
//     Tensor *hidden_state = create_tensor((int[]){2}, 1);
//     Tensor *output = gru_forward(gru, input, hidden_state);
//     print_tensor_shape(output);
//     return 0;
// }

// 测试pad
// int main() {
//     int shape_in[2] = {3, 4};
//     Tensor *t = create_tensor(shape_in, 2);
//     print_tensor_shape(t);
//     printf("%d\n", t->size);
//     float data[] = {
//         1, 2, 3, 4,
//         4, 5, 6, 7,
//         8, 9, 10, 11
//     };
//     init_tensor(t, data);
//     print_tensor(t);
//     printf("\n");
//     int pad[4] = {1, 0, 1, 2};
//     Tensor* o = tensor_pad(t, pad);
//     print_tensor(o);
//     return 0;
// }

// 测试permute
// int main() {
//     int shape_in[2] = {3, 4};
//     Tensor *t = create_tensor(shape_in, 2);
//     print_tensor_shape(t);
//     printf("%d\n", t->size);
//     float data[] = {
//         1, 2, 3, 4,
//         4, 5, 6, 7,
//         8, 9, 10, 11
//     };
//     init_tensor(t, data);
//     print_tensor(t);

//     Tensor* o = permute(t, (int[]){1, 0});
//     print_tensor(o);
//     return 0;
// }

// 测试decoder
// int main() {
//     Tensor *input = create_tensor((int[]){8, 70, 55}, 3);
//     Tensor *mid = create_tensor((int[]){8, 70, 54}, 3);
//     DecoderBlock* dec1 = create_decoder_block(8, 1, 1, 1);
//     Tensor* output = decoderblock_forward(dec1, input, mid);
//     print_tensor_shape(output);
//     return 0;
// }

// 测试正逆变换
// int main() {
//     const char *filename = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav";
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *wav_ = read_wav_file(filename, &num_samples, &sample_rate);
//     wav_norm(wav_, num_samples);
//     Tensor *wav = create_tensor((int[]){num_samples}, 1);
//     init_tensor(wav, wav_);
//     int length = num_samples / FRAME_SIZE - 1;
//     Tensor* cspecs = create_tensor((int[]){2, length, FREQ_SIZE}, 3);
//     Tensor* features = create_tensor((int[]){1, length, NB_FEATURES}, 3);
//     feature_extract(wav, cspecs, features);

//     const char *filename_out = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/test.wav";
//     float* out_wav = istft(cspecs, num_samples);
//     wav_invnorm(out_wav, num_samples);
//     if (write_wav_file(filename_out, out_wav, num_samples, sample_rate, 1, 16) == 0) {
//         printf("e WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }

//     return 0;
// }