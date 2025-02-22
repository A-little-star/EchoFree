/* 本文件记录所有测试代码 */ 
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

// 测试整个模型
// int main() {
//     // 非流式推理代码
//     // *** stage 1: 读取文件 ***
//     const char *filename_mic = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav";
//     const char *filename_ref = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav";
//     const char *filename_e = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/e.wav";
//     const char *filename_y = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/y.wav";
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *mic_ = read_wav_file(filename_mic, &num_samples, &sample_rate);
//     float *ref_ = read_wav_file(filename_ref, &num_samples, &sample_rate);

//     float *e_ = (float*)calloc(num_samples, sizeof(float));
//     float *y_ = (float*)calloc(num_samples, sizeof(float));

//     printf("stage 1 executed successfully.\n");

//     // *** stage 2: 线性滤波与特征提取 ***
//     pfdkf(ref_, mic_, e_, y_, num_samples);

//     wav_norm(mic_, num_samples);
//     wav_norm(ref_, num_samples);
//     wav_norm(e_, num_samples);
//     wav_norm(y_, num_samples);

//     Tensor *mic = create_tensor((int[]){num_samples}, 1);
//     Tensor *y = create_tensor((int[]){num_samples}, 1);
//     init_tensor(mic, mic_);
//     init_tensor(y, y_);

//     int length = num_samples / FRAME_SIZE;
//     Tensor* cspecs_mic = create_tensor((int[]){2, length, FREQ_SIZE}, 3);
//     Tensor* cspecs_y = create_tensor((int[]){2, length, FREQ_SIZE}, 3);
//     Tensor* features_mic = create_tensor((int[]){1, length, NB_FEATURES}, 3);
//     Tensor* features_y = create_tensor((int[]){1, length, NB_FEATURES}, 3);
//     feature_extract(mic, cspecs_mic, features_mic);
//     feature_extract(y, cspecs_y, features_y);
//     printf("stage 2 executed successfully.\n");

//     // *** stage 3: 读取模型参数并解析 ***

//     const char *cpt = "/home/node25_tmpdata/xcli/percepnet/c_aec/model_weights.json";
//     ModelStateDict *sd = parse_json_to_parameters(cpt);
//     printf("stage 3 executed successfully.\n");

//     // *** stage 4: 构建模型并加载模型参数 ***

//     RNNVQE *model = create_rnnvqe();
//     Parameter *params = sd->params;
//     rnnvqe_load_params(model, sd);
//     free_model_state_dict(sd);
//     printf("stage 4 executed successfully.\n");

//     // *** stage 5: 模型推理 ***
//     print_tensor_shape(features_mic, "input");
//     Tensor *gains = rnnvqe_forward(model, features_mic, features_y);
//     print_tensor_shape(gains, "gains");

//     float* out_wav = (float*)malloc(num_samples*sizeof(float));
//     post_process(cspecs_mic, gains, out_wav);

//     wav_invnorm(out_wav, num_samples);
//     printf("stage 5 executed successfully.\n");

//     // *** stage 6: 写入文件 ***
//     const char *filename_out = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/output_pfdkf256.wav";
//     if (write_wav_file(filename_out, out_wav, num_samples, sample_rate, 1, 16) == 0) {
//         printf("e WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }
//     printf("stage 6 executed successfully.\n");

//     return 0;
// }

// 测试流式推理
// int main() {
//     // 流式推理代码
//     // *** stage 1: 读取文件 ***
//     const char *filename_mic = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav";
//     const char *filename_ref = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav";
//     const char *filename_e = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/e.wav";
//     const char *filename_y = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/y.wav";
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *mic_ = read_wav_file(filename_mic, &num_samples, &sample_rate);
//     float *ref_ = read_wav_file(filename_ref, &num_samples, &sample_rate);

//     float *e_ = (float*)calloc(num_samples, sizeof(float));
//     float *y_ = (float*)calloc(num_samples, sizeof(float));

//     printf("stage 1 executed successfully.\n");

//     // *** stage 2: 读取模型参数并解析 ***

//     const char *cpt = "/home/node25_tmpdata/xcli/percepnet/c_aec/model_weights.json";
//     ModelStateDict *sd = parse_json_to_parameters(cpt);
//     printf("stage 2 executed successfully.\n");

//     // *** stage 3: 构建模型并加载模型参数 ***

//     RNNVQE *model = create_rnnvqe(1);  // 输入参数为stream，为1是流式推理，为0时非流式整句推理
//     Parameter *params = sd->params;
//     rnnvqe_load_params(model, sd);
//     free_model_state_dict(sd);
//     printf("stage 3 executed successfully.\n");

//     // *** stage 4: 模型推理 ***

//     int N = 10;
//     int M = 256;
//     float A = 0.999;
//     float P_initial = 10.0;

//     PFDKF *filter = init_pfdkf(N, M, A, P_initial);

//     Tensor* hidden_state = create_tensor((int[]){model->hidden_dim}, 1);
//     for (int i = 0; i < hidden_state->size; i ++ ) hidden_state->data[i] = 0;

//     DenoiseState* st_mic = rnnoise_create();
//     DenoiseState* st_y = rnnoise_create();
//     DenoiseState* st_out = rnnoise_create();

//     int win_len = WINDOW_SIZE;
//     int hop_len = FRAME_SIZE;
//     Tensor *mic = create_tensor((int[]){win_len}, 1);
//     Tensor *y = create_tensor((int[]){win_len}, 1);
//     Tensor* cspecs_mic = create_tensor((int[]){2, 1, FREQ_SIZE}, 3);
//     Tensor* cspecs_y = create_tensor((int[]){2, 1, FREQ_SIZE}, 3);
//     Tensor* features_mic = create_tensor((int[]){1, 1, NB_FEATURES}, 3);
//     Tensor* features_y = create_tensor((int[]){1, 1, NB_FEATURES}, 3);
//     float* out_wav = (float*)malloc(num_samples*sizeof(float));
//     for (int s = 0, num_frame = 0; s + win_len < num_samples; s += hop_len, num_frame ++ ) {
//         if (s == 0) {
//             // 先把第一个256的线性滤波做好
//             filt(filter, ref_, mic_, e_, y_);
//             update(filter);
//         }
//         float *e_n = (float*)malloc(M * sizeof(float));
//         float *y_n = (float*)malloc(M * sizeof(float));
//         filt(filter, ref_+s+hop_len, mic_+s+hop_len, e_+s+hop_len, y_+s+hop_len); // 注意线性滤波是先ref后mic，踩了很多次坑了
//         update(filter);
//         if (s == 0) {
//             wav_norm(mic_, hop_len);
//             wav_norm(y_, hop_len);
//         }
//         wav_norm(mic_+s+hop_len, hop_len);
//         wav_norm(y_+s+hop_len, hop_len);
//         init_tensor(mic, mic_ + s);
//         init_tensor(y, y_ + s);
//         feature_extract_frame(mic, cspecs_mic, features_mic, st_mic);
//         feature_extract_frame(y, cspecs_y, features_y, st_y);
//         Tensor* gains_frame = rnnvqe_forward(model, features_mic, features_y, hidden_state);
//         post_process_frame(cspecs_mic, gains_frame, out_wav + s, st_out);
//         wav_invnorm(out_wav+s, hop_len);
//     }
//     rnnvqe_reset_buffer(model);

//     printf("stage 4 executed successfully.\n");

//     // *** stage 5: 写入文件 ***
//     const char *filename_out = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/out_stream.wav";
//     if (write_wav_file(filename_out, out_wav, num_samples, sample_rate, 1, 16) == 0) {
//         printf("e WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }
//     printf("stage 5 executed successfully.\n");

//     return 0;
// }