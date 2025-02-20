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
#include "../include/helper.h"

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

//     const char *cpt = "/home/node25_tmpdata/xcli/percepnet/c_aec/checkpoint/model_weights_rnnvqe_last.json";
//     ModelStateDict *sd = parse_json_to_parameters(cpt);
//     printf("stage 3 executed successfully.\n");

//     // *** stage 4: 构建模型并加载模型参数 ***

//     RNNVQE *model = create_rnnvqe(0);
//     Parameter *params = sd->params;
//     rnnvqe_load_params(model, sd);
//     free_model_state_dict(sd);
//     printf("stage 4 executed successfully.\n");

//     // *** stage 5: 模型推理 ***
//     print_tensor_shape(features_mic, "input");
//     Tensor* hidden_state = create_tensor((int[]){model->hidden_dim}, 1);
//         for (int i = 0; i < hidden_state->size; i ++ ) hidden_state->data[i] = 0;
//     Tensor* cell_state = create_tensor((int[]){model->hidden_dim}, 1);
//         for (int i = 0; i < cell_state->size; i ++ ) cell_state->data[i] = 0;
//     Tensor *gains = rnnvqe_forward(model, features_mic, features_y, hidden_state, cell_state);
//     print_tensor_shape(gains, "gains");

//     float* out_wav = (float*)malloc(num_samples*sizeof(float));
//     post_process(cspecs_mic, gains, out_wav);

//     wav_invnorm(out_wav, num_samples);
//     printf("stage 5 executed successfully.\n");

//     // *** stage 6: 写入文件 ***
//     const char *filename_out = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/out_last.wav";
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

//     const char *cpt = "/home/node25_tmpdata/xcli/percepnet/c_aec/checkpoint/model_weights_rnnvqe_last.json";
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
//     Tensor* cell_state = create_tensor((int[]){model->hidden_dim}, 1);
//     for (int i = 0; i < cell_state->size; i ++ ) cell_state->data[i] = 0;

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
//         Tensor* gains_frame = rnnvqe_forward(model, features_mic, features_y, hidden_state, cell_state);
//         post_process_frame(cspecs_mic, gains_frame, out_wav + s, st_out);
//         wav_invnorm(out_wav+s, hop_len);
//     }
//     rnnvqe_reset_buffer(model);

//     printf("stage 4 executed successfully.\n");

//     // *** stage 5: 写入文件 ***
//     const char *filename_out = "/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/out_stream_last.wav";
//     if (write_wav_file(filename_out, out_wav, num_samples, sample_rate, 1, 16) == 0) {
//         printf("e WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }
//     printf("stage 5 executed successfully.\n");

//     return 0;
// }

// int infer_one_wav(RNNVQE* model, const char* filename_mic, const char* filename_ref, const char* filename_out) {
//     int num_samples;                        // 样本数
//     int sample_rate;                        // 采样率
//     float *mic_ = read_wav_file(filename_mic, &num_samples, &sample_rate);
//     float *ref_ = read_wav_file(filename_ref, &num_samples, &sample_rate);

//     float *e_ = (float*)calloc(num_samples, sizeof(float));
//     float *y_ = (float*)calloc(num_samples, sizeof(float));

//     Tensor* hidden_state = create_tensor((int[]){model->hidden_dim}, 1);
//     for (int i = 0; i < hidden_state->size; i ++ ) hidden_state->data[i] = 0;
//     Tensor* cell_state = create_tensor((int[]){model->hidden_dim}, 1);
//     for (int i = 0; i < cell_state->size; i ++ ) cell_state->data[i] = 0;

//     DenoiseState* st_mic = rnnoise_create();
//     DenoiseState* st_y = rnnoise_create();
//     DenoiseState* st_out = rnnoise_create();

//     int N = 10;
//     int M = 256;
//     float A = 0.999;
//     float P_initial = 10.0;
//     PFDKF *filter = init_pfdkf(N, M, A, P_initial);

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
//         float *e_n = (float*)malloc(filter->M * sizeof(float));
//         float *y_n = (float*)malloc(filter->M * sizeof(float));
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
//         Tensor* gains_frame = rnnvqe_forward(model, features_mic, features_y, hidden_state, cell_state);
//         post_process_frame(cspecs_mic, gains_frame, out_wav + s, st_out);
//         wav_invnorm(out_wav+s, hop_len);
//     }
//     rnnvqe_reset_buffer(model);
    
//     delete_tensor(hidden_state);
//     delete_tensor(cell_state);
//     rnnoise_destroy(st_mic);
//     rnnoise_destroy(st_y);
//     rnnoise_destroy(st_out);
//     delete_tensor(mic);
//     delete_tensor(y);
//     delete_tensor(cspecs_mic);
//     delete_tensor(cspecs_y);
//     delete_tensor(features_mic);
//     delete_tensor(features_y);
//     free_pfdkf(filter);

//     if (write_wav_file(filename_out, out_wav, num_samples, sample_rate, 1, 16) == 0) {
//         printf("e WAV file written successfully.\n");
//     } else {
//         printf("Failed to write WAV file.\n");
//     }
//     free(mic_);
//     free(ref_);
//     free(e_);
//     free(y_);
//     free(out_wav);
//     return 0;
// }

// 测试按lst文件逐条推理
int main() {
    const char *cpt = "./checkpoint/model_weights.json";
    ModelStateDict *sd = parse_json_to_parameters(cpt);

    int stream = 0;   // 是否流式推理，值为1时进行流式推理，值为0时进行非流式推理

    EchoFree *model = create_echofree(stream);  // 输入参数为stream，为1是流式推理，为0时非流式整句推理
    Parameter *params = sd->params;
    echofree_load_params(model, sd);
    free_model_state_dict(sd);

    const char* lst_file_path = "./goertek.lst";
    const char* out_dir = "./decode/paper";   // 最后不用加 "/"
    FILE* file = fopen(lst_file_path, "r");
    if (!file) {
        perror("无法打开文件");
        return;
    }

    char line[1024]; // 用于存储当前读取的行

    while (fgets(line, sizeof(line), file)) {
        // 去掉末尾的换行符或回车符
        size_t len = strlen(line);
        if (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[len - 1] = '\0';
        }
        char* filename_mic = line;
        char* filename_ref = replace_substring(filename_mic, "_mic", "_lpb");
        char* name = extract_filename(filename_mic);
        char* filename_out = create_output_path(out_dir, name);
        infer_one_wav(model, filename_mic, filename_ref, filename_out, stream);
    }
    free_echofree(model);

    return 0;
}