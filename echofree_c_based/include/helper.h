#ifndef HELPER_H
#define HELPER_H

#include "model.h"
// 替换字符串中的 "_mic" 为 "_lpb"
char* replace_substring(const char* input, const char* target, const char* replacement);
// 提取文件名并去除 "_mic" 后缀的函数
char* extract_filename(const char* path);
// 拼接输出路径的函数
char* create_output_path(const char* out_dir, const char* filename);
// 推理一条音频
int infer_one_wav(EchoFree* model, const char* filename_mic, const char* filename_ref, const char* filename_out, int stream);

#endif