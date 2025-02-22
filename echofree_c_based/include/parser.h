#ifndef PARSER_H
#define PARSER_H

#define MAX_LINE_LENGTH 1024000   // 每行最大长度
#define MAX_PARAM_NAME 256     // 参数名的最大长度
#define MAX_PARAMS 1000

// 结构体用于存储参数名和对应数组
typedef struct {
    char name[MAX_PARAM_NAME];
    float *values;
    int size;
} Parameter;

typedef struct {
    Parameter *params;
    int size;
} ModelStateDict;

int parse_line(const char *line, Parameter *param);
void parse_ckpt(const char *filename, ModelStateDict *sd);
ModelStateDict *create_model_state_dict();
void free_model_state_dict(ModelStateDict *sd);

// 函数声明
ModelStateDict *parse_json_to_parameters(const char *filename);
void free_parameters(Parameter *parameters, int param_count);

#endif