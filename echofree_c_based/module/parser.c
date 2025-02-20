#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../include/parser.h"

// 去除字符串前后空格的辅助函数
char *trim_whitespace(char *str) {
    char *end;
    while (isspace((unsigned char)*str)) str++; // 去除开头空格
    if (*str == 0) return str;                  // 全是空格

    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--; // 去除结尾空格

    *(end + 1) = '\0';
    return str;
}

// 解析一行并提取参数名和数组
int parse_line(const char *line, Parameter *param) {
    char *colon_pos = strchr(line, ':');
    if (!colon_pos) {
        fprintf(stderr, "Invalid line format: %s\n", line);
        return -1;
    }

    // 提取参数名
    int name_len = colon_pos - line;
    if (name_len >= MAX_PARAM_NAME) {
        fprintf(stderr, "Parameter name too long: %s\n", line);
        return -1;
    }
    strncpy(param->name, line, name_len);
    param->name[name_len] = '\0';
    trim_whitespace(param->name);

    // 提取数组部分
    const char *array_start = strchr(colon_pos + 1, '[');
    const char *array_end = strchr(colon_pos + 1, ']');
    if (!array_start || !array_end || array_start > array_end) {
        fprintf(stderr, "Invalid array format: %s\n", line);
        return -1;
    }

    // 解析数组元素
    char array_content[MAX_LINE_LENGTH];
    strncpy(array_content, array_start + 1, array_end - array_start - 1);
    array_content[array_end - array_start - 1] = '\0';

    // 动态分配数组
    param->values = malloc(MAX_LINE_LENGTH * sizeof(float));
    if (!param->values) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    param->size = 0;

    char *token = strtok(array_content, ", ");
    while (token) {
        param->values[param->size++] = strtof(token, NULL);
        token = strtok(NULL, ", ");
    }

    return 0;
}

// 释放参数内存
void free_parameter(Parameter *param) {
    free(param->values);
    param->values = NULL;
    param->size = 0;
}

void parse_ckpt(const char *filename, ModelStateDict *sd) {
    Parameter *params = sd->params;
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        // 去除行尾换行符
        line[strcspn(line, "\n")] = 0;

        if (parse_line(line, params + sd->size) == 0) {
            // 输出解析结果
            // printf("Parameter Name: %s\n", params[sd->size].name);
            // printf("Values: [");
            // for (int i = 0; i < params[sd->size].size; i++) {
            //     printf("%f", params[sd->size].values[i]);
            //     if (i < params[sd->size].size - 1) printf(", ");
            // }
            // printf("]\n");
            sd->size ++;
        }
    }

    fclose(file);
}

ModelStateDict *create_model_state_dict() {
    ModelStateDict *sd = (ModelStateDict*)malloc(sizeof(ModelStateDict));
    sd->params = (Parameter*)malloc(MAX_PARAMS * sizeof(Parameter));
    sd->size = 0;
    return sd;
}

void free_model_state_dict(ModelStateDict *sd) {
    for (int i = 0; i < sd->size; i ++ ) {
        free_parameter(&sd->params[i]);
    }
    free(sd->params);
}