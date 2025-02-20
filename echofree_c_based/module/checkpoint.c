#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/parser.h"
#include "../include/cJSON.h"

// 解析 JSON 文件
ModelStateDict *parse_json_to_parameters(const char *filename) {
    // 打开文件
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Could not open file");
        return NULL;
    }

    // 读取文件内容
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *content = (char *)malloc(file_size + 1);
    if (!content) {
        fclose(file);
        perror("Memory allocation failed");
        return NULL;
    }

    fread(content, 1, file_size, file);
    content[file_size] = '\0';
    fclose(file);

    // 解析 JSON
    cJSON *json = cJSON_Parse(content);
    free(content);
    if (!json) {
        fprintf(stderr, "Error before: %s\n", cJSON_GetErrorPtr());
        return NULL;
    }

    ModelStateDict *sd = (ModelStateDict*)malloc(sizeof(ModelStateDict));
    int count = cJSON_GetArraySize(json);
    sd->size = count;
    sd->params = (Parameter *)malloc(count * sizeof(Parameter));

    Parameter *parameters = sd->params;
    if (!parameters) {
        cJSON_Delete(json);
        perror("Memory allocation failed");
        return NULL;
    }

    // 遍历 JSON 对象
    cJSON *item = NULL;
    int index = 0;
    cJSON_ArrayForEach(item, json) {
        // 获取键名
        strncpy(parameters[index].name, item->string, MAX_PARAM_NAME - 1);
        parameters[index].name[MAX_PARAM_NAME - 1] = '\0';
        // printf("%s\n", parameters[index].name);

        // 获取值数组
        cJSON *values = cJSON_GetObjectItem(json, item->string);
        if (cJSON_IsArray(values)) {
            parameters[index].size = cJSON_GetArraySize(values);
            parameters[index].values = (float *)malloc(parameters[index].size * sizeof(float));
            if (!parameters[index].values) {
                perror("Memory allocation failed");
                free_parameters(parameters, index);
                cJSON_Delete(json);
                return NULL;
            }

            for (int i = 0; i < parameters[index].size; i++) {
                cJSON *value = cJSON_GetArrayItem(values, i);
                parameters[index].values[i] = (float)cJSON_GetNumberValue(value);
            }
        } else {
            parameters[index].size = 0;
            parameters[index].values = NULL;
        }
        index++;
    }

    cJSON_Delete(json);
    return sd;
}

// 释放内存
void free_parameters(Parameter *parameters, int param_count) {
    for (int i = 0; i < param_count; i++) {
        free(parameters[i].values);
    }
    free(parameters);
}
