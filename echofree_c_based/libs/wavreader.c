#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// WAV 文件头结构定义
typedef struct {
    char chunkID[4];        // "RIFF"
    uint32_t chunkSize;     // 文件大小 - 8
    char format[4];         // "WAVE"
    char subchunk1ID[4];    // "fmt "
    uint32_t subchunk1Size; // 16 for PCM
    uint16_t audioFormat;   // PCM = 1
    uint16_t numChannels;   // 声道数
    uint32_t sampleRate;    // 采样率
    uint32_t byteRate;      // 每秒字节数
    uint16_t blockAlign;    // 块对齐大小
    uint16_t bitsPerSample; // 每样本位数
    char subchunk2ID[4];    // "data"
    uint32_t subchunk2Size; // 数据块大小
} WAVHeader;

// 读取 WAV 文件并返回音频数据
float* read_wav_file(const char *filename, int *num_samples, int *sample_rate) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    // 读取 WAV 文件头
    WAVHeader header;
    fread(&header, sizeof(WAVHeader), 1, file);

    // 检查文件格式是否为 WAV
    if (strncmp(header.chunkID, "RIFF", 4) != 0 || strncmp(header.format, "WAVE", 4) != 0) {
        fprintf(stderr, "Invalid WAV file format\n");
        fclose(file);
        return NULL;
    }

    // 计算样本数（总数据字节数 / 每样本字节数）
    *num_samples = header.subchunk2Size / (header.bitsPerSample / 8) / header.numChannels;
    *sample_rate = header.sampleRate;

    // 为音频数据分配内存（转换为 float，一维数组形式）
    float *audio_data = (float *)malloc(*num_samples * sizeof(float));
    if (!audio_data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    // 读取音频数据
    if (header.bitsPerSample == 16) {
        int16_t *buffer = (int16_t *)malloc(header.subchunk2Size);
        if (!buffer) {
            fprintf(stderr, "Memory allocation failed for buffer\n");
            free(audio_data);
            fclose(file);
            return NULL;
        }
        fread(buffer, 1, header.subchunk2Size, file);
        for (int i = 0; i < *num_samples; i++) {
            audio_data[i] = buffer[i] / 32768.0f; // 归一化到 [-1.0, 1.0]
        }
        free(buffer);
    } else if (header.bitsPerSample == 8) {
        uint8_t *buffer = (uint8_t *)malloc(header.subchunk2Size);
        if (!buffer) {
            fprintf(stderr, "Memory allocation failed for buffer\n");
            free(audio_data);
            fclose(file);
            return NULL;
        }
        fread(buffer, 1, header.subchunk2Size, file);
        for (int i = 0; i < *num_samples; i++) {
            audio_data[i] = (buffer[i] - 128) / 128.0f; // 归一化到 [-1.0, 1.0]
        }
        free(buffer);
    } else {
        fprintf(stderr, "Unsupported bits per sample: %u\n", header.bitsPerSample);
        free(audio_data);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return audio_data;
}

// 写入 WAV 文件
int write_wav_file(const char *filename, const float *audio_data, int num_samples, int sample_rate, int num_channels, int bits_per_sample) {
    // 打开文件
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    // 准备 WAV 文件头
    WAVHeader header;
    memcpy(header.chunkID, "RIFF", 4);
    memcpy(header.format, "WAVE", 4);
    memcpy(header.subchunk1ID, "fmt ", 4);
    header.subchunk1Size = 16; // PCM 固定为 16
    header.audioFormat = 1; // PCM 格式
    header.numChannels = num_channels;
    header.sampleRate = sample_rate;
    header.bitsPerSample = bits_per_sample;
    header.blockAlign = (num_channels * bits_per_sample) / 8;
    header.byteRate = header.sampleRate * header.blockAlign;
    header.subchunk2Size = num_samples * header.blockAlign;
    header.chunkSize = 4 + (8 + header.subchunk1Size) + (8 + header.subchunk2Size);
    memcpy(header.subchunk2ID, "data", 4);

    // 写入文件头
    fwrite(&header, sizeof(WAVHeader), 1, file);

    // 写入音频数据
    if (bits_per_sample == 16) {
        int16_t *buffer = (int16_t *)malloc(num_samples * sizeof(int16_t));
        if (!buffer) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(file);
            return -1;
        }
        for (int i = 0; i < num_samples; i++) {
            buffer[i] = (int16_t)(audio_data[i] * 32768.0f); // 将浮点数据转换为 16 位整数
        }
        fwrite(buffer, sizeof(int16_t), num_samples, file);
        free(buffer);
    } else if (bits_per_sample == 8) {
        uint8_t *buffer = (uint8_t *)malloc(num_samples * sizeof(uint8_t));
        if (!buffer) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(file);
            return -1;
        }
        for (int i = 0; i < num_samples; i++) {
            buffer[i] = (uint8_t)((audio_data[i] * 128.0f) + 128); // 将浮点数据转换为 8 位整数
        }
        fwrite(buffer, sizeof(uint8_t), num_samples, file);
        free(buffer);
    } else {
        fprintf(stderr, "Unsupported bits per sample: %d\n", bits_per_sample);
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

void wav_norm(float* audio_data, int num_samples) {
    for (int i = 0; i < num_samples; i ++ )
        audio_data[i] *= 32768;
}

void wav_invnorm(float* audio_data, int num_samples) {
    for (int i = 0; i < num_samples; i ++ )
        audio_data[i] /= 32768;
}

// int main() {
//     const char *filename = "example.wav"; // 替换为实际文件名
//     read_wav_file(filename);
//     return 0;
// }
