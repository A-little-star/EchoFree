#ifndef WAVREADER_H
#define WAVREADER_H

float* read_wav_file(const char *filename, int *num_samples, int *sample_rate);
int write_wav_file(const char *filename, const float *audio_data, int num_samples, int sample_rate, int num_channels, int bits_per_sample);
void wav_norm(float* audio_data, int num_samples);
void wav_invnorm(float* audio_data, int num_samples);

#endif