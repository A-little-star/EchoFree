#ifndef STFT_H
#define STFT_H
#include "tensor.h"
#include "rnnoise.h"

void feature_extract(Tensor* input, Tensor* cspecs, Tensor* features);
void feature_extract_frame(Tensor* input, Tensor* cspecs, Tensor* features, DenoiseState* st);
float* istft(Tensor* cspecs, int num_samples);
void post_process(Tensor* cspecs, Tensor* gains, float* outputs_ptr);
void post_process_frame(Tensor* cspecs, Tensor* gains, float* out, DenoiseState* st);

#endif