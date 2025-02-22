#ifndef PFDKF_H
#define PFDKF_H
#include "kiss_fft.h"

void fft(kiss_fft_cpx *out, const float *in, int fft_len);

void ifft(float *out, const kiss_fft_cpx *in, int fft_len);

typedef struct {
    int N;
    int M;
    float A;
    float A2;
    float m_smooth_factor;

    float *x;
    float *m;
    float *P;
    kiss_fft_cpx *X;
    kiss_fft_cpx *H;
    kiss_fft_cpx *mu;
    kiss_fft_cpx *E;
    float *half_window;
} PFDKF;

PFDKF *init_pfdkf(int N, int M, float A, float P_initial);
void free_pfdkf(PFDKF *filter);

void filt(PFDKF *filter, const float *x, const float *d, float *e_out, float *y_out);
void update(PFDKF *filter);
void pfdkf(const float *x, const float *d, float *e_out, float *y_out, int wav_length);

#endif