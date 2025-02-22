#include "../include/kiss_fft.h"
#include "../include/_kiss_fft_guts.h"
#include "../include/pfdkf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void fft(kiss_fft_cpx *out, const float *in, int fft_len) {
    // 这个fft和python中各种fft差一个fft_len的倍数，这里给结果乘上fft_len和python对齐
    kiss_fft_cpx x[fft_len];
    kiss_fft_cpx y[fft_len];
    for (int i = 0; i < fft_len; i ++) {
      x[i].r = in[i];
      x[i].i = 0;
    }
    kiss_fft_state *kfft;
    kfft = opus_fft_alloc_twiddles(fft_len, NULL, NULL, NULL, 0);
    opus_fft(kfft, x, y, 0);
    for (int i = 0; i < fft_len/2+1; i ++) {
        out[i].r = y[i].r * fft_len;
        out[i].i = y[i].i * fft_len;
    }
    free(kfft);
}

void ifft(float *out, const kiss_fft_cpx *in, int fft_len) {
    kiss_fft_cpx x[fft_len];
    kiss_fft_cpx y[fft_len];
    int freq_size = fft_len / 2 + 1;
    int i;
    for (i = 0; i < freq_size; i++ ) {
        x[i] = in[i];
    }
    for (; i < fft_len; i++ ) {
        x[i].r = x[fft_len - i].r;
        x[i].i = -x[fft_len - i].i;
    }
    kiss_fft_state *kfft;
    kfft = opus_fft_alloc_twiddles(fft_len, NULL, NULL, NULL, 0);
    opus_fft(kfft, x, y, 0);
    /* output in reverse order for IFFT. */
    out[0] = y[0].r;
    for (i = 1; i < fft_len; i ++ ) {
      out[i] = y[fft_len - i].r;
    }
    free(kfft);
}

PFDKF *init_pfdkf(int N, int M, float A, float P_initial) {
    PFDKF *filter = (PFDKF *)malloc(sizeof(PFDKF));

    filter->N = N;
    filter->M = M;
    filter->A = A;
    filter->A2 = A * A;
    filter->m_smooth_factor = 0.5;

    filter->x = (float *)calloc(2 * M, sizeof(float));
    filter->m = (float *)calloc(M + 1, sizeof(float));
    filter->P = (float *)malloc(N * (M + 1) * sizeof(float));
    for (int i = 0; i < N * (M + 1); i++) {
        filter->P[i] = P_initial;
    }

    filter->X = (kiss_fft_cpx *)calloc(N * (M + 1), sizeof(kiss_fft_cpx));
    filter->H = (kiss_fft_cpx *)calloc(N * (M + 1), sizeof(kiss_fft_cpx));
    filter->mu = (kiss_fft_cpx *)calloc(N * (M + 1), sizeof(kiss_fft_cpx));
    filter->E = (kiss_fft_cpx *)calloc(M + 1, sizeof(kiss_fft_cpx));

    filter->half_window = (float *)calloc(2 * M, sizeof(float));
    for (int i = 0; i < M; i++) {
        filter->half_window[i] = 1.0f;
    }

    return filter;
}

void free_pfdkf(PFDKF *filter) {
    free(filter->x);
    free(filter->m);
    free(filter->P);
    free(filter->X);
    free(filter->H);
    free(filter->mu);
    free(filter->half_window);
    free(filter);
}

void filt(PFDKF *filter, const float *x, const float *d, float *e_out, float *y_out) {
    int M = filter->M;
    int N = filter->N;

    // Update x buffer
    memmove(filter->x, filter->x + M, M * sizeof(float));
    memcpy(filter->x + M, x, M * sizeof(float));

    // Perform FFT on x buffer
    kiss_fft_cpx X[M + 1];
    fft(X, filter->x, 2 * M);

    // Shift X and update the latest FFT
    memmove(filter->X + (M + 1), filter->X, (N - 1) * (M + 1) * sizeof(kiss_fft_cpx));
    memcpy(filter->X, X, (M + 1) * sizeof(kiss_fft_cpx));

    // Compute output Y
    kiss_fft_cpx Y[M + 1];
    for (int i = 0; i < (M + 1); i ++ ) {
        Y[i].r = 0;
        Y[i].i = 0;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= M; j++) {
            Y[j].r += filter->H[i * (M + 1) + j].r * filter->X[i * (M + 1) + j].r -
                      filter->H[i * (M + 1) + j].i * filter->X[i * (M + 1) + j].i;
            Y[j].i += filter->H[i * (M + 1) + j].r * filter->X[i * (M + 1) + j].i +
                      filter->H[i * (M + 1) + j].i * filter->X[i * (M + 1) + j].r;
        }
    }

    // Perform IFFT to get time-domain output y
    float y[2 * M];
    for (int i = 0; i < 2 * M; i ++ )
        y[i] = 0;
    ifft(y, Y, 2 * M);

    // Compute error signal
    for (int i = 0; i < M; i++) {
        e_out[i] = d[i] - y[M + i];
    }

    // Update error FFT
    float e_fft[2 * M];
    for (int i = 0; i < 2 * M; i ++ )
        e_fft[i] = 0;
    memcpy(e_fft + M, e_out, M * sizeof(float));
    fft(filter->E, e_fft, 2 * M);

    // Update variables for next iteration
    float X2[M + 1];
    for (int i = 0; i < (M + 1); i ++ )
        X2[i] = 0;
    for (int j = 0; j <= M; j++) {
        for (int i = 0; i < N; i++) {
            X2[j] += filter->X[i * (M + 1) + j].r * filter->X[i * (M + 1) + j].r +
                     filter->X[i * (M + 1) + j].i * filter->X[i * (M + 1) + j].i;
        }
    }

    kiss_fft_cpx W[M + 1];
    kiss_fft_cpx E_res[M + 1];
    for (int i = 0; i < M + 1; i ++ ) {
        W[i].r = 0;
        W[i].i = 0;
        E_res[i].r = 0;
        E_res[i].i = 0;
    }

    float R[M + 1];
    for (int i = 0; i < M + 1; i ++ )
        R[i] = 0;
    for (int j = 0; j <= M; j++) {
        filter->m[j] = filter->m_smooth_factor * filter->m[j] +
                       (1 - filter->m_smooth_factor) * (filter->E[j].r * filter->E[j].r + filter->E[j].i * filter->E[j].i);
        for (int i = 0; i < N; i++) {
            R[j] += filter->P[i * (M + 1) + j] *
                 (filter->X[i * (M + 1) + j].r * filter->X[i * (M + 1) + j].r +
                  filter->X[i * (M + 1) + j].i * filter->X[i * (M + 1) + j].i);
        }
        R[j] += 2 * filter->m[j] / N;

        // filter->mu[j].r = filter->P[j] / (R[j] + EPSILON);
        // filter->mu[j].i = 0; // Assuming real-valued mu

        for (int i = 0; i < N; i ++ ) {
            int index = i * (M + 1) + j;
            filter->mu[index].r = filter->P[index] / (R[j] + EPSILON);
            filter->mu[index].i = 0;
            W[j].r += filter->mu[index].r * 
                      (filter->X[i * (M + 1) + j].r * filter->X[i * (M + 1) + j].r +
                       filter->X[i * (M + 1) + j].i * filter->X[i * (M + 1) + j].i);
            W[j].i += filter->mu[index].i * 
                      (filter->X[i * (M + 1) + j].r * filter->X[i * (M + 1) + j].r +
                       filter->X[i * (M + 1) + j].i * filter->X[i * (M + 1) + j].i);
        }
        W[j].r = 1 - W[j].r;
        W[j].i = - W[j].i;

        E_res[j].r = W[j].r * filter->E[j].r - W[j].i * filter->E[j].i;
        E_res[j].i = W[j].r * filter->E[j].i + W[j].i * filter->E[j].r;
    }

    float e_mid[2 * M];
    for (int i = 0; i < 2 * M; i ++ )
        e_mid[i] = 0;
    ifft(e_mid, E_res, 2 * M);
    for (int i = 0; i < M; i ++ ) {
        e_out[i] = e_mid[M + i];
        y_out[i] = d[i] - e_out[i];
    }
}

void update(PFDKF *filter) {
    int N = filter->N;
    int M = filter->M;

    // Compute G = mu * X.conj()
    kiss_fft_cpx G[N * (M + 1)];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= M; j++) {
            int idx = i * (M + 1) + j;
            G[idx].r = filter->mu[idx].r * filter->X[idx].r + filter->mu[idx].i * filter->X[idx].i;
            G[idx].i = filter->mu[idx].r * -filter->X[idx].i + filter->mu[idx].i * filter->X[idx].r;
        }
    }

    // Update P = A2 * (1 - 0.5 * G * X) * P + (1 - A2) * abs(H)^2
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= M; j++) {
            int idx = i * (M + 1) + j;

            float G_X_real = G[idx].r * filter->X[idx].r - G[idx].i * filter->X[idx].i;
            float G_X = 0.5f * G_X_real;

            float abs_H2 = filter->H[idx].r * filter->H[idx].r + filter->H[idx].i * filter->H[idx].i;

            filter->P[idx] = filter->A2 * (1.0f - G_X) * filter->P[idx] + (1.0f - filter->A2) * abs_H2;
        }
    }

    // Update H = A * (H + fft_result)
    for (int i = 0; i < N; i++) {
        // Compute fft(half_window * ifft(E * G).real)
        kiss_fft_cpx EG[M + 1];
        for (int j = 0; j <= M; j++) {
            int index = i * (M + 1) + j;
            EG[j].r = filter->E[j].r * G[index].r - filter->E[j].i * G[index].i;
            EG[j].i = filter->E[j].r * G[index].i + filter->E[j].i * G[index].r;
        }

        float ifft_result[2 * M];
        for (int i = 0; i < 2 * M; i ++ ) {
            ifft_result[i] = 0;
        }
        ifft(ifft_result, EG, 2 * M);

        float windowed[2 * M];
        for (int i = 0; i < 2 * M; i++) {
            windowed[i] = filter->half_window[i] * ifft_result[i];
        }

        kiss_fft_cpx fft_result[M + 1];
        for (int i = 0; i < M + 1; i ++ ) {
            fft_result[i].r = 0;
            fft_result[i].i = 0;
        }
        fft(fft_result, windowed, 2 * M);
        for (int j = 0; j <= M; j++) {
            int idx = i * (M + 1) + j;

            filter->H[idx].r = filter->A * (filter->H[idx].r + fft_result[j].r);
            filter->H[idx].i = filter->A * (filter->H[idx].i + fft_result[j].i);
        }
    }
}


void pfdkf(const float *x, const float *d, float *e_out, float *y_out, int wav_length) {
    int N = 10;
    // int M = 400;
    int M = 256;
    float A = 0.999;
    float P_initial = 10.0;

    PFDKF *filter = init_pfdkf(N, M, A, P_initial);
    int num_block = wav_length / M;
    float *e_n = (float*)malloc(M * sizeof(float));
    float *y_n = (float*)malloc(M * sizeof(float));
    for (int i = 0; i < num_block; i ++ ) {
        filt(filter, x + i * M, d + i * M, e_n, y_n);
        update(filter);
        for (int j = 0; j < M; j ++ ) {
            e_out[i * M + j] = e_n[j];
            y_out[i * M + j] = y_n[j];
        }
    }
    free(e_n);
    free(y_n);
    free_pfdkf(filter);
}

// Main function
// int main() {
//     // Example usage of PFDKF
//     // Initialize filter
//     int N = 10;
//     int M = 256;
//     float A = 0.999;
//     float P_initial = 10.0;

//     PFDKF *filter = init_pfdkf(N, M, A, P_initial);

//     // Test data
//     float x[256]; // Input signal block
//     float d[256]; // Desired signal block

//     // Fill test data with example values
//     for (int i = 0; i < 256; i++) {
//         x[i] = sin(2 * M_PI * i / 256);
//         d[i] = x[i] + 0.1 * ((float)rand() / RAND_MAX - 0.5);
//     }

//     float e[256], y[256];
//     filt(filter, x, d, e, y);

//     // Output results
//     for (int i = 0; i < 256; i++) {
//         printf("e[%d] = %f, y[%d] = %f\n", i, e[i], i, y[i]);
//     }

//     free_pfdkf(filter);
//     return 0;
// }
