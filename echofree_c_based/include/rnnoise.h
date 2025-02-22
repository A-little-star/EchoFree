#ifndef RNNOISE_H
#define RNNOISE_H
#include "typedef.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
  float analysis_mem[WINDOW_SIZE-FRAME_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[WINDOW_SIZE-FRAME_SIZE];
  float last_gain;
  int last_period;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
} DenoiseState;

int rnnoise_get_size();

int rnnoise_init(DenoiseState *st);

DenoiseState *rnnoise_create();
void rnnoise_destroy(DenoiseState *st);

#endif