#ifndef ML_RNG_H
#define ML_RNG_H

#include "ml_error.h"
#include "ml_primitives.h"

typedef f32 (*ML_RngNextF32_01_Fn)(void* ctx);

typedef struct {
  void* ctx;
  ML_RngNextF32_01_Fn next01;
} ML_Rng;

ML_Status ML_Rng_next01(const ML_Rng* rng,f32* out);

ML_Status Mat_xavier_uniform_dense(Matf32* W, const ML_Rng* rng);
ML_Status Mat_xavier_uniform(Matf32* W, const ML_Rng* rng, u64 fan_in, u64 fan_out);

#endif //ML_RNG_H
