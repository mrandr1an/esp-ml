#include "ml_rng.h"
#include <math.h>

ML_Status ML_Rng_next01(const ML_Rng *rng, f32 *out) {
  if (!out) return ML_INVALID_ARGUMENT;
  if (!rng || !rng->next01) return ML_INVALID_ARGUMENT;
  *out = rng->next01(rng->ctx);
  return ML_OK;
}

ML_Status Mat_xavier_uniform(Matf32* W, const ML_Rng* rng, u64 fan_in, u64 fan_out) {
  if (!W || !W->data) return ML_INVALID_ARGUMENT;
  if (!rng || !rng->next01) return ML_INVALID_ARGUMENT;
  if (fan_in == 0 || fan_out == 0) return ML_INVALID_ARGUMENT;

  // a = sqrt(6 / (fan_in + fan_out))
  f32 denom = (f32)(fan_in + fan_out);
  f32 a = sqrtf(6.0f / denom);

  ML_Status status = ML_OK;

  for (u64 r = 0; r < W->rows; ++r) {
    for (u64 c = 0; c < W->cols; ++c) {
      f32 u = 0.0f;                    // u in [0,1)
      status = ML_Rng_next01(rng, &u);
      if (status != ML_OK) return status;

      // map to [-a, a]: x = (2u - 1) * a
      f32 x = (2.0f * u - 1.0f) * a;

      status = MatSet(W, r, c, x);
      if (status != ML_OK) return status;
    }
  }

  return ML_OK;
}

ML_Status Mat_xavier_uniform_dense(Matf32* W, const ML_Rng* rng) {
  if (!W || !W->data) return ML_INVALID_ARGUMENT;
  return Mat_xavier_uniform(W, rng, /*fan_in=*/W->cols, /*fan_out=*/W->rows);
}
