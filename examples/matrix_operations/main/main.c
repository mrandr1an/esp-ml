#include <stdio.h>
#include <math.h>
#include "esp_log.h"
#include "esp_random.h"

#include "ml_alloc.h"
#include "ml_error.h"
#include "ml_rng.h"
#include "ml_primitives.h"
#include "ml_operators.h"
#include "ml_models.h"

static const char *TAG = "ESP-ML";

static f32 esp32_next01(void* ctx) {
  (void)ctx;
  uint32_t x = esp_random();
  return (f32)((double)x / 4294967296.0); // [0,1)
}

static const char* class_name(u64 k) {
  switch (k) {
    case 0: return "ACE";
    case 1: return "PASS";
    case 2: return "MIDPASS";
    default: return "FAIL";
  }
}

static ML_Status set_onehot_row(Matf32* Y, u64 r, u64 cls) {
  if (!Y || !Y->data) return ML_INVALID_ARGUMENT;
  if (r >= Y->rows || cls >= Y->cols) return ML_INVALID_ARGUMENT;

  ML_Status st = ML_OK;
  for (u64 c = 0; c < Y->cols; ++c) {
    st = MatSet(Y, r, c, (c == cls) ? 1.0f : 0.0f);
    if (st != ML_OK) return st;
  }
  return ML_OK;
}

static f32 mark_from_features(f32 focus, f32 hours) {
  f32 m = 0.6f * focus + 0.4f * hours;
  if (m < 0.0f) m = 0.0f;
  if (m > 10.0f) m = 10.0f;
  return m;
}

static u64 class_from_mark(f32 mark) {
  if (mark >= 9.0f) return 0; // ACE
  if (mark >= 6.0f) return 1; // PASS
  if (mark >= 4.0f) return 2; // MIDPASS
  return 3;                   // FAIL
}

static ML_Status argmax_row(const Matf32 P, u64 r, u64* out_idx) {
  if (!P.data || !out_idx) return ML_INVALID_ARGUMENT;
  if (r >= P.rows) return ML_INVALID_ARGUMENT;

  ML_Status st = ML_OK;
  f32 best = 0.0f;
  st = MatGet(P, r, 0, &best);
  if (st != ML_OK) return st;

  u64 best_i = 0;
  for (u64 c = 1; c < P.cols; ++c) {
    f32 v = 0.0f;
    st = MatGet(P, r, c, &v);
    if (st != ML_OK) return st;
    if (v > best) { best = v; best_i = c; }
  }

  *out_idx = best_i;
  return ML_OK;
}

void app_main(void) {
  ESP_LOGI(TAG, "Starting...");

  // You might need bigger if you increase N or add more operators.
  static unsigned char arena_mem[KiB(5)];
  ml_arena arena;
  ML_Status st = create_ml_arena(&arena, arena_mem, (u64)sizeof(arena_mem));
  if (st != ML_OK) {
    ESP_LOGE(TAG, "create_ml_arena failed: %u", (unsigned)st);
    return;
  }

  // Dimensions
  const u64 N   = 32; // batch
  const u64 D   = 2;  // focus, hours
  const u64 C   = 4;  // ACE, PASS, MIDPASS, FAIL

  // Dataset buffers for training
  Matf32 X, Y;
  st = create_Mat(&arena, &X, N, D);
  if (st != ML_OK) { ESP_LOGE(TAG, "create_Mat(X) failed: %u", (unsigned)st); return; }
  st = create_Mat(&arena, &Y, N, C);
  if (st != ML_OK) { ESP_LOGE(TAG, "create_Mat(Y) failed: %u", (unsigned)st); return; }

  // RNG used for init + dataset generation
  ML_Rng rng = { .ctx = NULL, .next01 = esp32_next01 };

  // Fill synthetic dataset (focus,hours in [0,10])
  for (u64 i = 0; i < N; ++i) {
    f32 focus = 10.0f * rng.next01(rng.ctx);
    f32 hours = 10.0f * rng.next01(rng.ctx);

    st = MatSet(&X, i, 0, focus);
    if (st != ML_OK) return;
    st = MatSet(&X, i, 1, hours);
    if (st != ML_OK) return;

    f32 mark = mark_from_features(focus, hours);
    u64 cls = class_from_mark(mark);

    st = set_onehot_row(&Y, i, cls);
    if (st != ML_OK) return;
  }

  // ---- Create training model (N fixed) ----
  SoftmaxRegression model;
  SoftmaxRegressionConfig mconf;
  st = create_config_SoftmaxRegression(&mconf, N, D, C, &rng,
                                       FILL_XAVIER_UNIFORM, // W init
                                       FILL_ZEROS);         // b init
  if (st != ML_OK) { ESP_LOGE(TAG, "create_config_SoftmaxRegression failed: %u", (unsigned)st); return; }

  st = create_model_SoftmaxRegression(&arena, &model, mconf);
  if (st != ML_OK) { ESP_LOGE(TAG, "create_model_SoftmaxRegression failed: %u", (unsigned)st); return; }

  // ---- Train ----
  const f32 lr = 0.2f;
  const u64 epochs = 500;

  for (u64 e = 0; e < epochs; ++e) {
    f32 loss = 0.0f;
    st = train_step_SoftmaxRegression(&model, X, Y, lr, &loss);
    if (st != ML_OK) { ESP_LOGE(TAG, "train_step failed: %u", (unsigned)st); return; }

    if ((e % 20) == 0 || e == epochs - 1) {
      ESP_LOGI(TAG, "epoch=%llu loss=%.6f", (unsigned long long)e, (double)loss);
    }
  }

  // ---- Inference model (N=1) ----
  // This avoids forcing infer() to run on the training batch size.
  SoftmaxRegression model1;
  SoftmaxRegressionConfig mconf1;
  st = create_config_SoftmaxRegression(&mconf1, 1, D, C, &rng,
                                       FILL_ZEROS, FILL_ZEROS);
  if (st != ML_OK) { ESP_LOGE(TAG, "create_config_SoftmaxRegression(1) failed: %u", (unsigned)st); return; }

  st = create_model_SoftmaxRegression(&arena, &model1, mconf1);
  if (st != ML_OK) { ESP_LOGE(TAG, "create_model_SoftmaxRegression(1) failed: %u", (unsigned)st); return; }

  // Copy trained params into model1 so inference uses learned weights.
  st = MatCopy_into(&model1.lin.W, model.lin.W);
  if (st != ML_OK) return;
  st = MatCopy_into(&model1.lin.b, model.lin.b);
  if (st != ML_OK) return;

  // Single-sample buffers
  Matf32 Xt, Pt;
  st = create_Mat(&arena, &Xt, 1, D);
  if (st != ML_OK) { ESP_LOGE(TAG, "create_Mat(Xt) failed: %u", (unsigned)st); return; }
  st = create_Mat(&arena, &Pt, 1, C);
  if (st != ML_OK) { ESP_LOGE(TAG, "create_Mat(Pt) failed: %u", (unsigned)st); return; }

  struct { f32 focus; f32 hours; } tests[] = {
    {9.5f, 9.0f},
    {7.0f, 6.0f},
    {4.5f, 4.0f},
    {2.0f, 1.0f},
  };

  for (u64 t = 0; t < (u64)(sizeof(tests)/sizeof(tests[0])); ++t) {
    MatSet(&Xt, 0, 0, tests[t].focus);
    MatSet(&Xt, 0, 1, tests[t].hours);

    st = infer_SoftmaxRegression(&model1, Xt, &Pt);
    if (st != ML_OK) { ESP_LOGE(TAG, "infer failed: %u", (unsigned)st); return; }

    u64 pred = 0;
    st = argmax_row(Pt, 0, &pred);
    if (st != ML_OK) return;

    ESP_LOGI(TAG, "test focus=%.2f hours=%.2f => pred=%s",
             (double)tests[t].focus, (double)tests[t].hours, class_name(pred));

    printf("  P = [");
    for (u64 c = 0; c < C; ++c) {
      f32 p = 0.0f;
      MatGet(Pt, 0, c, &p);
      printf("%s%.4f", (c ? ", " : ""), (double)p);
    }
    printf("]\n");
  }

  printf("Free mem: %llu bytes\n", (unsigned long long)get_freemem_ml_arena_bytes(&arena));
  ESP_LOGI(TAG, "Done.");
}
