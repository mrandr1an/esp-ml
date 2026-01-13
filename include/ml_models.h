#ifndef ML_MODELS_H
#define ML_MODELS_H

#include "ml_operators.h"
#include "ml_rng.h"
typedef struct {
  // Fills X (N×D) and Y (N×C) for the next batch.
  // Return ML_OK if a batch was produced.
  // Return ML_DONE (or similar) when the epoch is finished.
  // Return other errors on failure.
  ML_Status (*next_batch)(void* ctx, Matf32* X, Matf32* Y);
  void* ctx;
} ML_BatchProvider;

typedef struct {
  u64 epochs;
  f32 lr;
  // Optional: call every k steps (0 disables)
  u64 log_every;
} ML_TrainConfig;

typedef struct {
  u64 N; //Batch size (fixed)
  u64 D; //input dim (fixed)
  u64 C; //number of classes (fixed)
  FillStrategy w_init;
  FillStrategy b_init;
  ML_Rng* rng;
} SoftmaxRegressionConfig;

ML_Status create_config_SoftmaxRegression(SoftmaxRegressionConfig* conf,
                                         u64 N, u64 D, u64 C,
                                         ML_Rng* rng,
                                         FillStrategy w_init,
                                         FillStrategy b_init);

typedef struct {
  SoftmaxRegressionConfig conf;
  Linear lin;
  Softmax sm;
  CrossEntropy ce;
} SoftmaxRegression;

ML_Status create_model_SoftmaxRegression(ml_arena* arena,
                                        SoftmaxRegression* m,
                                        SoftmaxRegressionConfig conf);

ML_Status infer_SoftmaxRegression(SoftmaxRegression* m,
                                 const Matf32 X,
                                 Matf32* outP);

// One SGD step: Y must be (N×C) one-hot, returns loss
ML_Status train_step_SoftmaxRegression(SoftmaxRegression* m,
                                      const Matf32 X,
                                      const Matf32 Y,
                                      f32 lr,
                                      f32* out_loss);

// Runs a standard loop over epochs, consuming batches from provider.
// Xbuf and Ybuf must be preallocated (N×D) and (N×C).
ML_Status train_SoftmaxRegression(SoftmaxRegression* m,
                                 ML_BatchProvider provider,
                                 ML_TrainConfig tconf,
                                 Matf32* Xbuf,
                                 Matf32* Ybuf,
                                 f32* out_last_loss);
#endif //ML_MODELS_H

