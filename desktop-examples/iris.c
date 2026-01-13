// desktop_examples/basic.c
#include "matrix_utils.h"
#include "ml_alloc.h"
#include "ml_error.h"
#include "ml_models.h"
#include "ml_operators.h"
#include "ml_primitives.h"
#include "ml_rng.h"
#include <sys/random.h> 
#include <errno.h> 
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "csv_utils.h"

#define PATH_MAX 1024

static f32 getrandom_next01(void* ctx) {
  (void)ctx;

  uint32_t x = 0;
  size_t got = 0;

  while (got < sizeof(x)) {
    ssize_t n =
      getrandom(((unsigned char*)&x) + got, sizeof(x) - got, 0);
    if (n > 0) { got += (size_t)n; continue; }
    if (n == -1 && errno == EINTR) continue;
    return 0.0f;
  }

  return (f32)((double)x / 4294967296.0);
}

#define setosa "Iris-setosa"
#define versicolor "Iris-versicolor"
#define virginica "Iris-virginica"

u64 class_to_index(char label[64]) {
  if (strcmp(label, setosa) == 0) {
    return 0.0f;
  } else if (strcmp(label, versicolor) == 0) { 
    return 1.0f;
  } else if (strcmp(label, virginica) == 0) {
    return 2.0f;
  } else {
    return 4.0f;
  }
}

typedef struct {
  Csv* csv;
  size_t rows;   // csv_num_rows (includes header)
  u64 N, D, C;
  size_t cursor; // current csv row (data starts at 1)
} IrisBatchCtx;

static ML_Status iris_next_batch(void* ctxp, Matf32* X, Matf32* Y) {
  IrisBatchCtx* ctx = (IrisBatchCtx*)ctxp;

  const size_t first_data_row = 1;
  if (ctx->cursor < first_data_row) ctx->cursor = first_data_row;

  if (ctx->cursor + (size_t)ctx->N > ctx->rows) {
    ctx->cursor = first_data_row;
    return ML_DONE;
  }

  ML_Status status = MatFillScalar(Y, 0.0f);
  if (status != ML_OK) return status;

  for (u64 i = 0; i < ctx->N; ++i) {
    size_t csv_row = ctx->cursor + (size_t)i;

    for (u64 j = 0; j < ctx->D; ++j) {
      size_t csv_col = (size_t)j + 1;
      f32 v = 0.0f;
      if (!csv_get_f32(ctx->csv, csv_row, csv_col, &v)) return ML_INVALID_ARGUMENT;
      status = MatSet(X, i, j, v);
      if (status != ML_OK) return status;
    }

    char label[64];
    if (!csv_get_string(ctx->csv, csv_row, 5, label, sizeof(label))) return ML_INVALID_ARGUMENT;

    int cls = class_to_index(label);
    if (cls < 0 || (u64)cls >= ctx->C) return ML_INVALID_ARGUMENT;

    status = MatSet(Y, i, (u64)cls, 1.0f);
    if (status != ML_OK) return status;
  }

  ctx->cursor += (size_t)ctx->N;
  return ML_OK;
}

int main() {
  printf("Starting...\n");
  ML_Status status = ML_OK;
  static unsigned char arena_mem[KiB(100)];
  ml_arena arena;
  status = create_ml_arena(&arena,arena_mem,(u64)sizeof(arena_mem));
  if (status != ML_OK)
    printf("Error at arena allocation: %d\n",status);

  //Dimensions for Iris Data Set

  //Samples
  const u64 N = 150;
  // 4 Features
  // sepal length
  // sepal width
  // petal length
  // petal width
  const u64 D = 4;
  // 3 Classes
  // Setosa (0)
  // Versicolor (1)
  // Virginica (2)
  const u64 C = 3;

  // Dataset Matrices for training.
  // X is the feature matrix
  // Y is the one-hot matrix
  Matf32 X,Y;
  status = create_Mat(&arena,&X,N,D);
  if (status != ML_OK)
    printf("Error at creating dataset X: %d\n",status);
  status = create_Mat(&arena,&Y,N,C);
  if (status != ML_OK)
    printf("Error at creating dataset Y: %d\n",status);

  //Fill from dataset
  printf("Opening dataset...\n");
  Csv iris_dataset;
  int res =
    csv_load("desktop-examples/iris-dataset/Iris.csv",&iris_dataset);
  if (res == 0)
    printf("Error loading dataset\n");

  size_t rows = csv_num_rows(&iris_dataset);
  print_matrix("X_train",X);
  print_matrix("Y_train",Y);

  //Create classification model
  ML_Rng rng = {.ctx = NULL, .next01 = getrandom_next01};
  
  SoftmaxRegressionConfig mconf;
  SoftmaxRegression model;
  status = create_config_SoftmaxRegression(&mconf, N, D, C,
                                           &rng,
                                          FILL_XAVIER_UNIFORM,FILL_ZEROS);
 
  if (status != ML_OK)
    printf("Error at creating model config: %d\n",status);

  status = create_model_SoftmaxRegression(&arena,&model,mconf);
  if (status != ML_OK)
    printf("Error at creating model: %d\n",status);

  ML_TrainConfig tconf = {.epochs = 50, .lr = 0.05f};

  IrisBatchCtx bctx = {
    .csv = &iris_dataset,
    .rows = rows,
    .N = N,
    .D = D,
    .C = C,
    .cursor = 1,
  };
  ML_BatchProvider provider = {
  .next_batch = iris_next_batch,
  .ctx = &bctx,
  };

  // train
  f32 last_loss = 0.0f;
  printf("Starting to train...");
  status = train_SoftmaxRegression(&model,
				   provider,
                                   tconf,
                                   &X,
                                   &Y,
                                   &last_loss);
  if (status != ML_OK) { printf("train error: %d\n", status); return 1; }

  printf("...Done training,loss=%.6f\n", (double)last_loss);

  Matf32 in, prob;

  // in:  N×D
  status = create_Mat(&arena, &in, N, D);
  if (status != ML_OK) printf("create in error: %d\n", status);

  // prob: N×C  
  status = create_Mat(&arena, &prob, N, C);
  if (status != ML_OK) printf("create prob error: %d\n", status);

  // Fill 5 samples (example iris measurements)
  f32 x0[4] = {4.4f, 3.0f, 1.3f, 0.2f}; //setosa (0)
  f32 x1[4] = {7.7f,3.8f,6.7f,2.2f}; //virginica (2)
  f32 x2[4] = {5.6f,2.5f,3.9f,1.1f}; //versicolor (1)
  f32 x3[4] = {5.5f,2.5f,4.0f,1.3f}; //versicolor (1)
  f32 x4[4] = {4.6f,3.2f,1.4f,0.2f};//setosa (0)

  MatFillScalar(&in,0.0f);
  status = MatFillRow(&in, 0, x0);
  if (status != ML_OK) printf("fill row 0 error: %d\n", status);
  status = MatFillRow(&in, 1, x1);
  if (status != ML_OK) printf("fill row 1 error: %d\n", status);
  status = MatFillRow(&in, 2, x2);
  if (status != ML_OK) printf("fill row 2 error: %d\n", status);
  status = MatFillRow(&in, 3, x3);
  if (status != ML_OK) printf("fill row 3 error: %d\n", status);
  status = MatFillRow(&in, 4, x4);
  if (status != ML_OK) printf("fill row 4 error: %d\n", status);

  // Run inference
  status = infer_SoftmaxRegression(&model, in, &prob);
  if (status != ML_OK) printf("infer error: %d\n", status);

  print_matrix("Inference Result:",prob);

  printf("...Done\n");
  return 0; 
}
