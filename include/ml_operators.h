#ifndef ML_OPERATORS_H
#define ML_OPERATORS_H

#include "ml_error.h"
#include "ml_primitives.h"
#include "ml_rng.h"

typedef enum {
  FILL_ONES,
  FILL_ZEROS,
  FILL_XAVIER_UNIFORM,
} FillStrategy;

typedef struct {
  u64 in_rows;
  u64 in_cols;
  u64 out_cols;
  FillStrategy fillW_strat;
  FillStrategy fillb_strat;
  ML_Rng* rng;
} LinearConfig;

ML_Status create_config_Linear(LinearConfig* conf,u64 inrows, u64 incols,
                               u64 outcols, ML_Rng* rng
			       ,FillStrategy w_strat,FillStrategy b_strat);

typedef struct {
  Matf32 W; 
  Matf32 X;
  Matf32 b;

  Matf32 dW;
  Matf32 db;
  Matf32 X_T;

  Matf32 Z;
} Linear;

ML_Status create_op_Linear(ml_arena* arena,Linear* lin,LinearConfig conf);
ML_Status execute_op_Linear_forward(Linear* lin,const Matf32 in);
ML_Status execute_op_Linear_backward(Linear* lin, const Matf32 dZ);
ML_Status execute_op_Linear_sgd_step(Linear* lin, f32 lr);

typedef struct { 
  u64 in_rows;
  u64 in_cols;
} SoftmaxConfig;

ML_Status create_config_Softmax(SoftmaxConfig* conf,u64 inrows,u64 incols);

typedef struct {
  Matf32 rowmax;
  Matf32 rowsum;
  
  Matf32 P;
} Softmax;

ML_Status create_op_Softmax(ml_arena* arena,Softmax* softmax,SoftmaxConfig conf);
ML_Status execute_op_Softmax_forward(Softmax* sm, const Matf32 Z);

typedef struct {
  u64 in_rows;  // N
  u64 in_cols;  // C
} CEConfig;

ML_Status create_config_CrossEntropy(CEConfig *conf, u64 inrows, u64 incols);

typedef struct {
  f32 loss;   // last forward loss (mean)

  // gradient wrt logits: dZ = (P - Y)/N
  Matf32 dZ;  // (N x C)
} CrossEntropy;

ML_Status create_op_CrossEntropy(ml_arena* arena, CrossEntropy* ce, CEConfig conf);

// Forward: computes ce->loss from probabilities P and labels Y
ML_Status execute_op_CrossEntropy_forward(CrossEntropy *ce, const Matf32 P,
                                          const Matf32 Y);

// Backward: computes ce->dZ = (P - Y)/N
ML_Status execute_op_CrossEntropy_backward(CrossEntropy *ce, const Matf32 P,
                                           const Matf32 Y);

#endif //MK_OPERATORS_H
