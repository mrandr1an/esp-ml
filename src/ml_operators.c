#include "ml_operators.h"
#include "ml_error.h"
#include "ml_primitives.h"
#include <math.h>

ML_Status create_config_Linear(LinearConfig* conf,u64 inrows, u64 incols, u64 outcols,
                               ML_Rng* rng,FillStrategy w_strat, FillStrategy b_strat) {

  conf->in_cols = incols;
  conf->out_cols = outcols;
  conf->in_rows = inrows;
  conf->fillW_strat = w_strat;
  conf->fillb_strat = b_strat;
  conf->rng = rng;

  return ML_OK;
}

ML_Status create_op_Linear(ml_arena *arena, Linear *lin, LinearConfig conf) {
  if(!arena || !lin) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;
  Matf32 W,X,b,Z;
  Matf32 dW,db,X_T;

  //Allocate feature Matrix
  status = create_Mat(arena,&X,conf.in_rows,conf.in_cols);
  if (status != ML_OK) return status;
  //Allocate weight Matrix
  status = create_Mat(arena,&W,conf.in_cols,conf.out_cols);
  if (status != ML_OK) return status;
  //Allocate bias Matrix
  status = create_Mat(arena, &b, 1, conf.out_cols);
  if (status != ML_OK) return status;
  //Allocate logits Matrix
  status = create_Mat(arena, &Z, conf.in_rows, conf.out_cols);
  if (status != ML_OK) return status;

  //Allocate dW
  status = create_Mat(arena,&dW,conf.in_cols,conf.out_cols);
  if (status != ML_OK) return status;
  //Allocate db
  status = create_Mat(arena, &db, 1, conf.out_cols);
  if (status != ML_OK) return status;
  //Allocate XT
  status = create_Mat(arena, &X_T, conf.in_cols, conf.in_rows);
  if (status != ML_OK) return status;

  //Init weights
  switch (conf.fillW_strat) {
   case FILL_XAVIER_UNIFORM: {
     status = Mat_xavier_uniform_dense(&W,conf.rng);
     if(status != ML_OK) return status;
     break;
   }
   case FILL_ONES: {
     status = MatFillScalar(&W,1.0f);
     if(status != ML_OK) return status;
     break; 
   }
   case FILL_ZEROS: {
     status = MatFillScalar(&W,0.0f);
     if(status != ML_OK) return status;
     break; 
   }
   default:
     return ML_UNIMPLEMENTED;
  }

  //Init bias
  switch (conf.fillb_strat) {
   case FILL_XAVIER_UNIFORM: {
     status = Mat_xavier_uniform_dense(&b,conf.rng);
     if(status != ML_OK) return status;
     break;
   }
   case FILL_ONES: {
     status = MatFillScalar(&b,1.0f);
     if(status != ML_OK) return status;
     break; 
   }
   case FILL_ZEROS: {
     status = MatFillScalar(&b,0.0f);
     if(status != ML_OK) return status;
     break; 
   }
   default:
     return ML_UNIMPLEMENTED;
  }

  lin->W = W;
  lin->X = X;
  lin->b = b;
  lin->Z = Z;
  lin->X_T = X_T;
  lin->dW = dW;
  lin->db = db;

  return status;
}

ML_Status execute_op_Linear_forward(Linear *lin, Matf32 in) {
  if (!lin) return ML_INVALID_ARGUMENT;
  if (!in.data) return ML_INVALID_ARGUMENT;
  if (!lin->X.data || !lin->W.data || !lin->b.data || !lin->Z.data)
    return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  // Copy input into lin->X (since lin owns X)
  // This should be replaced by a MatCopy_into
  for (u64 r = 0; r < in.rows; ++r) {
    for (u64 c = 0; c < in.cols; ++c) {
      f32 v = 0.0f;
      status = MatGet(in, r, c, &v);
      if (status != ML_OK) return status;

      status = MatSet(&lin->X, r, c, v);
      if (status != ML_OK) return status;
    }
  }

  status = Mat_Mul_Mat_into(&lin->Z,lin->X,lin->W);
  if (status != ML_OK) return status;

  status = Mat_rowwise_add_RowVec_inplace(&lin->Z,lin->b);
  if (status != ML_OK) return status;

  return status;
}

ML_Status execute_op_Linear_backward(Linear* lin, const Matf32 dZ) {
  if (!lin) return ML_INVALID_ARGUMENT;
  if (!dZ.data) return ML_INVALID_ARGUMENT;

  if (!lin->X.data || !lin->W.data || !lin->b.data ||
      !lin->dW.data || !lin->db.data || !lin->X_T.data)
    return ML_INVALID_ARGUMENT;

  // Shape checks:
  // X: (N×D)
  // dZ: (N×C)
  // dW: (D×C)
  // db: (1×C)
  if (dZ.rows != lin->X.rows) return ML_INVALID_ARGUMENT;
  if (dZ.cols != lin->W.cols) return ML_INVALID_ARGUMENT;

  if (lin->dW.rows != lin->W.rows || lin->dW.cols != lin->W.cols)
    return ML_INVALID_ARGUMENT;
  if (lin->db.rows != 1 || lin->db.cols != lin->W.cols) return ML_INVALID_ARGUMENT;

  if (lin->X_T.rows != lin->X.cols || lin->X_T.cols != lin->X.rows)
    return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  // Transpose X
  status = Mat_transpose_into(&lin->X_T, lin->X);
  if (status != ML_OK) return status;

  // dW = X_T * dZ
  status = Mat_Mul_Mat_into(&lin->dW, lin->X_T, dZ);
  if (status != ML_OK) return status;

  // db = colsum(dZ) into (1×C)
  status = Mat_colsum_into_rowvec(&lin->db, dZ);
  if (status != ML_OK) return status;

  return ML_OK;
}

ML_Status execute_op_Linear_sgd_step(Linear* lin, f32 lr) {
  if (!lin) return ML_INVALID_ARGUMENT;
  if (!lin->W.data || !lin->b.data || !lin->dW.data || !lin->db.data)
    return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  status = Mat_SGD_inplace(&lin->W, lin->dW, lr);
  if (status != ML_OK) return status;

  status = Mat_SGD_inplace(&lin->b, lin->db, lr);
  if (status != ML_OK) return status;

  return ML_OK;
}

ML_Status create_config_Softmax(SoftmaxConfig *conf, u64 inrows, u64 incols) {
  if (!conf) return ML_INVALID_ARGUMENT;
  if (inrows == 0 || incols == 0) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;
  conf->in_rows = inrows;
  conf->in_cols = incols;
  
  return status;
}

ML_Status create_op_Softmax(ml_arena* arena, Softmax* softmax, SoftmaxConfig conf) {
  if (!arena || !softmax) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  // Allocate rowmax: (N x 1)
  status = create_Mat(arena, &softmax->rowmax, conf.in_rows, 1);
  if (status != ML_OK) return status;

  // Allocate rowsum: (N x 1)
  status = create_Mat(arena, &softmax->rowsum, conf.in_rows, 1);
  if (status != ML_OK) return status;

  // Allocate P: (N x C)
  status = create_Mat(arena, &softmax->P, conf.in_rows, conf.in_cols);
  if (status != ML_OK) return status;

  return ML_OK;
}

 ML_Status execute_op_Softmax_forward(Softmax* sm, const Matf32 Z) {
  if (!sm) return ML_INVALID_ARGUMENT;
  if (!Z.data) return ML_INVALID_ARGUMENT;

  if (!sm->P.data || !sm->rowmax.data || !sm->rowsum.data)
    return ML_INVALID_ARGUMENT;

  // Shape checks
  if (Z.rows != sm->P.rows || Z.cols != sm->P.cols) return ML_INVALID_ARGUMENT;
  if (sm->rowmax.rows != Z.rows || sm->rowmax.cols != 1) return ML_INVALID_ARGUMENT;
  if (sm->rowsum.rows != Z.rows || sm->rowsum.cols != 1) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  // 1) P <- Z (workspace/output)
  for (u64 r = 0; r < Z.rows; ++r) {
    for (u64 c = 0; c < Z.cols; ++c) {
      f32 v = 0.0f;
      status = MatGet(Z, r, c, &v);
      if (status != ML_OK) return status;

      status = MatSet(&sm->P, r, c, v);
      if (status != ML_OK) return status;
    }
  }

  // 2) rowmax <- rowmax(P)
  status = Mat_rowmax_into(&sm->rowmax, sm->P);
  if (status != ML_OK) return status;

  // 3) P <- P - rowmax   (broadcast N×1)
  status = Mat_rowwise_sub_ColVec_inplace(&sm->P, sm->rowmax);
  if (status != ML_OK) return status;

  // 4) P <- exp(P)
  status = Mat_exp_inplace(&sm->P);
  if (status != ML_OK) return status;

  // 5) rowsum <- rowsum(P)
  status = Mat_rowsum_into(&sm->rowsum, sm->P);
  if (status != ML_OK) return status;

  // 6) P <- P / rowsum   (broadcast N×1)
  status = Mat_rowwise_div_ColVec_inplace(&sm->P, sm->rowsum);
  if (status != ML_OK) return status;

  return ML_OK;
 }

ML_Status create_config_CrossEntropy(CEConfig* conf, u64 inrows, u64 incols) {
  if (!conf) return ML_INVALID_ARGUMENT;
  if (inrows == 0 || incols == 0) return ML_INVALID_ARGUMENT;

  conf->in_rows = inrows;
  conf->in_cols = incols;
  return ML_OK;
}

ML_Status create_op_CrossEntropy(ml_arena* arena, CrossEntropy* ce, CEConfig conf) {
  if (!arena || !ce) return ML_INVALID_ARGUMENT;

  ML_Status status =
    create_Mat(arena, &ce->dZ, conf.in_rows, conf.in_cols);
  if (status != ML_OK) return status;

  ce->loss = 0.0f;
  return ML_OK;
}

ML_Status execute_op_CrossEntropy_forward(CrossEntropy *ce, const Matf32 P,
                                          const Matf32 Y) {
  if (!ce) return ML_INVALID_ARGUMENT;
  if (!P.data || !Y.data) return ML_INVALID_ARGUMENT;

  // Shapes must match and match ce->dZ
  if (P.rows != Y.rows || P.cols != Y.cols) return ML_INVALID_ARGUMENT;
  if (ce->dZ.rows != P.rows || ce->dZ.cols != P.cols) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  const f32 eps = 1e-12f; // avoid log(0)
  f32 acc = 0.0f;

  for (u64 r = 0; r < P.rows; ++r) {
    for (u64 c = 0; c < P.cols; ++c) {
      f32 y = 0.0f;
      status = MatGet(Y, r, c, &y);
      if (status != ML_OK) return status;

      // Skip zeros (common for one-hot)
      if (y == 0.0f) continue;

      f32 p = 0.0f;
      status = MatGet(P, r, c, &p);
      if (status != ML_OK) return status;

      if (p < eps) p = eps;
      acc += -y * logf(p);
    }
  }

  ce->loss = acc / (f32)P.rows; // mean over samples
  return ML_OK;
}

ML_Status execute_op_CrossEntropy_backward(CrossEntropy *ce, const Matf32 P,
                                           const Matf32 Y) {
  if (!ce) return ML_INVALID_ARGUMENT;
  if (!P.data || !Y.data) return ML_INVALID_ARGUMENT;

  if (P.rows != Y.rows || P.cols != Y.cols) return ML_INVALID_ARGUMENT;
  if (!ce->dZ.data) return ML_INVALID_ARGUMENT;
  if (ce->dZ.rows != P.rows || ce->dZ.cols != P.cols) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  // dZ = P - Y
  for (u64 r = 0; r < P.rows; ++r) {
    for (u64 c = 0; c < P.cols; ++c) {
      f32 p = 0.0f, y = 0.0f;
      status = MatGet(P, r, c, &p);
      if (status != ML_OK) return status;
      status = MatGet(Y, r, c, &y);
      if (status != ML_OK) return status;

      status = MatSet(&ce->dZ, r, c, p - y);
      if (status != ML_OK) return status;
    }
  }

  // dZ *= 1/N
  f32 invN = 1.0f / (f32)P.rows;
  status = Mat_Scale_inplace(&ce->dZ, invN);
  if (status != ML_OK) return status;

  return ML_OK;
}

