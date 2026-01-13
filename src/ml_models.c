#include "ml_models.h"

#include "ml_models.h"
#include "ml_alloc.h"
#include "ml_primitives.h"
#include "ml_error.h"

ML_Status create_config_SoftmaxRegression(SoftmaxRegressionConfig* conf,
                                         u64 N, u64 D, u64 C,
                                         ML_Rng* rng,
                                         FillStrategy w_init,
                                         FillStrategy b_init) {
  if (!conf) return ML_INVALID_ARGUMENT;
  if (N == 0 || D == 0 || C == 0) return ML_INVALID_ARGUMENT;

  conf->N = N;
  conf->D = D;
  conf->C = C;
  conf->rng = rng;
  conf->w_init = w_init;
  conf->b_init = b_init;

  return ML_OK;
}

ML_Status create_model_SoftmaxRegression(ml_arena* arena,
                                        SoftmaxRegression* m,
                                        SoftmaxRegressionConfig conf) {
  if (!arena || !m) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  // Store config in the model
  m->conf = conf;

  // ---- Linear ----
  LinearConfig lconf;
  status = create_config_Linear(&lconf,
                                conf.N,     // in_rows (batch)
                                conf.D,     // in_cols (features)
                                conf.C,     // out_cols (classes)
                                conf.rng,
                                conf.w_init,
                                conf.b_init);
  if (status != ML_OK) return status;

  status = create_op_Linear(arena, &m->lin, lconf);
  if (status != ML_OK) return status;

  // ---- Softmax ----
  SoftmaxConfig sconf;
  status = create_config_Softmax(&sconf, conf.N, conf.C);
  if (status != ML_OK) return status;

  status = create_op_Softmax(arena, &m->sm, sconf);
  if (status != ML_OK) return status;

  // ---- CrossEntropy ----
  // Assumes you have CEConfig + create_config_CrossEntropy + create_op_CrossEntropy
  CEConfig ceconf;
  status = create_config_CrossEntropy(&ceconf, conf.N, conf.C);
  if (status != ML_OK) return status;

  status = create_op_CrossEntropy(arena, &m->ce, ceconf);
  if (status != ML_OK) return status;

  return ML_OK;
}

ML_Status infer_SoftmaxRegression(SoftmaxRegression* m,
                                 const Matf32 X,
                                 Matf32* outP) {
  if (!m || !outP) return ML_INVALID_ARGUMENT;
  if (!X.data || !outP->data) return ML_INVALID_ARGUMENT;

  // Shape checks
  if (X.rows != m->conf.N || X.cols != m->conf.D) return ML_INVALID_ARGUMENT;
  if (outP->rows != m->conf.N || outP->cols != m->conf.C) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  status = execute_op_Linear_forward(&m->lin, X);
  if (status != ML_OK) return status;

  status = execute_op_Softmax_forward(&m->sm, m->lin.Z);
  if (status != ML_OK) return status;

  // Copy probabilities out
  status = MatCopy_into(outP, m->sm.P);
  if (status != ML_OK) return status;

  return ML_OK;
}

ML_Status train_step_SoftmaxRegression(SoftmaxRegression* m,
                                      const Matf32 X,
                                      const Matf32 Y,
                                      f32 lr,
                                      f32* out_loss) {
  if (!m || !out_loss) return ML_INVALID_ARGUMENT;
  if (!X.data || !Y.data) return ML_INVALID_ARGUMENT;

  if (X.rows != m->conf.N || X.cols != m->conf.D) return ML_INVALID_ARGUMENT;
  if (Y.rows != m->conf.N || Y.cols != m->conf.C) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  // Forward
  status = execute_op_Linear_forward(&m->lin, X);
  if (status != ML_OK) return status;

  status = execute_op_Softmax_forward(&m->sm, m->lin.Z);
  if (status != ML_OK) return status;

  status = execute_op_CrossEntropy_forward(&m->ce, m->sm.P, Y);
  if (status != ML_OK) return status;

  // Backward
  status = execute_op_CrossEntropy_backward(&m->ce, m->sm.P, Y);
  if (status != ML_OK) return status;

  status = execute_op_Linear_backward(&m->lin, m->ce.dZ);
  if (status != ML_OK) return status;

  // SGD update
  status = execute_op_Linear_sgd_step(&m->lin, lr);
  if (status != ML_OK) return status;

  *out_loss = m->ce.loss;
  return ML_OK;
}

ML_Status train_SoftmaxRegression(SoftmaxRegression* m,
                                 ML_BatchProvider provider,
                                 ML_TrainConfig tconf,
                                 Matf32* Xbuf,
                                 Matf32* Ybuf,
                                 f32* out_last_loss) {
  if (!m || !provider.next_batch || !Xbuf || !Ybuf || !out_last_loss)
    return ML_INVALID_ARGUMENT;

  // Check buffer shapes match model
  if (!Xbuf->data || Xbuf->rows != m->conf.N || Xbuf->cols != m->conf.D)
    return ML_INVALID_ARGUMENT;
  if (!Ybuf->data || Ybuf->rows != m->conf.N || Ybuf->cols != m->conf.C)
    return ML_INVALID_ARGUMENT;

  if (tconf.epochs == 0) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;
  f32 last_loss = 0.0f;
  u64 global_step = 0;

  for (u64 epoch = 0; epoch < tconf.epochs; ++epoch) {

    // Consume batches until provider says epoch is done
    while (1) {
      status = provider.next_batch(provider.ctx, Xbuf, Ybuf);

      if (status == ML_DONE) {
        // end of epoch
        break;
      }
      if (status != ML_OK) {
        return status;
      }

      status = train_step_SoftmaxRegression(m, *Xbuf, *Ybuf, tconf.lr, &last_loss);
      if (status != ML_OK) return status;

      ++global_step;

      // optional logging hook point; on ESP32 you might printf here
      // but keep library silent by default
      // user can wrap provider/loop externally if they want printing
      (void)epoch;
      if (tconf.log_every != 0 && (global_step % tconf.log_every) == 0) {
        // no-op: leave to user; or provide callback below
      }
    }
  }

  *out_last_loss = last_loss;
  return ML_OK;
}
