#ifndef ML_PRIMITIVES_H
#define ML_PRIMITIVES_H

#include "ml_error.h"
#include "ml_alloc.h"

typedef struct {
  u64 rows; 
  u64 cols; 
  f32* data;
} Matf32;

ML_Status create_Mat(ml_arena* arena,Matf32* dest,u64 rows,u64 cols);

ML_Status MatFillRow(Matf32* target,u64 row,const f32* val);
ML_Status MatFillScalar(Matf32* target,f32 val);
ML_Status MatGet(const Matf32 target,u64 row,u64 col,f32* val);
ML_Status MatSet(Matf32* target,u64 row,u64 col,const f32 val);
ML_Status MatCopy(Matf32 src,Matf32* dest,ml_arena* arena);
ML_Status MatCopy_into(Matf32* dest, const Matf32 src);

// Linear Matrix Operations
ML_Status Mat_Mul_Mat(Matf32* out,ml_arena* arena,const Matf32 lhs,const Matf32 rhs);
ML_Status Mat_Mul_Mat_into(Matf32* out,const Matf32 lhs,const Matf32 rhs);
ML_Status Mat_transpose_into(Matf32* out,const Matf32 target);
ML_Status Mat_transpose(Matf32* out,ml_arena* arena,const Matf32 target);

// Non Linear Matrix Operations
ML_Status Mat_Sub_Scalar(Matf32* lhs,f32 scalar);
ML_Status Mat_Scale_inplace(Matf32* lhs,f32 s);
ML_Status Mat_colsum_into_rowvec(Matf32* out, const Matf32 A);
ML_Status Mat_SGD_inplace(Matf32* param, const Matf32 grad, f32 lr);
ML_Status Mat_rowmax(Matf32* out,ml_arena* arena,const Matf32 Z);
ML_Status Mat_rowmax_into(Matf32* out,const Matf32 Z);
ML_Status Mat_rowsum(Matf32 *out, ml_arena *arena, const Matf32 A);
ML_Status Mat_rowsum_into(Matf32 *out, const Matf32 A);
ML_Status Mat_get_rowmax(const Matf32 target,u64 row,f32* val);

ML_Status Mat_rowwise_add_RowVec_inplace(Matf32* lhs,Matf32 rhs);
ML_Status Mat_rowwise_sub_ColVec_inplace(Matf32* lhs,Matf32 rhs);
ML_Status Mat_rowwise_div_ColVec_inplace(Matf32 *lhs, Matf32 rhs);

ML_Status Mat_exp_inplace(Matf32* target);

#endif //ML_PRIMITIVES_H
