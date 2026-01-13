#include "ml_primitives.h"
#include "ml_alloc.h"
#include "ml_error.h"

#include <stddef.h>
#include <math.h>

 ML_Status create_Mat(ml_arena *arena, Matf32* dest, u64 rows, u64 cols) {
  if(!arena || !dest) return ML_INVALID_ARGUMENT;
  ML_Status status = ML_OK;
  void* data_ptr = NULL;
  status = push_ml_arena(&data_ptr,arena,rows*cols*sizeof(f32));

  if(status != ML_OK ) return status;

  dest->cols = cols;
  dest->rows = rows;
  dest->data = data_ptr;

  return status;
 }

 ML_Status MatFillScalar(Matf32 *target, f32 val) {
   if(!target) return ML_INVALID_ARGUMENT;
   if(!target->data) return ML_INVALID_ARGUMENT;

   ML_Status status = ML_OK;

   for (u64 r = 0; r < target->rows; ++r) {
    for (u64 c = 0; c < target->cols; ++c) {
      status = MatSet(target,r,c,val);
      if(status != ML_OK ) return status;
    }  
   }

   return status;
 }

 ML_Status MatFillRow(Matf32 *target, u64 row, const f32 *val) {
   if(!target) return ML_INVALID_ARGUMENT;
   if(!target->data) return ML_INVALID_ARGUMENT;
   if(row >= target->rows) return ML_INVALID_ARGUMENT;
   if(!val) return ML_INVALID_ARGUMENT;
   
   ML_Status status = ML_OK;

   for (u64 col = 0; col < target->cols; ++col) {
     status = MatSet(target,row,col,val[col]);
     if(status != ML_OK ) return status;
   }
   
   return status;
 }

 ML_Status MatGet(const Matf32 target, u64 row, u64 col, f32 *val) {
   if(!(target.data)) return ML_INVALID_ARGUMENT;
   if(!val) return ML_INVALID_ARGUMENT;
   if(row >= target.rows) return ML_INVALID_ARGUMENT;
   if(col >= target.cols) return ML_INVALID_ARGUMENT;

   ML_Status status = ML_OK; 

   *val = target.data[row*target.cols + col];

   return status;
 }

 ML_Status MatSet(Matf32 *target, u64 row, u64 col, const f32 val) { 
   if(!target) return ML_INVALID_ARGUMENT;
   if(!target->data) return ML_INVALID_ARGUMENT;
   if(row >= target->rows) return ML_INVALID_ARGUMENT;
   if(col >= target->cols) return ML_INVALID_ARGUMENT;

   ML_Status status = ML_OK; 

   target->data[row*target->cols + col] = val;

   return status;
 }

 ML_Status MatCopy(Matf32 src, Matf32 *dest, ml_arena *arena) {
   if(!dest || !arena) return ML_INVALID_ARGUMENT;
   ML_Status status = ML_OK;

   status = create_Mat(arena,dest,src.rows,src.cols);
   if(status != ML_OK) return status;

   for (u64 src_row = 0; src_row < src.rows; ++src_row) {
     for (u64 src_col = 0; src_col < src.cols; ++src_col) {
       f32 cur_val = 0.0f;
       status = MatGet(src,src_row,src_col,&cur_val);
       if(status != ML_OK) return status;
       status = MatSet(dest,src_row,src_col,cur_val);
       if(status != ML_OK) return status;
     }
   }

   return status;
 }

 ML_Status MatCopy_into(Matf32* dest, const Matf32 src) {
  if (!dest) return ML_INVALID_ARGUMENT;
  if (!dest->data || !src.data) return ML_INVALID_ARGUMENT;

  if (dest->rows != src.rows) return ML_INVALID_ARGUMENT;
  if (dest->cols != src.cols) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  for (u64 r = 0; r < src.rows; ++r) {
    for (u64 c = 0; c < src.cols; ++c) {
      f32 v = 0.0f;
      status = MatGet(src, r, c, &v);
      if (status != ML_OK) return status;

      status = MatSet(dest, r, c, v);
      if (status != ML_OK) return status;
    }
  }

  return ML_OK;
 }

 ML_Status Mat_Mul_Mat(Matf32 *out, ml_arena *arena,
                       const Matf32 lhs, const Matf32 rhs) {
   if( !lhs.data || !rhs.data) return ML_INVALID_ARGUMENT;
   if(lhs.cols != rhs.rows) return ML_INVALID_ARGUMENT;
   if(!out || !arena) return ML_INVALID_ARGUMENT;
   ML_Status status = ML_OK;

   status = create_Mat(arena,out,lhs.rows,rhs.cols);
   if(status != ML_OK) return status;

   u64 m = lhs.rows;
   u64 k = lhs.cols; // == rhs.rows
   u64 n = rhs.cols;

   for (u64 i = 0; i < m; ++i) {
     for (u64 j = 0; j < n; ++j) {
       f32 sum = 0.0f;
       for (u64 t = 0; t < k; ++t) {
	 f32 a;
	 status = MatGet(lhs,i,t,&a);
	 if(status != ML_OK) return status;
	 f32 b;
	 status = MatGet(rhs,t,j,&b);
	 if(status != ML_OK) return status;
	 sum += a * b;
       }
       status = MatSet(out,i,j,sum);
       if(status != ML_OK) return status;
     }
   }

   return status;
 }

ML_Status Mat_Mul_Mat_into(Matf32 *out, const Matf32 lhs, const Matf32 rhs) {
  if (!out) return ML_INVALID_ARGUMENT;
  if (!out->data || !lhs.data || !rhs.data) return ML_INVALID_ARGUMENT;

  if (lhs.cols != rhs.rows) return ML_INVALID_ARGUMENT;

  // out must already be allocated with correct shape
  if (out->rows != lhs.rows) return ML_INVALID_ARGUMENT;
  if (out->cols != rhs.cols) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  u64 m = lhs.rows;
  u64 k = lhs.cols;
  u64 n = rhs.cols;

  for (u64 i = 0; i < m; ++i) {
    for (u64 j = 0; j < n; ++j) {
      f32 sum = 0.0f;
      for (u64 t = 0; t < k; ++t) {
        f32 a = 0.0f, b = 0.0f;
        status = MatGet(lhs, i, t, &a);
        if (status != ML_OK) return status;
        status = MatGet(rhs, t, j, &b);
        if (status != ML_OK) return status;
        sum += a * b;
      }
      status = MatSet(out, i, j, sum);
      if (status != ML_OK) return status;
    }
  }

  return ML_OK;
}


 ML_Status Mat_transpose(Matf32 *out, ml_arena *arena, const Matf32 target) {
   if(!out || !arena) return ML_INVALID_ARGUMENT;
   if(!target.data) return ML_INVALID_ARGUMENT;

   ML_Status status = ML_OK;

   status = create_Mat(arena,out,target.cols,target.rows);
   if (status != ML_OK) return status;

   for (u64 r = 0; r < target.rows; ++r) {
     for (u64 c = 0; c < target.cols; ++c) {
       f32 v = 0.0f; 

       status = MatGet(target,r,c,&v);
       if (status != ML_OK) return status;

       status = MatSet(out,c,r,v);
       if (status != ML_OK) return status; 
     }
   }

   return status;
 }

 ML_Status Mat_transpose_into(Matf32* out, const Matf32 target) {
  if (!out) return ML_INVALID_ARGUMENT;
  if (!out->data || !target.data) return ML_INVALID_ARGUMENT;

  // out must already be allocated as (target.cols x target.rows)
  if (out->rows != target.cols) return ML_INVALID_ARGUMENT;
  if (out->cols != target.rows) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  for (u64 r = 0; r < target.rows; ++r) {
    for (u64 c = 0; c < target.cols; ++c) {
      f32 v = 0.0f;
      status = MatGet(target, r, c, &v);
      if (status != ML_OK) return status;

      status = MatSet(out, c, r, v);
      if (status != ML_OK) return status;
    }
  }

  return ML_OK;
}

 ML_Status Mat_rowwise_add_RowVec_inplace(Matf32 *lhs, Matf32 rhs) {
   if(!lhs) return ML_INVALID_ARGUMENT;
   if(!lhs->data) return ML_INVALID_ARGUMENT;
   if(!rhs.data) return ML_INVALID_ARGUMENT;
   if(rhs.cols != lhs->cols) return ML_INVALID_ARGUMENT;
   if(rhs.rows != 1) return ML_INVALID_ARGUMENT;

   ML_Status status = ML_OK;

   for (u64 i = 0; i < lhs->rows; ++i) {  
    for (u64 j = 0; j < lhs->cols; ++j) {
      f32 lhs_val = 0.0f;
      status = MatGet(*lhs,i,j,&lhs_val);
      if (status != ML_OK) return status; 

      f32 rhs_val = 0.0f;
      status = MatGet(rhs,0,j,&rhs_val);
      if (status != ML_OK) return status;

      status = MatSet(lhs,i,j,rhs_val + lhs_val);
      if (status != ML_OK) return status;
    }
   }

   return status;
 }

 ML_Status Mat_rowwise_sub_ColVec_inplace(Matf32 *lhs, Matf32 rhs) {
   if(!lhs) return ML_INVALID_ARGUMENT;
   if(!lhs->data || !rhs.data) return ML_INVALID_ARGUMENT;
   if(lhs->rows != rhs.rows) return ML_INVALID_ARGUMENT;
   if(rhs.cols != 1) return ML_INVALID_ARGUMENT;

   ML_Status status = ML_OK;

   for (u64 r = 0; r < lhs->rows; ++r) {
     f32 sub = 0.0f;
     status = MatGet(rhs,r,0,&sub);
     if (status != ML_OK) return status;
     for (u64 c = 0; c < lhs->cols; ++c) {
       f32 v = 0.0f;
       status = MatGet(*lhs,r,c,&v);
       if (status != ML_OK) return status;
       status = MatSet(lhs,r,c,v-sub);
       if (status != ML_OK) return status;
     }
   }

   return status;
 }

 ML_Status Mat_rowmax(Matf32 *out, ml_arena *arena, const Matf32 Z) {
   if (!out || !arena || !Z.data) return ML_INVALID_ARGUMENT;

   ML_Status status = ML_OK;
   status = create_Mat(arena,out,Z.rows,1);
   if (status != ML_OK) return status;

   f32 maxv = 0.0f;
   for (u64 r = 0; r < out->rows; ++r) {
     status = Mat_get_rowmax(Z,r,&maxv);
     if (status != ML_OK) return status;
     status = MatSet(out,r,0,maxv);
     if (status != ML_OK) return status;
   }
     
   return status;
 }

 ML_Status Mat_rowmax_into(Matf32* out, const Matf32 Z) {
  if (!out) return ML_INVALID_ARGUMENT;
  if (!out->data || !Z.data) return ML_INVALID_ARGUMENT;

  // out must be (N x 1)
  if (out->rows != Z.rows) return ML_INVALID_ARGUMENT;
  if (out->cols != 1) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  for (u64 r = 0; r < Z.rows; ++r) {
    f32 maxv = 0.0f;
    status = Mat_get_rowmax(Z, r, &maxv);
    if (status != ML_OK) return status;

    status = MatSet(out, r, 0, maxv);
    if (status != ML_OK) return status;
  }

  return ML_OK;
 }

 ML_Status Mat_rowsum_into(Matf32* out, const Matf32 A) {
  if (!out) return ML_INVALID_ARGUMENT;
  if (!out->data || !A.data) return ML_INVALID_ARGUMENT;

  // out must be (N x 1)
  if (out->rows != A.rows) return ML_INVALID_ARGUMENT;
  if (out->cols != 1) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  for (u64 r = 0; r < A.rows; ++r) {
    f32 sum = 0.0f;

    for (u64 c = 0; c < A.cols; ++c) {
      f32 v = 0.0f;
      status = MatGet(A, r, c, &v);
      if (status != ML_OK) return status;
      sum += v;
    }

    status = MatSet(out, r, 0, sum);
    if (status != ML_OK) return status;
  }

  return ML_OK;
 }

 ML_Status Mat_get_rowmax(const Matf32 target, u64 row, f32 *val) {
   if(!val) return ML_INVALID_ARGUMENT;
   if(!target.data) return ML_INVALID_ARGUMENT;
   if (row >= target.rows) return ML_INVALID_ARGUMENT;

   ML_Status status = ML_OK;


   f32 maxv = 0.0f;
   status = MatGet(target,row,0,&maxv);
   if (status != ML_OK) return status;

   for (u64 col = 1; col < target.cols; ++col) {
     f32 temp = 0.0f;
     status = MatGet(target, row, col, &temp);
     if (status != ML_OK) return status;
     if (temp > maxv) maxv = temp;
   }

   *val = maxv;

   return status;
 }

 ML_Status Mat_exp_inplace(Matf32 *target) {
   if (!target || !target->data) return ML_INVALID_ARGUMENT;
   ML_Status status = ML_OK;

   for (u64 r = 0; r < target->rows; ++r) {  
    for (u64 c = 0; c < target->cols; ++c) {
      f32 val = 0.0f;
      status = MatGet(*target,r,c,&val);
      if (status != ML_OK) return status;
      f32 exp_val = expf(val);
      status = MatSet(target,r,c,exp_val);
      if (status != ML_OK) return status;
    }
   }
   return status;
 }

 ML_Status Mat_Sub_Scalar(Matf32 *lhs, f32 scalar) {
   if(!lhs) return ML_INVALID_ARGUMENT;
   ML_Status status = ML_OK;

   for (u64 r = 0;r < lhs->rows; ++r) {
    for (u64 c = 0;c < lhs->cols; ++c) {
      f32 val = 0.0f;
      status = MatGet(*lhs,r,c,&val);
      if (status != ML_OK) return status;
      f32 new_val = val - scalar;
      status = MatSet(lhs,r,c,new_val);
      if (status != ML_OK) return status;
    }
   }
   
   return status;
 }

 ML_Status Mat_Scale_inplace(Matf32* A, f32 s) {
  if (!A || !A->data) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  for (u64 r = 0; r < A->rows; ++r) {
    for (u64 c = 0; c < A->cols; ++c) {
      f32 v = 0.0f;
      status = MatGet(*A, r, c, &v);
      if (status != ML_OK) return status;

      status = MatSet(A, r, c, v * s);
      if (status != ML_OK) return status;
    }
  }

  return ML_OK;
 }

 ML_Status Mat_colsum_into_rowvec(Matf32* out, const Matf32 A) {
  if (!out) return ML_INVALID_ARGUMENT;
  if (!out->data || !A.data) return ML_INVALID_ARGUMENT;

  if (out->rows != 1) return ML_INVALID_ARGUMENT;
  if (out->cols != A.cols) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  for (u64 c = 0; c < A.cols; ++c) {
    f32 sum = 0.0f;

    for (u64 r = 0; r < A.rows; ++r) {
      f32 v = 0.0f;
      status = MatGet(A, r, c, &v);
      if (status != ML_OK) return status;
      sum += v;
    }

    status = MatSet(out, 0, c, sum);
    if (status != ML_OK) return status;
  }

  return ML_OK;
}

 ML_Status Mat_SGD_inplace(Matf32* param, const Matf32 grad, f32 lr) {
  if (!param) return ML_INVALID_ARGUMENT;
  if (!param->data || !grad.data) return ML_INVALID_ARGUMENT;

  if (param->rows != grad.rows) return ML_INVALID_ARGUMENT;
  if (param->cols != grad.cols) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  for (u64 r = 0; r < param->rows; ++r) {
    for (u64 c = 0; c < param->cols; ++c) {
      f32 p = 0.0f;
      status = MatGet(*param, r, c, &p);
      if (status != ML_OK) return status;

      f32 g = 0.0f;
      status = MatGet(grad, r, c, &g);
      if (status != ML_OK) return status;

      status = MatSet(param, r, c, p - lr * g);
      if (status != ML_OK) return status;
    }
  }

  return ML_OK;
}

 ML_Status Mat_rowsum(Matf32 *out, ml_arena *arena, const Matf32 target) {
  if (!out || !arena || !target.data) return ML_INVALID_ARGUMENT;

  ML_Status status = create_Mat(arena, out, target.rows, 1);
  if (status != ML_OK) return status;

  for (u64 r = 0; r < target.rows; ++r) {
    f32 sum = 0.0f;

    for (u64 c = 0; c < target.cols; ++c) {
      f32 v = 0.0f;
      status = MatGet(target, r, c, &v);
      if (status != ML_OK) return status;
      sum += v;
    }

    status = MatSet(out, r, 0, sum);
    if (status != ML_OK) return status;
  }

  return ML_OK;
 }

 ML_Status Mat_rowwise_div_ColVec_inplace(Matf32 *lhs, Matf32 rhs) {
  if (!lhs) return ML_INVALID_ARGUMENT;
  if (!lhs->data || !rhs.data) return ML_INVALID_ARGUMENT;
  if (lhs->rows != rhs.rows) return ML_INVALID_ARGUMENT;
  if (rhs.cols != 1) return ML_INVALID_ARGUMENT;

  ML_Status status = ML_OK;

  for (u64 r = 0; r < lhs->rows; ++r) {
    f32 denom = 0.0f;
    status = MatGet(rhs, r, 0, &denom);
    if (status != ML_OK) return status;

    // denom should be > 0
    if (denom == 0.0f) return ML_INVALID_ARGUMENT;

    for (u64 c = 0; c < lhs->cols; ++c) {
      f32 v = 0.0f;
      status = MatGet(*lhs, r, c, &v);
      if (status != ML_OK) return status;

      status = MatSet(lhs, r, c, v / denom);
      if (status != ML_OK) return status;
    }
  }

  return ML_OK;
}
