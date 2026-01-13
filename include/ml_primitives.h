#ifndef ML_PRIMITIVES_H
#define ML_PRIMITIVES_H

#include "ml_error.h"
#include "ml_alloc.h"

/**
 * @file ml_primitives.h
 * @brief Dense f32 matrix primitives used by the ML operators.
 *
 * Matrices are dense, row-major, contiguous buffers of 32-bit floats.
 *
 * Naming conventions used in this API:
 * - *_inplace: modifies the first matrix argument in-place.
 * - *_into: writes results into a preallocated output matrix (no allocation).
 * - functions without *_into that take an arena typically allocate the output.
 *
 * Shape errors and null pointers return ML_INVALID_ARGUMENT.
 */

/**
 * @brief Dense matrix of 32-bit floats.
 *
 * Storage is row-major:
 * element (r,c) is stored at data[r*cols + c].
 *
 * @note This type does not own memory; it points into an arena allocation.
 */
typedef struct {
  /** Number of rows. */
  u64 rows;
  /** Number of columns. */
  u64 cols;
  /** Row-major contiguous storage (rows*cols elements). */
  f32* data;
} Matf32;

/**
 * @brief Allocate a matrix of shape (rows x cols) in the given arena.
 *
 * The allocated buffer contains uninitialized values.
 *
 * @param arena Arena used for allocation.
 * @param dest Output matrix descriptor to initialize.
 * @param rows Number of rows (may be 0 if you want an empty matrix, but
 *        current implementation treats capacity/size normally).
 * @param cols Number of cols.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p arena or @p dest is NULL.
 * @return ML_OUT_OF_MEMORY if the arena cannot satisfy the allocation.
 */
ML_Status create_Mat(ml_arena* arena, Matf32* dest, u64 rows, u64 cols);

/**
 * @brief Fill an entire matrix with a scalar value.
 *
 * @param target Matrix to write into (in-place).
 * @param val Scalar to assign to each element.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p target is NULL or target->data is NULL.
 */
ML_Status MatFillScalar(Matf32* target, f32 val);

/**
 * @brief Fill a specific row with values from a contiguous array.
 *
 * @param target Matrix to write into.
 * @param row Row index in [0, target->rows).
 * @param val Pointer to an array of length target->cols.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if any pointer is NULL or row is out of range.
 */
ML_Status MatFillRow(Matf32* target, u64 row, const f32* val);

/**
 * @brief Read one element from a matrix.
 *
 * @param target Matrix to read from.
 * @param row Row index.
 * @param col Column index.
 * @param val Output location for the element.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if target.data is NULL, @p val is NULL,
 *         or indices are out of range.
 */
ML_Status MatGet(const Matf32 target, u64 row, u64 col, f32* val);

/**
 * @brief Write one element to a matrix.
 *
 * @param target Matrix to write into.
 * @param row Row index.
 * @param col Column index.
 * @param val Value to store.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p target is NULL, target->data is NULL,
 *         or indices are out of range.
 */
ML_Status MatSet(Matf32* target, u64 row, u64 col, const f32 val);

/**
 * @brief Allocate and copy a matrix into a new arena-backed matrix.
 *
 * Equivalent to:
 * 1) create_Mat(arena, dest, src.rows, src.cols)
 * 2) element-wise copy
 *
 * @param src Source matrix.
 * @param dest Output matrix descriptor to allocate and fill.
 * @param arena Arena used for allocation.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p dest or @p arena is NULL.
 * @return ML_OUT_OF_MEMORY if allocation fails.
 */
ML_Status MatCopy(Matf32 src, Matf32* dest, ml_arena* arena);

/**
 * @brief Copy a matrix into an existing, preallocated destination.
 *
 * @param dest Destination matrix (must be allocated and same shape as src).
 * @param src Source matrix.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes differ.
 */
ML_Status MatCopy_into(Matf32* dest, const Matf32 src);

/* -------------------------------------------------------------------------- */
/* Linear algebra operations                                                   */
/* -------------------------------------------------------------------------- */

/**
 * @brief Matrix multiplication with allocation: out = lhs * rhs.
 *
 * Allocates @p out in @p arena with shape (lhs.rows x rhs.cols).
 *
 * @param out Output matrix descriptor to allocate and fill.
 * @param arena Arena used for allocation.
 * @param lhs Left operand (m x k).
 * @param rhs Right operand (k x n).
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or lhs.cols != rhs.rows.
 * @return ML_OUT_OF_MEMORY if output allocation fails.
 */
ML_Status Mat_Mul_Mat(Matf32* out, ml_arena* arena, const Matf32 lhs, const Matf32 rhs);

/**
 * @brief Matrix multiplication into preallocated output: out = lhs * rhs.
 *
 * Requires:
 * - out is allocated with shape (lhs.rows x rhs.cols)
 * - lhs.cols == rhs.rows
 *
 * @param out Preallocated output matrix.
 * @param lhs Left operand.
 * @param rhs Right operand.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes are incompatible.
 */
ML_Status Mat_Mul_Mat_into(Matf32* out, const Matf32 lhs, const Matf32 rhs);

/**
 * @brief Transpose with allocation: out = target^T.
 *
 * Allocates @p out in @p arena with shape (target.cols x target.rows).
 *
 * @param out Output matrix descriptor to allocate and fill.
 * @param arena Arena used for allocation.
 * @param target Matrix to transpose.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL.
 * @return ML_OUT_OF_MEMORY if output allocation fails.
 */
ML_Status Mat_transpose(Matf32* out, ml_arena* arena, const Matf32 target);

/**
 * @brief Transpose into preallocated output: out = target^T.
 *
 * Requires out shape (target.cols x target.rows).
 *
 * @param out Preallocated output matrix.
 * @param target Matrix to transpose.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes mismatch.
 */
ML_Status Mat_transpose_into(Matf32* out, const Matf32 target);

/* -------------------------------------------------------------------------- */
/* Elementwise / reduction operations                                          */
/* -------------------------------------------------------------------------- */

/**
 * @brief In-place subtract a scalar: lhs[i] -= scalar.
 *
 * @param lhs Matrix to modify in-place.
 * @param scalar Scalar value to subtract.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p lhs is NULL or lhs->data is NULL.
 */
ML_Status Mat_Sub_Scalar(Matf32* lhs, f32 scalar);

/**
 * @brief In-place scale: lhs[i] *= s.
 *
 * @param lhs Matrix to modify in-place.
 * @param s Scalar multiplier.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p lhs is NULL or lhs->data is NULL.
 */
ML_Status Mat_Scale_inplace(Matf32* lhs, f32 s);

/**
 * @brief Column-wise sum into a row vector: out[0,c] = sum_r A[r,c].
 *
 * Requires out shape (1 x A.cols).
 *
 * @param out Preallocated output row vector.
 * @param A Input matrix.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes mismatch.
 */
ML_Status Mat_colsum_into_rowvec(Matf32* out, const Matf32 A);

/**
 * @brief In-place SGD update: param -= lr * grad.
 *
 * Shapes must match.
 *
 * @param param Parameter matrix to update in-place.
 * @param grad Gradient matrix.
 * @param lr Learning rate.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes mismatch.
 */
ML_Status Mat_SGD_inplace(Matf32* param, const Matf32 grad, f32 lr);

/**
 * @brief Allocate row-wise max vector: out[r,0] = max_c Z[r,c].
 *
 * Allocates out with shape (Z.rows x 1).
 *
 * @param out Output column vector to allocate and fill.
 * @param arena Arena used for allocation.
 * @param Z Input matrix.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL.
 * @return ML_OUT_OF_MEMORY if output allocation fails.
 */
ML_Status Mat_rowmax(Matf32* out, ml_arena* arena, const Matf32 Z);

/**
 * @brief Row-wise max into preallocated output: out[r,0] = max_c Z[r,c].
 *
 * Requires out shape (Z.rows x 1).
 *
 * @param out Preallocated output column vector.
 * @param Z Input matrix.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes mismatch.
 */
ML_Status Mat_rowmax_into(Matf32* out, const Matf32 Z);

/**
 * @brief Allocate row-wise sum vector: out[r,0] = sum_c A[r,c].
 *
 * Allocates out with shape (A.rows x 1).
 *
 * @param out Output column vector to allocate and fill.
 * @param arena Arena used for allocation.
 * @param A Input matrix.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL.
 * @return ML_OUT_OF_MEMORY if output allocation fails.
 */
ML_Status Mat_rowsum(Matf32* out, ml_arena* arena, const Matf32 A);

/**
 * @brief Row-wise sum into preallocated output: out[r,0] = sum_c A[r,c].
 *
 * Requires out shape (A.rows x 1).
 *
 * @param out Preallocated output column vector.
 * @param A Input matrix.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes mismatch.
 */
ML_Status Mat_rowsum_into(Matf32* out, const Matf32 A);

/**
 * @brief Get max of a single row in a matrix.
 *
 * Writes the maximum across columns for the given row into @p val.
 *
 * @param target Input matrix.
 * @param row Row index.
 * @param val Output location for the maximum.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or indices out of range.
 */
ML_Status Mat_get_rowmax(const Matf32 target, u64 row, f32* val);

/**
 * @brief In-place broadcast add of a row vector: lhs[r,c] += rhs[0,c].
 *
 * Requires:
 * - rhs.rows == 1
 * - rhs.cols == lhs.cols
 *
 * @param lhs Matrix to modify in-place.
 * @param rhs Row vector (1 x C).
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes mismatch.
 */
ML_Status Mat_rowwise_add_RowVec_inplace(Matf32* lhs, Matf32 rhs);

/**
 * @brief In-place broadcast subtract of a column vector: lhs[r,c] -= rhs[r,0].
 *
 * Requires rhs shape (lhs.rows x 1).
 *
 * @param lhs Matrix to modify in-place.
 * @param rhs Column vector (N x 1).
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes mismatch.
 */
ML_Status Mat_rowwise_sub_ColVec_inplace(Matf32* lhs, Matf32 rhs);

/**
 * @brief In-place broadcast divide by a column vector: lhs[r,c] /= rhs[r,0].
 *
 * Requires rhs shape (lhs.rows x 1).
 *
 * @param lhs Matrix to modify in-place.
 * @param rhs Column vector (N x 1).
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if pointers are NULL or shapes mismatch.
 *
 * @note Current implementation returns ML_INVALID_ARGUMENT if any rhs element is 0.
 */
ML_Status Mat_rowwise_div_ColVec_inplace(Matf32* lhs, Matf32 rhs);

/**
 * @brief In-place elementwise exponential: target[i] = exp(target[i]).
 *
 * @param target Matrix to modify in-place.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p target is NULL or target->data is NULL.
 */
ML_Status Mat_exp_inplace(Matf32* target);

#endif // ML_PRIMITIVES_H
