#ifndef ML_ALLOC_H
#define ML_ALLOC_H

#include "ml_defs.h"
#include "ml_error.h"

/**
 * @file ml_alloc.h
 * @brief Simple bump-pointer arena allocator with fixed backing storage.
 *
 * This module provides a minimal, deterministic allocator suitable for embedded
 * targets. Memory is provided by the caller as a single contiguous buffer.
 *
 * The arena:
 * - never frees individual allocations (bump allocator)
 * - performs aligned allocations (alignment is pointer-size)
 * - is O(1) per allocation
 *
 * Typical usage:
 * @code
 * static unsigned char mem[KiB(64)];
 * ml_arena arena;
 * ML_Status st = create_ml_arena(&arena, mem, sizeof(mem));
 * void* p = NULL;
 * st = push_ml_arena(&p, &arena, 128);
 * @endcode
 */

/** @brief Convert KiB to bytes (1024 * n). */
#define KiB(n) ( (u64)(n) << 10 )
/** @brief Convert MiB to bytes (1024^2 * n). */
#define MiB(n) ( (u64)(n) << 20 )
/** @brief Convert GiB to bytes (1024^3 * n). */
#define GiB(n) ( (u64)(n) << 30 )
/**
 * @brief Default arena alignment in bytes.
 *
 * All allocations are aligned to the size of a pointer, which is sufficient
 * for storing pointers and most scalar types on typical targets.
 */
#define ARENA_ALIGN (sizeof(void*))
/**
 * @brief Align @p n upward to the next multiple of @p p, where @p p is a power of two.
 *
 * @param n Value to align.
 * @param p Alignment (must be power of two).
 * @return Smallest value >= n that is a multiple of p.
 */
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

/**
 * @brief Bump-pointer arena allocator.
 *
 * @note The arena does not own the backing storage; the caller supplies it.
 */
typedef struct {
  /** Pointer to the start of the backing memory buffer. */
  void* base;
  /** Total capacity of the backing buffer in bytes. */
  u64 capacity;
  /** Current bump position (offset in bytes from @ref base). */
  u64  pos; 
} ml_arena;

/**
 * @brief Initialize an arena with caller-provided backing memory.
 *
 * @param target Output arena to initialize.
 * @param mem Backing memory buffer (must remain valid for the arena lifetime).
 * @param capacity Size of @p mem in bytes.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p target is NULL, @p mem is NULL, or @p capacity is 0.
 */
ML_Status create_ml_arena(ml_arena* target,void* mem,u64 capacity);

/**
 * @brief Allocate @p size bytes from the arena (bump allocation).
 *
 * Allocations are aligned to @ref ARENA_ALIGN.
 *
 * Behavior:
 * - On success, *@p ptr receives the allocated pointer and the arena position advances.
 * - If @p size == 0, *@p ptr is set to NULL and the function returns ML_OK.
 * - If there is not enough space, returns ML_OUT_OF_MEMORY and does not modify the arena.
 *
 * @param ptr Output pointer to receive the allocated address.
 * @param arena Arena to allocate from.
 * @param size Number of bytes to allocate.
 *
 * @return ML_OK on success.
 * @return ML_INVALID_ARGUMENT if @p ptr is NULL, or @p arena is NULL, or @p arena->base is NULL.
 * @return ML_OUT_OF_MEMORY if the arena does not have sufficient remaining capacity.
 */
ML_Status push_ml_arena(void** ptr,ml_arena* arena, u64 size);

/**
 * @brief Get the number of free bytes remaining in the arena.
 *
 * This is a simple capacity - pos calculation (no alignment padding is considered
 * for a *future* allocation).
 *
 * @param arena Arena to query.
 * @return Remaining free bytes, or 0 if @p arena is NULL or exhausted.
 */
u64 get_freemem_ml_arena_bytes(const ml_arena* arena);

#endif //ML_ALLOC_H
