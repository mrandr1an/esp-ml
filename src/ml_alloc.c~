#include "ml_alloc.h"
#include <stdint.h>   // uintptr_t
#include <stddef.h>   // size_t

static inline u64 arena_align_up(u64 n) {
  return ALIGN_UP_POW2(n, (u64)ARENA_ALIGN);
}

ML_Status create_ml_arena(ml_arena* target, void* mem, u64 capacity) {
  if (!target) return ML_INVALID_ARGUMENT;
  if (!mem) return ML_INVALID_ARGUMENT;
  if (capacity == 0) return ML_INVALID_ARGUMENT;

  target->base = mem;
  target->capacity = capacity;
  target->pos = 0;
  return ML_OK;
}

ML_Status push_ml_arena(void** ptr, ml_arena* arena, u64 size) {
  if (!ptr) return ML_INVALID_ARGUMENT;
  if (!arena || !arena->base) return ML_INVALID_ARGUMENT;

  if (size == 0) { // define: size==0 gives NULL and succeeds
    *ptr = NULL;
    return ML_OK;
  }

  // Align current position to ARENA_ALIGN
  u64 aligned_pos = arena_align_up(arena->pos);

  // Check overflow and capacity
  if (aligned_pos > arena->capacity) return ML_OUT_OF_MEMORY;
  if (size > arena->capacity - aligned_pos) return ML_OUT_OF_MEMORY;

  // Compute pointer
  uintptr_t base = (uintptr_t)arena->base;
  *ptr = (void*)(base + (uintptr_t)aligned_pos);

  // Bump
  arena->pos = aligned_pos + size;
  return ML_OK;
}

u64 get_freemem_ml_arena_bytes(const ml_arena* arena) {
  if (!arena) return 0;
  if (arena->pos >= arena->capacity) return 0;
  return arena->capacity - arena->pos;
}

