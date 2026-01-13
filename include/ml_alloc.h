#ifndef ML_ALLOC_H
#define ML_ALLOC_H

#include "ml_defs.h"
#include "ml_error.h"

#define KiB(n) ( (u64)(n) << 10 )
#define MiB(n) ( (u64)(n) << 20 )
#define GiB(n) ( (u64)(n) << 30 )

#define ARENA_ALIGN (sizeof(void*))
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

typedef struct {
  void* base;
  u64 capacity;
  u64  pos; 
} ml_arena;

ML_Status create_ml_arena(ml_arena* target,void* mem,u64 capacity);
ML_Status push_ml_arena(void** ptr,ml_arena* arena, u64 size);
u64 get_freemem_ml_arena_bytes(const ml_arena* arena);

#endif //ML_ALLOC_H
