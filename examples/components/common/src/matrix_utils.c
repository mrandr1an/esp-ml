#include "matrix_utils.h"
#include "inttypes.h"
#include <stdio.h>

ML_Status print_matrix(const char *name, Matf32 m) {
  if (!name) name = "(unnamed)";
  ML_Status status = ML_OK;
  printf("\n%s: %" PRIu64 " x %" PRIu64 "\n",
         name, m.rows, m.cols);

  for (u64 r = 0; r < m.rows; ++r) {  
    for (u64 c = 0; c < m.cols; ++c) {
      f32 val = 0.0f;
      status = MatGet(m,r,c,&val);
      if(status != ML_OK) return status;
      printf("%0.3f ",val);
    }
    printf("\n");
  }

  return status;
}
