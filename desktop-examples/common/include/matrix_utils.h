#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stdio.h>
#include "ml_primitives.h"  // Matf32, MatGet

// Print full matrix with a name/title.
ML_Status print_matrix(const char *name, Matf32 m);

#endif // MATRIX_UTILS_H
