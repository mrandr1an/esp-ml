#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <stddef.h>

typedef struct {
  char  *data;        
  size_t size;        
  size_t *row_offs;  
  size_t rows;        
} Csv;

void csv_free(Csv *csv);
int csv_load(const char*path,Csv *out_csv);
size_t csv_num_rows(const Csv *csv);
int csv_cell_span(const Csv *csv, size_t row, size_t col,
                  const char **start, const char **end);
int csv_get_string(const Csv *csv, size_t row, size_t col, char *out, size_t outcap);
int csv_get_f32(const Csv *csv, size_t row, size_t col, float *out_val);

#endif // CSV_UTILS_H
