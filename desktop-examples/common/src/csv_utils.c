#include "csv_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

void csv_free(Csv *csv) {
    if (!csv) return;
    free(csv->data);
    free(csv->row_offs);
    csv->data = NULL;
    csv->row_offs = NULL;
    csv->size = 0;
    csv->rows = 0;
}

int csv_load(const char *path, Csv *out_csv) {
    if (!out_csv) return 0;
    memset(out_csv, 0, sizeof(*out_csv));

    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long fsz = ftell(f);
    if (fsz < 0) { fclose(f); return 0; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 0; }

    out_csv->data = (char*)malloc((size_t)fsz + 1);
    if (!out_csv->data) { fclose(f); return 0; }

    size_t rd = fread(out_csv->data, 1, (size_t)fsz, f);
    fclose(f);
    if (rd != (size_t)fsz) { csv_free(out_csv); return 0; }

    out_csv->data[rd] = '\0';
    out_csv->size = rd;

    // Count rows by counting '\n'. Handle a final line without '\n'.
    size_t approx_rows = 1;
    for (size_t i = 0; i < rd; i++) {
        if (out_csv->data[i] == '\n') approx_rows++;
    }

    out_csv->row_offs = (size_t*)malloc(sizeof(size_t) * approx_rows);
    if (!out_csv->row_offs) { csv_free(out_csv); return 0; }

    // Build row offsets
    size_t r = 0;
    out_csv->row_offs[r++] = 0;

    for (size_t i = 0; i < rd; i++) {
        if (out_csv->data[i] == '\n') {
            // next row starts after '\n', if any bytes remain
            if (i + 1 < rd) out_csv->row_offs[r++] = i + 1;
        }
    }
    out_csv->rows = r;
    return 1;
}

size_t csv_num_rows(const Csv *csv) {
    return csv ? csv->rows : 0;
}

// Internal: find [start,end) of cell at (row,col). Returns 1 on success.
int csv_cell_span(const Csv *csv, size_t row, size_t col, const char **start, const char **end) {
    if (!csv || !start || !end) return 0;
    if (row >= csv->rows) return 0;

    const char *p = csv->data + csv->row_offs[row];

    // Row ends at '\n' or '\0'
    const char *row_end = strchr(p, '\n');
    if (!row_end) row_end = csv->data + csv->size;

    // Handle CRLF: trim trailing '\r'
    const char *trim_end = row_end;
    if (trim_end > p && *(trim_end - 1) == '\r') trim_end--;

    // Walk columns
    size_t c = 0;
    const char *field_start = p;
    for (const char *q = p; ; q++) {
        int at_end = (q >= trim_end);
        int at_comma = (!at_end && *q == ',');

        if (at_comma || at_end) {
            if (c == col) {
                *start = field_start;
                *end   = q;
                return 1;
            }
            c++;
            field_start = q + 1;
            if (at_end) break;
        }
    }
    return 0; // col out of range
}

int csv_get_string(const Csv *csv, size_t row, size_t col, char *out, size_t outcap) {
    if (!out || outcap == 0) return 0;
    out[0] = '\0';

    const char *s = NULL, *e = NULL;
    if (!csv_cell_span(csv, row, col, &s, &e)) return 0;

    size_t len = (size_t)(e - s);
    if (len + 1 > outcap) return 0;

    memcpy(out, s, len);
    out[len] = '\0';
    return 1;
}

int csv_get_f32(const Csv *csv, size_t row, size_t col, float *out_val) {
    if (!out_val) return 0;

    const char *s = NULL, *e = NULL;
    if (!csv_cell_span(csv, row, col, &s, &e)) return 0;

    // Copy into a small temp buffer to NUL-terminate (primitive, safe)
    char tmp[128];
    size_t len = (size_t)(e - s);
    if (len == 0 || len >= sizeof(tmp)) return 0;

    memcpy(tmp, s, len);
    tmp[len] = '\0';

    errno = 0;
    char *endptr = NULL;
    float v = strtof(tmp, &endptr);

    // Ensure full parse (allow trailing spaces)
    while (endptr && (*endptr == ' ' || *endptr == '\t')) endptr++;
    if (errno != 0) return 0;
    if (!endptr || *endptr != '\0') return 0;

    *out_val = v;
    return 1;
}
