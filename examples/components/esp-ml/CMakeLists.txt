set(ESP_ML_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../") 

idf_component_register(
  SRCS
    "${ESP_ML_ROOT}/src/ml_primitives.c"
    "${ESP_ML_ROOT}/src/ml_alloc.c"
    "${ESP_ML_ROOT}/src/ml_operators.c"
    "${ESP_ML_ROOT}/src/ml_rng.c"
    "${ESP_ML_ROOT}/src/ml_models.c"
  INCLUDE_DIRS
    "${ESP_ML_ROOT}/include"
)
