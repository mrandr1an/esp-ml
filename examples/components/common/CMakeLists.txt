set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../"
)

idf_component_register(
  SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/matrix_utils.c"
  INCLUDE_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/include"
  REQUIRES esp-ml
)
