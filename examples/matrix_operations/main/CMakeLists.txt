idf_component_register(SRCS "main.c"
  REQUIRES esp-ml
           common
)


spiffs_create_partition_image(storage ../spiffs_image FLASH_IN_PROJECT)
