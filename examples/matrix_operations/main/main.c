#include "esp_log.h"
#include "ml_alloc.h"
#include "ml_defs.h"
#include "ml_error.h"
#include "ml_models.h"
#include "ml_operators.h"
#include "ml_primitives.h"
#include "ml_rng.h"
#include "nvs_flash.h"
#include "webpage.h"
#include "esp_random.h"

static const char *ML_TAG = "ESP-ML";

static f32 esp32_next01(void* ctx) {
  (void)ctx;
  uint32_t x = esp_random();
  return (f32)((double)x / 4294967296.0); // [0,1)
}

void app_main(void) {
  ESP_LOGI(TAG, "Starting...");

  ML_Status status = ML_OK;
  static unsigned char arena_mem[KiB(10)];
  ml_arena arena;
  status = create_ml_arena(&arena,arena_mem,(u64)sizeof(arena_mem));
  if (status != ML_OK)
    printf("Error at arena allocation: %d\n",status);

  //Samples
  const u64 N = 1;
  //Features
  const u64 D = 4;
  //Classes
  const u64 C = 3;

  ESP_LOGI(ML_TAG,"Begin, setting up dataset");
  Matf32 X_train,Y_train;
  status = create_Mat(&arena,&X_train,N,D);

  if (status != ML_OK)
    printf("Error at allocating X_train: %d\n",status);

  status = create_Mat(&arena,&Y_train,N,D);
  if (status != ML_OK)
    printf("Error at allocating Y_train: %d\n", status);
  
  ESP_LOGI(ML_TAG,"Done, setting up dataset");

  ML_Rng rng = {.ctx = NULL, .next01 = esp32_next01};
  SoftmaxRegressionConfig mconf;

  create_config_SoftmaxRegression(&mconf,N,D,C,&rng,FILL_XAVIER_UNIFORM,FILL_ZEROS);

  create_Mat(&arena,&in,N,D);
  if (status != ML_OK)
    printf("Error at allocating in: %d\n",status);

  create_Mat(&arena,&prob,N,C);
  if (status != ML_OK)
    printf("Error at allocating prob: %d\n",status);

  status = create_model_SoftmaxRegression(&arena,&model,mconf);
  if (status != ML_OK)
    printf("Error at allocating model: %d\n",status);

  if (status != ML_OK)
    printf("error at creating model");
  
  esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
  ESP_ERROR_CHECK(err);

  spiffs_init();
  wifi_init_sta();
  start_webserver();

  ESP_LOGI(TAG, "Done.");
}
