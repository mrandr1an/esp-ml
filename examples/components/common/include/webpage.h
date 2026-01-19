#ifndef WEBPAGE_H
#define WEBPAGE_H
#include "esp_event.h"
#include "esp_http_server.h"
#include "ml_models.h"
#include "ml_primitives.h"
#include <stdint.h>

#define WIFI_SSID "Vodafone 2.4"
#define WIFI_PASS "2299047039"

static const char* TAG = "WEBPAGE";
static EventGroupHandle_t s_wifi_event_group;
static const int WIFI_CONNECTED_BIT = BIT0;
static int s_retry_num = 0;
static const int WIFI_MAX_RETRIES = 10;

extern Matf32 in;
extern Matf32 prob;
extern SoftmaxRegression model;

esp_err_t set_content_type_from_path(httpd_req_t* req,const char* path);
esp_err_t send_file(httpd_req_t* req,const char* filepath);

esp_err_t static_get_handler(httpd_req_t* req);
esp_err_t post_train_handler(httpd_req_t *req);
esp_err_t post_infer_handler(httpd_req_t *req);
esp_err_t read_req_body(httpd_req_t *req, char **out, size_t *out_len);
bool parse_float_kv(const char *body, const char *key, float *out);
void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data);
void wifi_init_sta(void);
void spiffs_init(void);

httpd_handle_t start_webserver(void);
#endif //WEBPAGE_H
