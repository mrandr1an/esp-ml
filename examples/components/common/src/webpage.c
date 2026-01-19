#include "webpage.h"
#include "esp_http_server.h"
#include "esp_spiffs.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "cJSON.h"
#include "matrix_utils.h"
#include "ml_models.h"
#include "ml_primitives.h"
#include <errno.h>
#include <stdio.h>

Matf32 in;
Matf32 prob;
SoftmaxRegression model;

esp_err_t set_content_type_from_path(httpd_req_t *req, const char *path)
{
    const char *ext = strrchr(path, '.');
    if (!ext) return httpd_resp_set_type(req, "text/plain");

    if (strcmp(ext, ".html") == 0)
      return httpd_resp_set_type(req, "text/html");
    if (strcmp(ext, ".css") == 0)
      return httpd_resp_set_type(req, "text/css");
    if (strcmp(ext, ".js") == 0)
      return httpd_resp_set_type(req, "application/javascript");
    if (strcmp(ext, ".png") == 0)
      return httpd_resp_set_type(req, "image/png");
    if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0)
      return httpd_resp_set_type(req, "image/jpeg");
    if (strcmp(ext, ".ico") == 0)
      return httpd_resp_set_type(req, "image/x-icon");

    return httpd_resp_set_type(req, "application/octet-stream");
}


esp_err_t send_file(httpd_req_t *req, const char *filepath)
{
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        ESP_LOGW(TAG, "File not found: %s", filepath);
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "Not found");
        return ESP_FAIL;
    }

    set_content_type_from_path(req, filepath);

    char buf[1024];
    size_t read_bytes;
    while ((read_bytes = fread(buf, 1, sizeof(buf), f)) > 0) {
        if (httpd_resp_send_chunk(req, buf, read_bytes) != ESP_OK) {
            fclose(f);
            httpd_resp_sendstr_chunk(req, NULL); // end response
            return ESP_FAIL;
        }
    }

    fclose(f);
    return httpd_resp_sendstr_chunk(req, NULL); // end response
}


esp_err_t static_get_handler(httpd_req_t *req)
{
    // Basic traversal guard
    if (strstr(req->uri, "..")) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Bad path");
        return ESP_FAIL;
    }

    // Default document
    const char *uri = req->uri;
    if (strcmp(uri, "/") == 0) {
        uri = "/index.html";
    }

    char path[256];

    // "/spiffs" + uri + '\0'
    const size_t prefix_len = strlen("/spiffs");
    const size_t uri_len = strnlen(uri, 512);  // cap scan to something reasonable

    if (uri_len == 512) { // no terminator within cap => suspiciously long
        httpd_resp_send_err(req, HTTPD_414_URI_TOO_LONG, "URI too long");
        return ESP_FAIL;
    }

    if (prefix_len + uri_len + 1 > sizeof(path)) {
        httpd_resp_send_err(req, HTTPD_414_URI_TOO_LONG, "URI too long");
        return ESP_FAIL;
    }

    int n = snprintf(path, sizeof(path), "/spiffs%s", uri);
    if (n < 0 || (size_t)n >= sizeof(path)) {
      httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                          "Path build failed");
        return ESP_FAIL;
    }

    return send_file(req, path);
}

void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < WIFI_MAX_RETRIES) {
            s_retry_num++;
            ESP_LOGW(TAG,
                     "Wi-Fi disconnected, retry %d/%d", s_retry_num, WIFI_MAX_RETRIES);
            esp_wifi_connect();
        } else {
            ESP_LOGE(TAG, "Wi-Fi failed to connect");
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group,
                           WIFI_CONNECTED_BIT);
    }
}

void spiffs_init(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = "storage",
        .max_files = 8,
        .format_if_mount_failed = true,
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SPIFFS mount failed (%s)", esp_err_to_name(ret));
        abort();
    }

    size_t total = 0, used = 0;
    ESP_ERROR_CHECK(esp_spiffs_info(conf.partition_label,
                                    &total, &used));

    ESP_LOGI(TAG,
             "SPIFFS mounted: total=%u, used=%u", (unsigned)total, (unsigned)used);
}

void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_config_t wifi_config = { 0 };
    strncpy((char *)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char *)wifi_config.sta.password, WIFI_PASS, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi init done, connecting...");

    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "Wi-Fi connected");
}

esp_err_t post_train_handler(httpd_req_t *req)
{
  ESP_LOGI(TAG,"Begin training");
  
  ESP_LOGI(TAG,"End training");
  return ESP_OK;
}

esp_err_t post_infer_handler(httpd_req_t *req)
{
  ESP_LOGI(TAG,"Begin inference");
  char* body = NULL;
  size_t body_len = 0;
  if (read_req_body(req, &body, &body_len) != ESP_OK) {
        return ESP_FAIL;
  }

  cJSON* root = cJSON_Parse(body);
  free(body);

  if (!root) {
    httpd_resp_send_err(req,HTTPD_400_BAD_REQUEST,"Invalid Json");
    return ESP_FAIL;
  }

  cJSON* sepal_length = cJSON_GetObjectItem(root,"sepal_length");
  cJSON* sepal_width= cJSON_GetObjectItem(root,"sepal_width");
  cJSON* petal_length = cJSON_GetObjectItem(root,"petal_length");
  cJSON* petal_width= cJSON_GetObjectItem(root,"petal_width");

  if (cJSON_IsNumber(sepal_length))
    ESP_LOGI(TAG, "sepal_length = %f", (float) sepal_length->valuedouble);
  else ESP_LOGW(TAG, "sepal_length missing or not a number");

  if (cJSON_IsNumber(sepal_width))
    ESP_LOGI(TAG, "sepal_width  = %f", (float) sepal_width->valuedouble);
  else ESP_LOGW(TAG, "sepal_width missing or not a number");

  if (cJSON_IsNumber(petal_length))
    ESP_LOGI(TAG, "petal_length = %f", (float) petal_length->valuedouble);
  else ESP_LOGW(TAG, "petal_length missing or not a number");

  if (cJSON_IsNumber(petal_width))
    ESP_LOGI(TAG, "petal_width  = %f", (float) petal_width->valuedouble);
  else ESP_LOGW(TAG, "petal_width missing or not a number");

  httpd_resp_set_type(req, "application/json");
  httpd_resp_sendstr(req, "{\"ok\":true}");

  MatSet(&in,0,0,(float) sepal_length->valuedouble);
  MatSet(&in,0,1,(float) sepal_width->valuedouble);
  MatSet(&in,0,2,(float) petal_length->valuedouble);
  MatSet(&in,0,3,(float) petal_width->valuedouble);
  print_matrix("Infering:\n",in);

  infer_SoftmaxRegression(&model,in,&prob);
  print_matrix("Result of inference:\n",prob);
  print_matrix("Model Inner:\n",model.lin.W);
  print_matrix("Result of inference:\n",prob);
 
  cJSON_Delete(root);
  ESP_LOGI(TAG,"End inference");
  return ESP_OK;
}

httpd_handle_t start_webserver(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.uri_match_fn = httpd_uri_match_wildcard; 

    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start http server");
        return NULL;
    }

    httpd_uri_t uri_get = {
        .uri      = "/*",
        .method   = HTTP_GET,
        .handler  = static_get_handler,
        .user_ctx = NULL
    };
    httpd_register_uri_handler(server, &uri_get);

    httpd_uri_t uri_post_infer = {
      .uri      = "/infer",
      .method   = HTTP_POST,
      .handler  = post_infer_handler,
      .user_ctx = NULL
    };
    httpd_register_uri_handler(server, &uri_post_infer);

    httpd_uri_t uri_post_train = {
      .uri      = "/train",
      .method   = HTTP_POST,
      .handler  = post_train_handler,
      .user_ctx = NULL
    };

    httpd_register_uri_handler(server, &uri_post_train);

    ESP_LOGI(TAG, "HTTP server started");
    return server;
}

esp_err_t read_req_body(httpd_req_t *req, char **out, size_t *out_len)
{
    size_t total = req->content_len;

    if (total == 0) {
        *out = NULL;
        *out_len = 0;
        return ESP_OK;
    }
    if (total > 4096) { 
        httpd_resp_send_err(req, HTTPD_413_CONTENT_TOO_LARGE, "Body too large");
        return ESP_FAIL;
    }

    char *buf = malloc(total + 1);
    if (!buf) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "OOM");
        return ESP_FAIL;
    }

    size_t received = 0;
    while (received < total) {
        int r = httpd_req_recv(req, buf + received, total - received);
        if (r == HTTPD_SOCK_ERR_TIMEOUT) {
            // retry
            continue;
        }
        if (r <= 0) {
            free(buf);
            httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Failed to read body");
            return ESP_FAIL;
        }
        received += (size_t)r;
    }

    buf[received] = '\0'; 
    *out = buf;
    *out_len = received;
    return ESP_OK;
}

bool parse_float_kv(const char *body, const char *key, float *out)
{
    char tmp[64];
    if (httpd_query_key_value(body, key, tmp, sizeof(tmp)) != ESP_OK) {
        return false;
    }

    errno = 0;
    char *end = NULL;
    float v = strtof(tmp, &end);

    if (errno != 0 || end == tmp || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}
