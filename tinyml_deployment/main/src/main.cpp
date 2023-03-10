#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "Sdcard.h"
#include "main_functions.h"
#include <cstdio>
#include <string.h>

#include <chrono>
#include <iostream>

#include "DataProvider.h"


void gatherData(Microphone& microphone, SDCard& filesystem, uint8_t seconds) {
    // read data for 1 second and save them to the sd card
    for (auto i = 0; i < seconds * 16; i++) { // comment why 16
      std::array<int16_t, DataProvider::BUFFERDEPTH> data;
      if (dataProvider.Read(&data) != Status::Success) {
        // log error while reading
        return;
      }
      if (sdcard.saveData(data) != Status::Success) {
          // log error while saving
        return;
      }
    }

    // log finished
    vTaskDelay(1 * pdSECOND); // needed?
    filesystem.unmount();
    // log unmound
    return;
}

void mainTask()
{
  Microphone microphone;
  auto microphoneResult = microphone.init();
  if (microphoneResult != Microphone::Status::Success) {
    // log
    return;
  }
  // log -> microphone initialized

  SDCard filesystem;
  auto fileSystemResult = filesystem.mount();
  if (fileSystemResult != SDCard::Status::Success) {
    // log
    reutrn;
  }
  // log -> sdCard initialized

  vTaskDelay(1 * pdSECOND); // wirklich?
  gatherData(&microphone, &filesystem , 300);
}


extern "C" void app_main()
{
  xTaskCreate((TaskFunction_t)&mainTask, "collect_data", 25 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}