#pragma once

// check incldues? Welche brauchst du wirklich
#include <string.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include <stdio.h>
#include <iostream>
#include <array>
#include "Constants.h"

static const char *TAG = "Test";
#define MOUNT_POINT "/sdcard" // keine defines, std::string

// rename file 
// rename class -> SDCard
class Database
{
public:
    // brauchst du die ganzen variablen?
    struct stat st;
    int file_created = 0;
    sdmmc_card_t *card; // pointer? ist uninitializiert
    esp_err_t ret;
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    const char *INMP441 = MOUNT_POINT "/INMP441.txt"; // std::string
    //const char mount_point[] = MOUNT_POINT;

    void mount();
    void unmount();
    void save(const std::array<int16_t, bufferdepth>& outbuffer_i2s0); // outbuffer als const reference, sample rate -> stdint typen nehmen (unitX_t)

private:
};