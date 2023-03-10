#pragma once

#include <cstdint>
#include <array>
#include "Constants.h" // is this needed?

class DataProvider // Rename to Microphone?
{
public:
    enum class Status {
        Success,
        Error,
    }

    // check which one's are needed to be public..? -> make private
    constexpr uint32_t I2S_SAMPLE_RATE = 8000; // nachschauen: constexpr
    constexpr uint8_t I2S_WS_I2S0 = 25;
    constexpr uint8_t I2S_SD_I2S0 = 33;
    constexpr uint8_t I2S_SCK_I2S0 = 26;
    constexpr uint16_t BUFFERDEPTH = 512;

    // outbuffers are filled in one call of Read
    // a buffer array is written to according to the following pattern
    // [LSB, sample1-left, sample2-left, MSB, LSB, sample1-right, sample2-right, MSB]
    Status Init();
    Status Read(std::array<int16_t, DataProvider::BUFFERDEPTH>& data);

    // --> Alle funktionen klein schreiben
};