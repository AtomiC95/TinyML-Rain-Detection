#pragma once
#include <cmath>
#include <array>
#include <vector>

#include "esp_dsp.h"
#include "tensorflow/lite/c/common.h"

class FeatureProvider {
   public:
    static constexpr int16_t n_fft = 256;
    static constexpr int16_t hop_length = 500;
    static constexpr int16_t num_frequency_bins = 129;
    static constexpr int16_t num_windows = 17;
    static float fft_work_buffer[n_fft * 2];
    std::array<int16_t, 8000> audio_buffer;
    std::array<float,num_windows*num_frequency_bins> spectrogram;
    
    FeatureProvider() = default;
    ~FeatureProvider() = default;
    // void SetInputData(const std::vector<float>& inputData);
    void compute_spectrogram(std::array<int16_t, 8000>& data);
    void ExtractFeatures();
    void WriteDataToModel(TfLiteTensor* modelInput);

   private:
};
