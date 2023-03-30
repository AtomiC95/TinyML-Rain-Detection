#include "FeatureProvider.h"

float FeatureProvider::fft_work_buffer[n_fft * 2];

// void FeatureProvider::SetInputData(const std::vector<float>& inputData) {
//     std::copy(inputData.begin(), inputData.end(), data.begin());
// }

void FeatureProvider::ExtractFeatures() {
    // apply something to your data
}

void FeatureProvider::WriteDataToModel(TfLiteTensor* modelInput) {
    // TODO(nrieder@itemis.com): check size of modelInput data.
    std::copy(spectrogram.begin(), spectrogram.end(), modelInput->data.f);
}

void FeatureProvider::compute_spectrogram(std::array<int16_t, 8000>& data) {
    audio_buffer = data;
    for (int t = 0; t < FeatureProvider::num_windows; ++t) {
        // All the code inside the compute_spectrogram function

        // Initialize the FFT and windowing objects
        dsps_fft2r_init_fc32(FeatureProvider::fft_work_buffer, FeatureProvider::n_fft);
        // dsps_wind_hann_f32(audio_buffer, FeatureProvider::n_fft);

        float temp_complex[FeatureProvider::n_fft * 2] = {0};
        float temp_magnitude[FeatureProvider::n_fft / 2 + 1] = {0};

        // Iterate through the audio buffer with the specified hop length
        for (int i = 0; i < FeatureProvider::n_fft; ++i) {
            // float audio_sample = static_cast<float>(audio_buffer[t * FeatureProvider::hop_length + i]);
            // float hann_coeff = 0.5f * (1.0f - std::cos(2.0f * M_PI * static_cast<float>(i) / (FeatureProvider::n_fft - 1)));
            // temp_complex[i * 2] = audio_sample * hann_coeff;
            // temp_complex[i * 2 + 1] = 0;
            // Make sure the index is within the bounds of the audio_buffer
            size_t index = t * FeatureProvider::hop_length + i;
            if (index >= audio_buffer.size()) {
                break;
            }

            float audio_sample = static_cast<float>(audio_buffer[index]);
            float hann_coeff = 0.5f * (1.0f - std::cos(2.0f * M_PI * static_cast<float>(i) / (FeatureProvider::n_fft - 1)));
            temp_complex[i * 2] = audio_sample * hann_coeff;
            temp_complex[i * 2 + 1] = 0;
        }

        // Perform FFT
        dsps_fft2r_fc32(temp_complex, FeatureProvider::n_fft);
        dsps_bit_rev_fc32(temp_complex, FeatureProvider::n_fft);
        dsps_cplx2reC_fc32(temp_complex, FeatureProvider::n_fft);

        // Compute magnitudes
        for (int i = 0; i < ((FeatureProvider::n_fft / 2) + 1); ++i) {
            float real = temp_complex[i * 2];
            float imag = temp_complex[i * 2 + 1];
            temp_magnitude[i] = std::sqrt(real * real + imag * imag);
        }

        // Copy the magnitude values to the spectrogram
        for (int f = 0; f < FeatureProvider::num_frequency_bins; ++f) {
            // spectrogram[t][f] = temp_magnitude[f];
            spectrogram[(t * num_frequency_bins) + f] = temp_magnitude[f];
        }
    }
    // for (int ite = 0; ite < spectrogram.size(); ite++) {
    //     std::cout << FeatureProvider::spectrogram[ite] << std::endl;
    // }
}
