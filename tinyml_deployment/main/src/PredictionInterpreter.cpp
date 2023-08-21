#include "PredictionInterpreter.h"

Prediction PredictionInterpreter::GetResult(TfLiteTensor* model_output) {
    int max_index = 0;
    float max_value = model_output->data.f[0];
    for (int i = 1; i < 4; i++) {
        if (model_output->data.f[i] > max_value) {
            max_value = model_output->data.f[i];
            max_index = i;
        }
    }

    switch (max_index) {
        case 0:
            return Prediction::heavy_rain;
        case 1:
            return Prediction::light_rain;
        case 2:
            return Prediction::medium_rain;
        case 3:
            return Prediction::no_rain;
        default:
            return Prediction::UNKNOWN;
    }
}
