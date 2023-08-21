// #include "PredictionHandler.h"

// void PredictionHandler::Update(Prediction prediction) {
//   switch (prediction) {
//     case Prediction::UNKNOWN:
//       break;
//     default:
//       break;
//   }
// }

#include "PredictionHandler.h"

PredictionHandler::PredictionHandler(SDcard &sdcardInstance) : sdcard(sdcardInstance) {}

void PredictionHandler::Update(Prediction prediction) {
    std::string predictionStr;
    switch (prediction) {
        case Prediction::heavy_rain:
            predictionStr = "heavy_rain";
            break;
        case Prediction::light_rain:
            predictionStr = "light_rain";
            break;
        case Prediction::medium_rain:
            predictionStr = "medium_rain";
            break;
        case Prediction::no_rain:
            predictionStr = "no_rain";
            break;
        default:
            predictionStr = "UNKNOWN";
            break;
    }

    // Save the prediction string to the SD card
    if (sdcard.save_string(predictionStr, PREDICTION_FILE) != SDcard::Status::Success) {
        ESP_LOGE("PredictionHandler", "Failed to save prediction to SD card");
    }
}
