#pragma once

#include "tensorflow/lite/c/common.h"

enum class Prediction {
  UNKNOWN = 0,
  heavy_rain = 1,
  light_rain = 2,
  medium_rain = 3,
  no_rain = 4,
};

class PredictionInterpreter {
 public:
  PredictionInterpreter() = default;
  ~PredictionInterpreter() = default;
  virtual Prediction GetResult(TfLiteTensor* model_output);
};