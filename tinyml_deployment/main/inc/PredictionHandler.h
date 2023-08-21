// #pragma once

// #include "PredictionInterpreter.h"

// class PredictionHandler {
//  public:
//   void Update(Prediction prediction);

//  private:
// };

#pragma once

#include "PredictionInterpreter.h"
#include "SDcard.h"  
#include <string>
class PredictionHandler {
 public:
  PredictionHandler(SDcard &sdcardInstance); 
  void Update(Prediction prediction);

 private:
  SDcard &sdcard; 
  const std::string PREDICTION_FILE = "/SDcard/predictions.txt"; 
};
