#pragma once

#include "esp_system.h"
#include "FeatureProvider.h"
#include "Microphone.h"
#include "PredictionHandler.h"
#include "PredictionInterpreter.h"
#include "SDcard.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "iostream"
#include "micro_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <stdio.h>
#include <stdint.h>

class RainDetection{

private:

public:

    enum class Status
        {
            OK,
            Error
        };

    Microphone microphone;
    SDcard filesystem;
    FeatureProvider feature_provider;
    PredictionInterpreter prediction_interpreter;
    PredictionHandler prediction_handler;

    static tflite::MicroErrorReporter errorReporter;
    const tflite::AllOpsResolver resolver;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;

    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;

    std::array<int16_t, Microphone::BUFFERDEPTH> audioData;
    static constexpr size_t kTensorArenaSize = 70 * 1024;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];

    RainDetection::Status initialize();
    RainDetection::Status measure();
    RainDetection::Status checkModelVersion();
    RainDetection::Status initializeInterpreter();
    RainDetection::Status initializeModelInput();
    RainDetection::Status initializePeriphery();
    RainDetection::Status initializeMicrophone();
    RainDetection::Status initializeSdCard();
    void gatherData(uint16_t seconds);
};