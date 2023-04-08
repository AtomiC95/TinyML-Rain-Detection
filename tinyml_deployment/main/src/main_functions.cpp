#include "main_functions.h"

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

// delay connstant -> 1 sec
#define pdSECOND pdMS_TO_TICKS(1000)

namespace {
// declare ErrorReporter, a TfLite class for error logging
tflite::ErrorReporter *error_reporter = nullptr;
// declare the model that will hold the generated C array
const tflite::Model *model = nullptr;
// declare interpreter, runs inference using model and data
tflite::MicroInterpreter *interpreter = nullptr;
// declare model input and output as 1D-arrays
TfLiteTensor *model_input = nullptr;
TfLiteTensor *model_output = nullptr;
// create an area of mvoid gatherData(Microphone &microphone, std::array<int16_t, Microphone::BUFFERDEPTH> &data, uint16_t seconds) {
//     // filesystem.files_checker();
//     // read data for seconds and save them to the sd card
//     for (int i = 0; i < seconds; i++) {
//         // std::cout << "Hallo\n";
//         microphone.read(data);
//         // if (filesystem.save(data) == SDcard::Status::Error) {
//         //     i = seconds * Microphone::counter;
//     };emory to use for input, output, and intermediate arrays.
// the size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 70 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// processing pipeline
Microphone microphone;
SDcard filesystem;
FeatureProvider feature_provider;
PredictionInterpreter prediction_interpreter;
PredictionHandler prediction_handler;
std::array<int16_t, Microphone::BUFFERDEPTH> data;  // 1/16 of a second direktly read from the INMP441
// std::array<int16_t, Microphone::BUFFERDEPTH* 16> collected_data; // one second of collected data
}  // namespace

void setup() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // import the trained weights from the C array
    model = tflite::GetModel(micro_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // load all tflite micro built-in operations
    // for example layers, activation functions, pooling
    static tflite::AllOpsResolver resolver;

    // initialize interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // interpreter allocates memory according to model requirements
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed\n");
        return;
    }

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    /*
    Assert that real input matches expect input
    Types supported for model_input->type
    IMPORTANT: dimensions need to be updated for each use case
        typedef enum {
            kTfLiteNoType = 0,
            kTfLiteFloat32 = 1,
            kTfLiteInt32 = 2,
            kTfLiteUInt8 = 3, // IMPORTANT: deprecated, see
                              //
    https://github.com/tensorflow/tflite-micro/issues/216 kTfLiteInt64 = 4,
            kTfLiteString = 5,
            kTfLiteBool = 6,
            kTfLiteInt16 = 7,
            kTfLiteComplex64 = 8,
            kTfLiteInt8 = 9,
            kTfLiteFloat16 = 10,
            kTfLiteFloat64 = 11,
        } TfLiteType;
    */
    if ((model_input->dims->size != 4 || (model_input->dims->data[0] != 1) ||
         (model_input->dims->data[1] != 128) ||
         (model_input->dims->data[2] != 3) || (model_input->dims->data[3] != 1) ||
         (model_input->type != kTfLiteFloat32))) {
        error_reporter->Report("Bad input tensor parameters in model\n");
    }

    // initialize periphery
    Microphone::Status microphoneResult = microphone.init();
    SDcard::Status fileSystemResult = filesystem.mount();
    std::cout << int(microphoneResult) << std::endl;
    if (microphoneResult != Microphone::Status::Success) {
        ESP_LOGE(Microphone::MicrophoneTAG, "Microphone could not be initialized");
        return;
    }

    if (fileSystemResult != SDcard::Status::Success) {
        ESP_LOGE(SDcard::SDcardTAG, "Filesytem could not be initialized");
        return;
    }
    ESP_LOGI(SDcard::SDcardTAG, "SDcard initialized");
    ESP_LOGI(Microphone::MicrophoneTAG, "Micrphone initialized");
}

void gatherData(Microphone &microphone, std::array<int16_t, Microphone::BUFFERDEPTH> &data, uint16_t seconds) {
    // filesystem.files_checker();
    // read data for seconds and save them to the sd card
    for (int i = 0; i < seconds; i++) {
        // std::cout << "Hallo\n";
        microphone.read(data);
        // if (filesystem.save(data) == SDcard::Status::Error) {
        //     i = seconds * Microphone::counter;
    };
}

void loop() {
    gatherData(microphone, data, 1);
    //  read raw data and convert data to format suitable for model
    // feature_provider.SetInputData(data_provider.Read());
    //  feature_provider.ExtractFeatures(data);
    feature_provider.compute_spectrogram(data);
    feature_provider.WriteDataToModel(model_input);
    // run inference on pre-processed data
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed");
        return;
    }

    std::cout << model_output->data.f[0] << std::endl;
    std::cout << model_output->data.f[1] << std::endl;
    std::cout << model_output->data.f[2] << std::endl;
    std::cout << model_output->data.f[3] << std::endl;
    std::cout << "---------------------" << std::endl;
    // interpret raw model predictions
    // auto prediction = prediction_interpreter.GetResult(model_output);

    // // act upon processed predictions
    // prediction_handler.Update(prediction);

    vTaskDelay(3 * pdSECOND);
}
