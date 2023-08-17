#include "main.h"

#define pdSECOND pdMS_TO_TICKS(1000)

tflite::MicroErrorReporter RainDetection::errorReporter;


RainDetection::Status RainDetection::initialize()
{
  RainDetection::model = tflite::GetModel(micro_model);

  if (checkModelVersion() != RainDetection::Status::OK)
  {
    return RainDetection::Status::Error;
  }

  if (initializeInterpreter() != RainDetection::Status::OK)
  {
    return RainDetection::Status::Error;
  }

  if (initializeModelInput() != RainDetection::Status::OK)
  {
    return RainDetection::Status::Error;
  }

  if (initializePeriphery() != RainDetection::Status::OK)
  {
    return RainDetection::Status::Error;
  }

  return RainDetection::Status::OK;
}

RainDetection::Status RainDetection::measure()
{
  gatherData(1);
  feature_provider.compute_spectrogram(this->audioData);
  feature_provider.WriteDataToModel(input);
  TfLiteStatus result = interpreter->Invoke();
  if (result != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(&errorReporter, "Unable to invoke intperpreter");
    return RainDetection::Status::Error;
  }

  // TODO: Write data to SDCard
  vTaskDelay(3 * pdSECOND);
  return RainDetection::Status::OK;
}

RainDetection::Status RainDetection::checkModelVersion()
{
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(&errorReporter, "Model version does not match. Expected: %d, found: %d", model->version(), TFLITE_SCHEMA_VERSION);
    return RainDetection::Status::OK;
  }
  return RainDetection::Status::Error;
}

RainDetection::Status RainDetection::initializeInterpreter()
{
  static tflite::MicroInterpreter static_interpreter(
    this->model, resolver, this->tensor_arena, this->kTensorArenaSize, &errorReporter);

  interpreter = &static_interpreter;

  TfLiteStatus result = interpreter->AllocateTensors();
  if (result != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(&errorReporter, "Can not allocate memory for interpreter");
    return Status::Error;
  }
  return Status::OK;
}

RainDetection::Status RainDetection::initializeModelInput()
{
  this->input = interpreter->input(0);
  this->output = interpreter->output(0);

  if ((input->dims->size != 4 || (input->dims->data[0] != 1) ||
        (input->dims->data[1] != 17) ||
        (input->dims->data[2] != 129) || (input->dims->data[3] != 1) ||
        (input->type != kTfLiteFloat32)))
  {
    TF_LITE_REPORT_ERROR(&errorReporter, "Input parameters and model parameters do not match.");
    return RainDetection::Status::Error;
  }
  return RainDetection::Status::OK;
}

RainDetection::Status RainDetection::initializePeriphery()
{
  if (initializeMicrophone() != Status::OK)
  {
    return Status::Error;
  }
  if (initializeSdCard() != Status::OK)
  {
    return Status::Error;
  }
  return Status::OK;
}

RainDetection::Status RainDetection::initializeMicrophone()
{
  Microphone::Status microphoneResult = microphone.init();
  if (microphoneResult != Microphone::Status::Success)
  {
    ESP_LOGE(Microphone::MicrophoneTAG, "Microphone could not be initialized");
    return Status::Error;
  }
  ESP_LOGI(SDcard::SDcardTAG, "SDcard initialized");
  return Status::OK;
}

RainDetection::Status RainDetection::initializeSdCard()
{
  SDcard::Status fileSystemResult = filesystem.mount();
  if (fileSystemResult != SDcard::Status::Success)
  {
    ESP_LOGE(SDcard::SDcardTAG, "Filesytem could not be initialized");
    return Status::Error;
  }
  ESP_LOGI(Microphone::MicrophoneTAG, "Micrphone initialized");
  return Status::OK;
}

void RainDetection::gatherData(uint16_t seconds)
{
  // read data for n seconds and save them to the sd card
  for (int i = 0; i < seconds; i++)
  {
    microphone.read(this->audioData);
  };
}


void mainTask()
{
  RainDetection rainDetection;
  if (rainDetection.initialize() != RainDetection::Status::OK)
  {
    return;
  }
  while (rainDetection.measure() == RainDetection::Status::OK)
  {
  }
}

extern "C" void app_main()
{
  xTaskCreate((TaskFunction_t)&mainTask, "mainTask", 100 * 1024, NULL, 8, NULL); // rename tensorflow?
  vTaskDelete(NULL);
}