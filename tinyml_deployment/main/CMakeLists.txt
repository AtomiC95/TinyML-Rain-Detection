set(SOURCES ./src/main.cpp ./src/FeatureProvider.cpp ./src/Microphone.cpp ./src/main_functions.cpp ./src/micro_model.cpp ./src/PredictionInterpreter.cpp ./src/PredictionHandler.cpp ./src/SDcard.cpp)
idf_component_register(SRCS ${SOURCES}
                            INCLUDE_DIRS . inc
                            REQUIRES freertos esp_common tfmicro esp-nn fatfs esp_hw_support esp_wifi mqtt nvs_flash esp-dsp)