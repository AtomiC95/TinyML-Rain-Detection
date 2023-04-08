#!/bin/bash
MODEL_PATH=/home/nikolas/git/TinyML-Rain-Detection/model_creation/models/
xxd -i $MODEL_PATH/lite_model.tflite > $MODEL_PATH/model_hexgraph.cc
