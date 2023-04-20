value             |Best Value So Far |Hyperparameter
28                |12                |units1
24                |24                |units2
28                |28                |units3
0.3               |0                 |dropout
3.6804e-06        |9.5302e-05        |learning_rate




Search: Running Trial #46

Value             |Best Value So Far |Hyperparameter
24                |12                |units1
32                |8                 |units2
24                |28                |units3
0                 |0.3               |dropout
1.7642e-05        |9.8982e-05        |learning_rate

2023-04-16 14:32:35.698488: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
         [[{{node Placeholder/_0}}]]
Epoch 1/20
125/128 [============================>.] - ETA: 0s - loss: 4.2284 - accuracy: 0.38402023-04-16 14:32:37.672239: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
         [[{{node Placeholder/_0}}]]
128/128 [==============================] - 2s 13ms/step - loss: 4.1582 - accuracy: 0.3848 - val_loss: 1.7773 - val_accuracy: 0.5312
Epoch 1/20
128/128 [==============================] - 1s 12ms/step - loss: 1.6269 - accuracy: 0.5195 - val_loss: 1.3174 - val_accuracy: 0.5781
Epoch 2/20
128/128 [==============================] - 2s 12ms/step - loss: 1.2158 - accuracy: 0.5781 - val_loss: 1.0228 - val_accuracy: 0.6641
Epoch 3/20
128/128 [==============================] - 2s 12ms/step - loss: 0.9000 - accuracy: 0.6660 - val_loss: 0.9009 - val_accuracy: 0.6797
Epoch 4/20
128/128 [==============================] - 2s 12ms/step - loss: 0.7111 - accuracy: 0.7266 - val_loss: 0.5848 - val_accuracy: 0.7422
Epoch 5/20
128/128 [==============================] - 2s 12ms/step - loss: 0.4953 - accuracy: 0.7930 - val_loss: 0.5257 - val_accuracy: 0.7812
Epoch 6/20
128/128 [==============================] - 2s 12ms/step - loss: 0.3826 - accuracy: 0.8535 - val_loss: 0.3814 - val_accuracy: 0.8047
Epoch 7/20
128/128 [==============================] - 2s 12ms/step - loss: 0.2843 - accuracy: 0.8867 - val_loss: 0.3272 - val_accuracy: 0.8594
Epoch 8/20
128/128 [==============================] - 2s 12ms/step - loss: 0.1907 - accuracy: 0.9258 - val_loss: 0.2767 - val_accuracy: 0.8672
Epoch 10/20
128/128 [==============================] - 2s 12ms/step - loss: 0.1373 - accuracy: 0.9512 - val_loss: 0.3523 - val_accuracy: 0.8672
Epoch 11/20
128/128 [==============================] - 2s 12ms/step - loss: 0.1122 - accuracy: 0.9629 - val_loss: 0.2150 - val_accuracy: 0.9297
Epoch 12/20
128/128 [==============================] - 2s 12ms/step - loss: 0.0681 - accuracy: 0.9844 - val_loss: 0.2012 - val_accuracy: 0.9219
Epoch 13/20
128/128 [==============================] - 2s 12ms/step - loss: 0.0561 - accuracy: 0.9863 - val_loss: 0.1918 - val_accuracy: 0.9219
Epoch 14/20
128/128 [==============================] - 2s 12ms/step - loss: 0.0393 - accuracy: 0.9980 - val_loss: 0.1641 - val_accuracy: 0.9375