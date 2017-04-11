# keras-temporal-pooling
Implementation of temporal pooling using keras

This implementation is based on the original keras [pooling layer](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py)

## Temporal Pooling Layers 
- TemporalAveragePooling
- TemporalMaxPooling
- TemporalAverageGlobalPooling
- TemporalMaxGlobalPooling

## Temporal Pooling
Temporal pooling layers accept 5D input of shape B x T x H x W x C(if tensorflow backend) but do not work in the same way as [Pooling3D layers](https://github.com/fchollet/keras/blob/master/keras/layers/pooling.py#L275) do. Temporal pooling layers will apply pooling on the corresponding channels at each time step rather than apply pooling to the entire volume. This is helpful when we use RNN/LSTM/GRU as feature extractor and want to average (or maximize) the feature maps at each time step.
