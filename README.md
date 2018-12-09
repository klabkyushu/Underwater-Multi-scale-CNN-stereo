# Underwater-Multi-scale-CNN-stereo
Multi-scale CNN stereo network proposed in 3DV 2018.

## System requirements (our environment)
- OS: Windows 7 or higher
- CPU: Intel Xeon E5640 or higher
- RAM: 8GB
- GPU: GeForce GTX 1080
- Free Space: 45GB (for training dataset)

## Dependencies

- CUDA 8.0
- cuDNN v6.0
- Python 3.5
- Tensorflow 1.4
- Keras 2.1

## Usage

### Test
Left image, right image, maximum disparity
```
python ml-cnn_test.py test\view1.png test\view5.png 256
```
