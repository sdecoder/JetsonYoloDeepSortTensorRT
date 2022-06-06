# Project README

##  Perquisite
In order to successuflly compile this project, following runtime requirement should be satisfied:

2. 1. Jetson nano or Jetson Xavier nx deveploment platform
2. Jetpack 4.5.1
3. python3 with default(jetson nano or jetson xavier nx has default python3 with tensorrt 7.1.3.0 )
4. tensorrt 7.1.3.0
5. torch 1.8.0
6. torchvision 0.9.0
7. torch2trt 0.3.0
8. onnx 1.4.1
9. opencv-python 4.5.3.56
10. protobuf 3.17.3
11. scipy 1.5.4

## Build and Run
1. Getting the models: Download the Yolov5 and deepsort models from the Github community; You need two model, one is yolov5 model, for detection, generating from [tensorrtx](https://github.com/wang-xinyu/tensorrtx). And the other is deepsort model, for tracking. You should generate the model the same way.
1. Generate the tensorrt engine files: input ????. Once the models are loaded and converted, the program will load the specified video and start to inference. The result will be shown 
 



```shell 
cd tensorrtx/yolov5
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/yolov5/yolov5s.wts {tensorrtx}/yolov5/build
cmake ..
make
// yolov5s
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
// test your engine file
sudo ./yolov5 -d yolov5s.engine ../samples
```
Then you get the yolov5s.engine, and you can put `yolov5s.engine` in My project. For example

```shell
cd {yolov5-deepsort-tensorrt}
mkdir resources
cp {tensorrtx}/yolov5/build/yolov5s.engine {yolov5-deepsort-tensorrt}/resources
```

1. Get deepsort engine file 
```shell
git clone https://github.com/RichardoMrMu/deepsort-tensorrt.git
// 根据github的说明
cp {deepsort-tensorrt}/exportOnnx.py {deep_sort_pytorch}/
python3 exportOnnx.py
mv {deep_sort_pytorch}/deepsort.onnx {deepsort-tensorrt}/resources
cd {deepsort-tensorrt}
mkdir build
cd build
cmake ..
make 
./onnx2engine ../resources/deepsort.onnx ../resources/deepsort.engine
// test
./demo ../resource/deepsort.engine ../resources/track.txt
```
After all 5 step, you can get the yolov5s.engine and deepsort.engine.

You may face some problems in getting yolov5s.engine and deepsort.engine, you can upload your issue in github or [csdn artical](https://blog.csdn.net/weixin_42264234/article/details/120152117).
<details>
<summary>Different versions of yolov5</summary>

<summary>Config</summary>

- Choose the model s/m/l/x/s6/m6/l6/x6 from command line arguments.
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- INT8/FP16/FP32 can be selected by the macro in yolov5.cpp, **INT8 need more steps, pls follow `How to Run` first and then go the `INT8 Quantization` below**
- GPU id can be selected by the macro in yolov5.cpp
- NMS thresh in yolov5.cpp
- BBox confidence thresh in yolov5.cpp
- Batch size in yolov5.cpp
</details>
 
```shell
// before 
static constexpr int CLASS_NUM = 80; // 20
static constexpr int INPUT_H = 640;  // 21  yolov5's input height and width must be divisible by 32.
static constexpr int INPUT_W = 640; // 22

// after 
// if your model is 2 classfication and image size is 416*416
static constexpr int CLASS_NUM = 2; // 20
static constexpr int INPUT_H = 416;  // 21  yolov5's input height and width must be divisible by 32.
static constexpr int INPUT_W = 416; // 22
```

```shell
cd {tensorrtx}/yolov5/
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset

mkdir build
cd build
cp {ultralytics}/yolov5/yolov5s.wts {tensorrtx}/yolov5/build
cmake ..
make
sudo ./yolov5 -s [.wts] [.engine] [s/m/l/x/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file
sudo ./yolov5 -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov5s
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
sudo ./yolov5 -d yolov5s.engine ../samples
// For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
sudo ./yolov5 -s yolov5_custom.wts yolov5.engine c 0.17 0.25
sudo ./yolov5 -d yolov5.engine ../samples
```

