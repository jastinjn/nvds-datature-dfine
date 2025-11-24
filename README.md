# nvds-datature-dfine

This project is an example of how to integrate Datature models with Nvidia Deepstream

# Prerequisites

Nvidia Jetson with Jetpack 5.x or above
Deepstream 6.3 or above

# Usage

1. Add your Datature exported D-FINE (or other models) .onnx file to models/, following the Triton repo structure:

models/\n
--dfine/\n
--config.pbtxt\n
--1/\n
--model.onnx

2. Compile the TensorRT engine file (takes 15-30 minutes)

```
cd models/dfine/1
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

3. Compile the custom parser lib with make

```
cd parser
export CUDA_VER=(your cuda version)
sudo -E make install
```

4. Run Deepstream app

```
deepstream-app -c source1_triton_osd.txt
```
