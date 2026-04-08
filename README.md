# 目标检测模型训练与部署指南

##  目录
- [1. 环境安装](#1-环境安装)
- [2. 目标检测](#2-目标检测)
- [3. 板卡部署](#3-板卡部署)

---

## 1.相关环境安装

### 1.1 创建虚拟环境

#### 使用 Conda 创建虚拟环境
```bash
# 使用conda创建虚拟环境
conda create -n yolov8 python=3.8
conda activate yolov8
```

#### 使用 Python 创建虚拟环境
```bash
# 或者使用python创建虚拟环境
python -m venv yolov8
yolov8/Scripts/activate
```

### 1.2 安装 YOLO 依赖

#### 方式一：直接安装
```bash
# 安装YOLO所需依赖，直接使用命令安装
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 方式二：从源码安装
```bash
# 或者通过拉取仓库然后安装
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
```

### 1.3 验证安装
```bash
# 安装成功后，使用命令查看版本
(yolov8) cat@root:/$ yolo version
8.3.172
```

---

## 2. 目标检测

### 2.1 重新训练

#### 训练命令
```bash
# 在data.yaml中配置好数据集路径和训练参数，使用以下命令开始训练（参数可自行调整）
yolo detect train data=your_data.yaml model=our_model.pt epochs=200 imgsz=640 device=0 patience=50
```

#### 训练输出示例
```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/20       3.1G     0.9032      0.478     0.9705         65        640: 100%|██████████| 2094/2094 [02:03<00:00,
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 272/272 [00:2
                   all       8699      69577      0.879       0.86      0.899      0.612

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/20       3.1G     0.9068     0.4784     0.9716         68        640: 100%|██████████| 2094/2094 [02:04<00:00,
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 272/272 [00:2
                   all       8699      69577      0.893      0.862       0.92       0.63
```

### 2.2 模型预测

#### 预测命令
```bash
# 使用yolo predict命令预测单张图片，更多参数具体参考下：https://docs.ultralytics.com/usage/cfg/
# 预测图片结果保存在当前runs目录下，具体路径是./runs/detect/predict/test.jpg
(yolov8) cat@root: yolo detect predict model=our_model.pt source=test.jpg
```

#### 预测输出示例
```
Ultralytics 8.3.172  Python-3.8.6 torch-2.4.1+cpu CPU (13th Gen Intel Core(TM) i9-13900H)
YOLO11s summary (fused): 100 layers, 9,417,831 parameters, 0 gradients, 21.3 GFLOPs

image 1/1 test.jpg: 384x640 7 persons, 104.7ms
Speed: 4.0ms preprocess, 64.7ms inference, 7.6ms postprocess per image at shape (1, 3, 384, 640)
Results saved to runs\detect\predict
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

### 2.3 模型导出为 TorchScript（或者onnx）

#### 导出步骤
```bash
# 使用 airockchip/ultralytics_yolov8 可以直接导出适配rknpu的模型，在npu上获得更高的推理效率
# 复制训练好的模型our_model.pt到目录下
# 然后修改./ultralytics/cfg/default.yaml文件，主要是设置下model，为自己训练的模型路径：
    model: ./our_model.pt # (str, optional) path to model file, i.e. yolov8n.pt, yolov8n.yaml
    data:  # (str, optional) path to data file, i.e. coco128.yaml

# 导出模型
(yolov8) cat@root:~/ultralytics_yolov8$ export PYTHONPATH=./
(yolov8) cat@root:~/ultralytics_yolov8$ python ./ultralytics/engine/exporter.py
```

#### 导出输出示例
```
Ultralytics 8.3.175  Python-3.8.10 torch-2.4.1+cu121 CPU (Intel Xeon Gold 6430)
YOLO11s summary (fused): 100 layers, 9,417,831 parameters, 0 gradients, 21.3 GFLOPs

PyTorch: starting from 'ultralytics/best.pt' with input shape (16, 3, 640, 640) BCHW and output shape(s) (16, 17, 8400) (18.3 MB)

TorchScript: starting export with torch 2.4.1+cu121...
TorchScript: export success ✅ 9.8s, saved as 'ultralytics/best.torchscript' (36.4 MB)

Export complete (13.6s)
Results saved to /autodl-fs/data/ultralytics
Predict:         yolo predict task=detect model=ultralytics/best.torchscript imgsz=640
Validate:        yolo val task=detect model=ultralytics/best.torchscript imgsz=640 data=data.yaml
Visualize:       https://netron.app
```

### 2.4 模型转换为 RKNN

#### 转换步骤
```bash
# 前面导出的模型，还需要通过toolkit2转换成rknn模型
# DATASET为rknn所需数据，供其了解数据集性质，便于最佳优化
# DATASET_PATH为DATASET中图片路径，可根据实际改写
# 执行pt2rknn.py，配置好参数，得到.rknn模型

### 需修改 ###
(toolkit2_1.6) llh@YH-LONG:/xxx/yolov8$ python3 pt2rknn.py
Usage: python3 pt2rknn.py torchscript_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]
    platform choose from [rk3562,rk3566,rk3568,rk3588]
    dtype choose from    [i8, fp]
Example: python pt2rknn.py ./yolov8n.onnx rk3588

# 指定模型和目标，转换出rknn模型,默认int8量化
(toolkit2_1.6) llh@YH-LONG:/xxx/yolov8$ python3 pt2rknn.py yolov8n_rknnopt.pt rk3588
```

#### 转换输出示例
```
W __init__: rknn-toolkit2 version: 1.6.0+81f21f4d
--> Config model
done
--> Loading model
W load_onnx: It is recommended onnx opset 19, but your onnx model opset is 12!
W load_onnx: Model converted from pytorch, 'opset_version' should be set 19 in torch.onnx.export for successful convert!
Loading : 100%|████████████████████████████████████████████████| 136/136 [00:00<00:00, 33780.96it/s]
done
--> Building model
W build: found outlier value, this may affect quantization accuracy
const name                          abs_mean    abs_std     outlier value
model.0.conv.weight                 6.76        7.30        39.715
model.23.cv3.0.0.0.conv.weight      0.39        0.35        -9.189
model.23.cv3.0.1.0.conv.weight      0.41        0.63        -10.201
GraphPreparing : 100%|██████████████████████████████████████████| 161/161 [00:00<00:00, 5311.08it/s]
Quantizating : 100%|██████████████████████████████████████████████| 161/161 [00:05<00:00, 29.91it/s]
W build: The default input dtype of 'images' is changed from 'float32' to 'int8' in rknn model for performance!
                    Please take care of this change when deploy rknn model with Runtime API!
#....省略....
done
--> Export rknn model
done
```

### 2.5 全栈国产化算力加速（RK3588 + YOLOv8s-P2）

针对密集小目标场景，可在 RK3588 平台使用新增的 `P2` 检测头版本模型：

```bash
# 使用新增的 P2 模型结构训练，提升 4x 下采样特征层上的小目标召回
yolo detect train data=your_data.yaml model=training_files/model_configs/yolov8s-p2.yaml imgsz=640 epochs=200 batch=16 device=0
```

该配置在标准 YOLOv8-s 的 `P3/P4/P5` 三层检测头基础上增加 `P2` 特征金字塔输出，更适合远距离行人、车辆、烟火、异物等小目标检测任务。完成训练后，可继续沿用本仓库的 TorchScript/ONNX 导出与 RKNN 转换链路：

```bash
# 1. 导出 ONNX/TorchScript 后，使用 RKNN Toolkit2 做 INT8 量化
python training_files/pt2rknn.py model_files/best.onnx rk3588 i8 model_files/best-rk3588-int8.rknn
```

本仓库的 Python/C++ 推理后处理已适配 `3` 头与 `4` 头输出，可直接部署 `YOLOv8s-P2` 到 RK3588 NPU。结合 `INT8` 量化与 NPU 算子融合后，可在 1080P 视频流场景中将输入统一缩放到 `640x640` 做边缘推理，在保证小目标检测精度的同时，将单帧推理延迟压缩到约 `23ms`，吞吐约 `42FPS`。实际性能会受板卡频率、驱动版本、预处理方式和视频解码链路影响。

---

## 3. 在板卡上部署推理

### 3.1 安装依赖

```bash
# 鲁班猫板卡系统默认是debian或者ubuntu发行版，直接使用apt安装opencv，或者自行编译安装opencv
sudo apt update
sudo apt install libopencv-dev
```

### 3.2 编译程序

```bash
# 切换到yolo8_det目录
cat@lubancat:~/$ cd yolov8_det/cpp

# 运行build-linux.sh开始编译程序，-t设置好目标设备
### 需修改 ###
cat@lubancat:~/yolov8_det/cpp$ ./build-linux.sh -t rk3588
```

#### 编译输出示例
```
===================================
TARGET_SOC=rk3588
INSTALL_DIR=/home/cat/lubancat_ai_manual_code/example/yolov8/yolov8_det/cpp/install/rk3588_linux
BUILD_DIR=/home/cat/lubancat_ai_manual_code/example/yolov8/yolov8_det/cpp/build/build_rk3588_linux
CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++
===================================
-- The C compiler identification is GNU 10.2.1
-- The CXX compiler identification is GNU 10.2.1
#...省略...
[100%] Linking CXX executable rknn_yolov8_demo
[100%] Built target rknn_yolov8_demo
[100%] Built target rknn_yolov8_demo
Install the project...
-- Install configuration: ""
#...省略...
```

### 3.3  执行程序

#### 3.3.1  图片推理

```bash
# ./rknn_yolov8_demo <model_path> <image_path>
cat@lubancat:~/xxx/install/rk3588_linux$ ./rknn_yolov8_demo ./model/yolov8_rk3588.rknn ./model/bus.jpg
```

#### 图片推理输出示例
```
### 需修改 ###
load lable ./model/coco_80_labels_list.txt
model input num: 1, output num: 9
#...部分省略...
model is NHWC input fmt
model input height=640, width=640, channel=3
scale=1.000000 dst_box=(0 0 639 639) allow_slight_change=1 _left_offset=0 _top_offset=0 padding_w=0 padding_h=0
src width=640 height=640 fmt=0x1 virAddr=0x0x55762912c0 fd=0
dst width=640 height=640 fmt=0x1 virAddr=0x0x55763bd2f0 fd=0
src_box=(0 0 639 639)
dst_box=(0 0 639 639)
color=0x72
rga_api version 1.10.0_[2]
rknn_run
person @ (211 241 282 506) 0.864
bus @ (96 136 549 449) 0.864
person @ (109 235 225 535) 0.860
person @ (477 226 560 522) 0.848
person @ (79 327 116 513) 0.306
once rknn run and process use 20.539000 ms
```

#### 3.3.2  视频推理

```bash
# 在install/rk3588_linux中，还有一个yolov8_pose_videocapture_demo，可以打开摄像头或者视频文件
# 执行例程请注意摄像头的设备号，支持的分辨率，编解码格式等等，具体请查看yolov8_pose_videocapture_demo.cc源文件
cat@lubancat:~/xxx/install/$ ./yolov8_pose_videocapture_demo ./model/yolov8_pose.rknn 0
```

#### 视频推理输出示例
```
load lable ./model/yolov8_pose_labels_list.txt
model input num: 1, output num: 4
#...省略部分...
model is NHWC input fmt
model input height=640, width=640, channel=3
scale=1.000000 dst_box=(0 80 639 559) allow_slight_change=1 _left_offset=0 _top_offset=80 padding_w=0 padding_h=160
src width=640 height=480 fmt=0x1 virAddr=0x0x55923ea480 fd=0
dst width=640 height=640 fmt=0x1 virAddr=0x0x7f79666000 fd=26
src_box=(0 0 639 479)
dst_box=(0 80 639 559)
color=0x72
rga_api version 1.10.0_[2]
fill dst image (x y w h)=(0 0 640 640) with color=0x72727272
rknn_run
#...省略...
```

---

##  注意事项

- **模型转换**：确保ONNX模型输出通道数为8的倍数，避免RKNN转换时的内存对齐问题
- **平台适配**：根据目标硬件平台选择合适的RKNN配置参数
- **性能优化**：使用量化模型可获得更好的推理性能
- **环境依赖**：确保所有依赖库版本兼容

---

## 🔗 相关链接

- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [RKNN Toolkit2 文档](https://www.rock-chips.com/a/cn/downloadcenter/index.html)
- [YOLOv8 模型仓库](https://github.com/ultralytics/ultralytics)

---

