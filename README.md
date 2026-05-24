# Mobile Detect

一个基于 Android CameraX 和 ONNX Runtime 的移动端实时检测应用。

应用可以使用手机摄像头进行实时画面分析，先通过 YOLO 检测人员，再对人员区域进行安全帽或反光衣分类判断。也可以切换到标准 YOLO11n 检测模型，按 COCO 内置类别进行通用目标检测。

## 主要功能

- 使用手机摄像头实时预览和推理。
- 支持后置主摄像头和前置摄像头切换。
- 支持双指手势缩放镜头，并在右下角显示当前缩放倍率。
- 支持左侧可收拉设置栏，收起后只显示“设置”按钮。
- 使用人员检测模型 `person_detector.onnx` 检测人员。
- 对检测到的人员进行二次分类：
  - `helmet_classifier.onnx`：判断是否未戴安全帽。
  - `vest_classifier.onnx`：判断是否未穿反光衣。
- 支持切换到标准检测模型 `standard_detector.onnx`。
- 支持选择 YOLO11n 的 COCO 内置检测类别。
- 支持 YOLO11n 实例分割模型 `seg_detector.onnx`。
- 支持 YOLO11n 旋转框模型 `obb_detector.onnx`。
- 支持 YOLO11n 姿态估计模型 `pose_detector.onnx`。
- 支持 YOLO11n 分类模型 `yolo_classifier.onnx`。
- 在标准检测模型识别目标后，会裁剪检测框区域送入分类模型。
- 分类置信度超过 `0.6` 时认为分类成功，并绘制绿色分类框。
- 支持手动选择新的 ONNX 检测模型文件。
- 正常目标使用蓝色框显示。
- 未戴安全帽、未穿反光衣等违规目标使用红色框显示。

## 内置模型

App 当前内置以下 ONNX 模型：

- `app/src/main/assets/person_detector.onnx`
  - 人员检测模型。
  - 当前人员类别 ID 为 `4`。

- `app/src/main/assets/helmet_classifier.onnx`
  - 安全帽分类模型。
  - `Nohelmet` 会被视为违规。

- `app/src/main/assets/vest_classifier.onnx`
  - 反光衣分类模型。
  - `Novest` 会被视为违规。

- `app/src/main/assets/standard_detector.onnx`
  - 由 `yolo11n.pt` 导出的标准 YOLO11n 检测模型。
  - 支持 COCO 80 类目标检测。

- `app/src/main/assets/seg_detector.onnx`
  - YOLO11n 实例分割模型。
  - 支持 COCO 80 类。
  - App 会绘制检测框和半透明分割区域。

- `app/src/main/assets/obb_detector.onnx`
  - YOLO11n OBB 旋转框模型。
  - 支持 DOTA 类别。
  - App 会绘制旋转四边形。

- `app/src/main/assets/pose_detector.onnx`
  - YOLO11n Pose 姿态估计模型。
  - App 会绘制人体检测框、关键点和骨架。

- `app/src/main/assets/yolo_classifier.onnx`
  - YOLO11n 分类模型。
  - 标准检测模型识别目标后，会裁剪目标区域进行二次分类。
  - 分类置信度超过 `0.6` 时显示绿色分类框。

- `app/src/main/assets/cls_labels.txt`
  - 分类模型的 ImageNet 类别名称。

## 自定义模型

侧边栏中可以点击“选择 ONNX 检测模型”手动选择新的检测模型。

注意事项：

- 当前只支持 ONNX 检测模型。
- `.pt` 训练权重需要先导出为 ONNX。
- 推荐导出检测模型时使用 `imgsz=640`。
- 自定义模型需要是 YOLO 检测输出格式。
- 当前自定义模型的类别下拉仍沿用 COCO 80 类；如果模型类别不是 COCO，需要后续扩展类别配置。

推荐导出命令：

```bash
yolo export model=your_model.pt format=onnx imgsz=640 opset=12 simplify=False dynamic=False
```

分类模型建议导出为：

```bash
yolo export model=your_classifier.pt format=onnx imgsz=320 opset=12 simplify=False dynamic=False
```

## 模型导出脚本

项目根目录提供了 `export_onnx.py`，用于把 Ultralytics `.pt` 权重导出为 ONNX、TensorRT Engine 或 TFLite。

查看参数：

```bash
python export_onnx.py --help
```

导出 ONNX：

```bash
python export_onnx.py models/05person_best11m.pt --format onnx --imgsz 640
```

导出分类模型 ONNX：

```bash
python export_onnx.py models/human_hat_cls_v4_bestm.pt --format onnx --imgsz 320
```

导出 Jetson 可用的 TensorRT Engine：

```bash
python export_onnx.py models/05person_best11m.pt --format engine --imgsz 640 --device 0 --half
```

导出 TFLite：

```bash
python export_onnx.py models/human_hat_cls_v4_bestm.pt --format tflite --imgsz 320
```

导出 INT8 TFLite 时需要提供校准数据：

```bash
python export_onnx.py models/05person_best11m.pt --format tflite --imgsz 640 --int8 --data data.yaml
```

常用参数：

- `--format onnx|engine|tflite`：导出格式，默认 `onnx`。
- `--imgsz`：导出输入尺寸，检测模型通常用 `640`，分类模型通常用 `320`。
- `--output-dir`：指定输出目录，避免覆盖原目录模型。
- `--simplify`：简化 ONNX 图，使用前建议先验证 App 或推理端能正常加载。
- `--dynamic`：导出动态输入尺寸。
- `--half`：导出 FP16 模型，Jetson TensorRT Engine 常用。
- `--int8`：导出 INT8 模型，通常需要 `--data` 提供校准数据。
- `--workspace`：TensorRT workspace 大小，单位 GiB。

注意：TensorRT `.engine` 与 Jetson 硬件、CUDA 和 TensorRT 版本强绑定，建议直接在目标 Jetson 设备上生成，不要在其他机器生成后拷贝使用。

## 项目结构

```text
app/src/main/java/com/example/mobiledetect/
├── MainActivity.java      # UI、相机、侧边栏、手势缩放、自定义模型选择
├── ModelRunner.java       # 模型加载、ONNX 推理、YOLO 后处理、分类判断
├── OverlayView.java       # 检测框绘制
├── ImageUtils.java        # CameraX 图像转换
└── Detection.java         # 检测结果数据结构
```

其他重要文件：

- `DETECTION_LOGIC.md`
  - 详细说明模型检测逻辑和后续业务规则扩展方式。

- `export_onnx.py`
  - 本地模型导出辅助脚本，支持 ONNX、TensorRT Engine 和 TFLite。

- `app/src/main/assets/`
  - App 实际加载的 ONNX 模型资源。

## 检测逻辑入口

核心检测逻辑在：

```text
app/src/main/java/com/example/mobiledetect/ModelRunner.java
```

主要入口：

- `runSafety(...)`
  - 人员检测 + 安全帽/反光衣分类。

- `runStandard(...)`
  - 标准 YOLO11n 检测。

- `runSegment(...)`
  - YOLO11n 实例分割检测。

- `runObb(...)`
  - YOLO11n 旋转框检测。

- `runPose(...)`
  - YOLO11n 姿态估计。

- `runCustom(...)`
  - 用户手动选择的 ONNX 检测模型。

如果后续要增加“人员过于密集”“进入危险区域”“车辆靠近人员”等业务规则，优先阅读：

```text
DETECTION_LOGIC.md
```

## 构建运行

1. 使用 Android Studio 打开项目根目录。

```text
D:\work\codex\mobile_detect
```

2. 等待 Gradle Sync 完成。

3. 连接 Android 真机。

4. 手机开启开发者模式和 USB 调试。

5. 在 Android Studio 中选择手机设备，点击 Run。

## 运行环境

- Android Studio
- Android Gradle Plugin 8.5.2
- CameraX 1.4.1
- ONNX Runtime Android 1.20.0
- minSdk 26
- targetSdk 35

## 作者

- Pyrrhus
- zhangxuefeng@batonsoft.com
