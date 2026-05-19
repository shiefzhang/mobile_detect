# Mobile Detect

Android CameraX demo for detecting people with a YOLO model, then classifying each detected person with either the helmet classifier or reflective-vest classifier.

## What It Does

- Opens the rear camera.
- Runs `person_detector.onnx` and keeps detections whose class is `person`.
- Crops each detected person and runs the selected classifier:
  - `helmet_classifier.onnx`: red box for `Nohelmet`.
  - `vest_classifier.onnx`: red box for `Novest`.
- Can switch to `standard_detector.onnx`, exported from `yolo11n.pt`, and choose one of the built-in COCO detection classes.
- Draws compliant/other people with blue boxes and violations with red boxes.
- Provides top dropdowns for switching the detector model, safety classifier, and standard-model class.

## Project Layout

- `models/*.pt`: original Ultralytics training checkpoints.
- `models/*.onnx`: exported ONNX models.
- `app/src/main/assets/*.onnx`: ONNX models loaded by the Android app.
- `app/src/main/java/com/example/mobiledetect`: CameraX, ONNX Runtime, preprocessing, postprocessing, and overlay code.

## Build

Open this folder in Android Studio, let Gradle sync, then run the `app` configuration on a physical Android device with a camera.

The local machine used to create the project does not have `gradle` or `ANDROID_HOME` configured, so command-line build verification was not available here.
