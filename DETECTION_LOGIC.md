# 检测逻辑与后续扩展说明

本文档说明当前 App 的模型检测代码位置、运行流程，以及后续如何增加业务规则，例如“人员过于密集”判断。

## 关键文件

- `app/src/main/java/com/example/mobiledetect/MainActivity.java`
  - 相机取帧
  - UI 控件
  - 模型模式选择
  - 调用检测逻辑
  - 手动选择自定义 ONNX 模型

- `app/src/main/java/com/example/mobiledetect/ModelRunner.java`
  - 模型加载
  - ONNX Runtime 推理
  - YOLO 后处理
  - 人员检测
  - 安全帽/反光衣分类
  - 标准 YOLO11n 检测
  - 自定义模型检测

- `app/src/main/java/com/example/mobiledetect/OverlayView.java`
  - 检测框绘制
  - 蓝框/红框显示
  - 前置摄像头镜像绘制

- `app/src/main/java/com/example/mobiledetect/Detection.java`
  - 单个检测结果的数据结构

## 当前整体流程

App 启动后，`MainActivity.onCreate()` 执行：

```java
buildUi();
loadModels();
startCamera();
```

其中：

- `buildUi()` 创建界面和左侧设置栏。
- `loadModels()` 创建 `ModelRunner` 并加载模型。
- `startCamera()` 启动 CameraX。

摄像头每一帧会进入：

```java
private void analyzeFrame(ImageProxy image)
```

这个方法位于 `MainActivity.java`。

流程如下：

1. 将 CameraX 的 `ImageProxy` 转成 `Bitmap`。

```java
bitmap = ImageUtils.imageProxyToBitmap(image);
```

2. 根据当前检测模式选择不同推理入口。

```java
if (selectedDetectorMode == ModelRunner.DetectorMode.SAFETY) {
    detections = runner.runSafety(bitmap, classifierMode);
} else if (selectedDetectorMode == ModelRunner.DetectorMode.CUSTOM) {
    detections = runner.runCustom(bitmap, selectedStandardClassId);
} else {
    detections = runner.runStandard(bitmap, selectedStandardClassId);
}
```

3. 将检测结果传给 `OverlayView` 绘制。

```java
overlayView.setDetections(detections, width, height);
```

## 模型加载逻辑

模型加载在 `ModelRunner` 构造函数中：

```java
ModelRunner(Context context)
```

当前内置模型：

```text
person_detector.onnx
standard_detector.onnx
helmet_classifier.onnx
vest_classifier.onnx
```

自定义模型通过下面的方法加载：

```java
synchronized void loadCustomDetector(File modelFile)
```

该方法会用 ONNX Runtime 创建新的检测模型 Session：

```java
customDetector = env.createSession(modelFile.getAbsolutePath(), options);
```

注意：当前手动选择模型仅支持 ONNX。`.pt` 训练权重需要先导出 ONNX。

## 人员安全检测逻辑

入口方法：

```java
synchronized List<Detection> runSafety(Bitmap frame, ClassifierMode mode)
```

位于 `ModelRunner.java`。

流程：

1. 将原始画面 letterbox 到 `640x640`。
2. 调用人员检测模型。
3. 获取人员框。
4. 对每个人员框裁剪图片。
5. 根据下拉菜单选择安全帽或反光衣分类模型。
6. 判断是否违规。
7. 返回 `List<Detection>`。

当前人员检测调用：

```java
people = detectObjects(
        detector,
        detectorInput,
        letterbox,
        frame.getWidth(),
        frame.getHeight(),
        PERSON_CLASS,
        5,
        PERSON_THRESHOLD,
        "person"
);
```

这里：

```java
private static final int PERSON_CLASS = 4;
```

因为当前人员检测模型类别是：

```text
0 cranebody
1 excavator
2 hook
3 othervehicle
4 person
```

所以 `person` 类别 ID 是 `4`。

分类逻辑在：

```java
private Classification classify(Bitmap frame, RectF box, ClassifierMode mode)
```

安全帽分类中，`Nohelmet` 被视为违规。

反光衣分类中，`Novest` 被视为违规。

当前违规判断：

```java
boolean violation = classification.violation && classification.score >= CLASS_THRESHOLD;
```

最终生成：

```java
new Detection(person.box, classification.score, violation, classification.label)
```

## 标准 YOLO11n 检测逻辑

入口方法：

```java
synchronized List<Detection> runStandard(Bitmap frame, int classId)
```

流程：

1. 使用 `standardDetector`。
2. 按 `classId` 过滤 COCO 类别。
3. 返回检测结果。
4. 标准模型结果默认不是违规，显示蓝框。

类别名称来自：

```java
static final String[] COCO_NAMES = { ... };
```

## 自定义模型检测逻辑

入口方法：

```java
synchronized List<Detection> runCustom(Bitmap frame, int classId)
```

流程和 `runStandard()` 类似，只是使用：

```java
customDetector
customDetectorInput
```

当前自定义模型假设是 YOLO 检测模型 ONNX，输出格式类似：

```text
[1, 84, 8400]
[1, 9, 8400]
```

其中前 4 个值是：

```text
cx, cy, w, h
```

后面是类别置信度。

## YOLO 后处理逻辑

核心方法：

```java
private List<PersonBox> detectObjects(...)
```

该方法负责：

1. 将 `Bitmap` 转为 ONNX Tensor。

```java
bitmapToTensor(letterbox.bitmap, DETECTOR_SIZE)
```

2. 执行 ONNX 推理。

```java
session.run(Collections.singletonMap(inputName, input))
```

3. 将 YOLO 输出整理成一行一个检测框。

```java
float[][] raw = toDetectionRows(value);
```

4. 读取目标类别置信度。

```java
float score = row[4 + targetClass];
```

5. 过滤低置信度框。

```java
if (score < threshold) {
    continue;
}
```

6. 将中心点格式转成左上右下坐标。

```java
RectF box = new RectF(
        (cx - w * 0.5f - letterbox.padX) / letterbox.scale,
        (cy - h * 0.5f - letterbox.padY) / letterbox.scale,
        (cx + w * 0.5f - letterbox.padX) / letterbox.scale,
        (cy + h * 0.5f - letterbox.padY) / letterbox.scale
);
```

7. 对检测框做 NMS 去重。

```java
return nms(candidates);
```

## Detection 数据结构

`Detection.java` 当前内容：

```java
final class Detection {
    final RectF box;
    final float score;
    final boolean violation;
    final String label;
}
```

字段含义：

- `box`：检测框坐标。
- `score`：置信度。
- `violation`：是否违规。`true` 会画红框，`false` 会画蓝框。
- `label`：显示在框上的文字。

## 绘制逻辑

检测框绘制在：

```java
OverlayView.onDraw(Canvas canvas)
```

颜色判断：

```java
boxPaint.setColor(detection.violation ? Color.rgb(220, 38, 38) : Color.rgb(37, 99, 235));
```

含义：

- `violation == true`：红框。
- `violation == false`：蓝框。

如果以后要增加更多颜色，可以扩展 `Detection`，例如：

```java
final int color;
final String rule;
```

然后在 `OverlayView` 中按 `detection.color` 绘制。

## 人员过于密集逻辑写在哪里

短期建议写在：

```java
ModelRunner.runSafety()
```

原因：

- 这里已经拿到了所有人员框。
- 这里正在生成最终 `Detection`。
- 人员密集属于业务规则，不属于 YOLO 基础后处理。

当前 `runSafety()` 中有类似逻辑：

```java
List<Detection> detections = new ArrayList<>();
for (PersonBox person : people) {
    Classification classification = classify(frame, person.box, mode);
    boolean violation = classification.violation && classification.score >= CLASS_THRESHOLD;
    detections.add(new Detection(person.box, classification.score, violation, classification.label));
}
return detections;
```

可以在 `people` 检测出来之后插入密集判断。

## 示例 1：按人数和面积判断人员过密

在 `ModelRunner.java` 中新增常量：

```java
private static final int CROWD_COUNT_THRESHOLD = 6;
private static final float CROWD_AREA_RATIO_THRESHOLD = 0.35f;
```

新增方法：

```java
private boolean isCrowded(List<PersonBox> people, int frameWidth, int frameHeight) {
    if (people.size() < CROWD_COUNT_THRESHOLD) {
        return false;
    }

    float totalArea = 0f;
    for (PersonBox person : people) {
        totalArea += person.box.width() * person.box.height();
    }

    float frameArea = frameWidth * frameHeight;
    float areaRatio = totalArea / frameArea;

    return areaRatio >= CROWD_AREA_RATIO_THRESHOLD;
}
```

修改 `runSafety()`：

```java
boolean crowded = isCrowded(people, frame.getWidth(), frame.getHeight());

List<Detection> detections = new ArrayList<>();
for (PersonBox person : people) {
    Classification classification = classify(frame, person.box, mode);
    boolean safetyViolation = classification.violation && classification.score >= CLASS_THRESHOLD;
    boolean violation = crowded || safetyViolation;
    String label = crowded ? "Crowded" : classification.label;
    detections.add(new Detection(person.box, classification.score, violation, label));
}
return detections;
```

效果：

- 画面里人员数量达到阈值。
- 人员框面积总和达到画面比例阈值。
- 满足条件时所有人员标红。

## 示例 2：按人员中心点距离判断人员过密

这种方式比单纯人数更合理。

新增常量：

```java
private static final float CLOSE_DISTANCE_RATIO = 0.12f;
private static final int CLOSE_PAIR_THRESHOLD = 4;
```

新增方法：

```java
private boolean isCrowdedByDistance(List<PersonBox> people, int frameWidth, int frameHeight) {
    int closePairs = 0;
    float reference = Math.min(frameWidth, frameHeight);

    for (int i = 0; i < people.size(); i++) {
        RectF a = people.get(i).box;
        float ax = a.centerX();
        float ay = a.centerY();

        for (int j = i + 1; j < people.size(); j++) {
            RectF b = people.get(j).box;
            float bx = b.centerX();
            float by = b.centerY();

            float dx = ax - bx;
            float dy = ay - by;
            float distance = (float) Math.sqrt(dx * dx + dy * dy);

            if (distance < reference * CLOSE_DISTANCE_RATIO) {
                closePairs++;
            }
        }
    }

    return closePairs >= CLOSE_PAIR_THRESHOLD;
}
```

在 `runSafety()` 中使用：

```java
boolean crowded = isCrowdedByDistance(people, frame.getWidth(), frame.getHeight());
```

## 示例 3：只标红密集区域中的人员

如果不想所有人都红框，可以逐个人判断附近人数。

新增方法：

```java
private boolean isPersonInCrowd(PersonBox target, List<PersonBox> people, int frameWidth, int frameHeight) {
    int nearby = 0;
    float reference = Math.min(frameWidth, frameHeight);

    for (PersonBox other : people) {
        if (other == target) {
            continue;
        }

        float dx = target.box.centerX() - other.box.centerX();
        float dy = target.box.centerY() - other.box.centerY();
        float distance = (float) Math.sqrt(dx * dx + dy * dy);

        if (distance < reference * 0.12f) {
            nearby++;
        }
    }

    return nearby >= 2;
}
```

修改 `runSafety()`：

```java
List<Detection> detections = new ArrayList<>();
for (PersonBox person : people) {
    Classification classification = classify(frame, person.box, mode);
    boolean safetyViolation = classification.violation && classification.score >= CLASS_THRESHOLD;
    boolean crowdViolation = isPersonInCrowd(person, people, frame.getWidth(), frame.getHeight());

    boolean violation = safetyViolation || crowdViolation;
    String label = crowdViolation ? "Crowded" : classification.label;

    detections.add(new Detection(person.box, classification.score, violation, label));
}
return detections;
```

效果：

- 只标红附近人员过多的人。
- 不会因为远处有很多人导致全画面红框。

## 后续修改模型逻辑的推荐步骤

### 1. 确定要改哪一层

- 改模型文件加载：
  - `ModelRunner` 构造方法
  - `loadCustomDetector()`

- 改 YOLO 输出解析：
  - `detectObjects()`
  - `toDetectionRows()`

- 改人员安全业务逻辑：
  - `runSafety()`

- 改标准 YOLO11n 逻辑：
  - `runStandard()`

- 改自定义模型逻辑：
  - `runCustom()`

- 改画框颜色、文字、样式：
  - `OverlayView.onDraw()`

- 改 UI 按钮、下拉菜单、侧边栏：
  - `MainActivity.buildSettingsDrawer()`

### 2. 新增单帧规则

例如：

- 人员过密
- 人员进入危险区域
- 车辆靠近人员
- 检测到某类物体后报警

建议写在 `ModelRunner.runSafety()` 或 `runStandard()` / `runCustom()` 中。

如果规则依赖人员安全检测，优先写在 `runSafety()`。

### 3. 新增跨帧规则

例如：

- 未戴安全帽持续 3 秒才报警
- 人员停留危险区域超过 N 秒
- 同一个目标连续多帧出现才确认

这种规则需要保存历史状态，不建议只写在当前帧的 `runSafety()` 中。

推荐方式：

- 在 `MainActivity.analyzeFrame()` 附近维护状态。
- 或新增 `SafetyRuleEngine.java` 保存历史帧状态。

### 4. 扩展 Detection

如果需要更复杂的报警信息，可以扩展 `Detection.java`。

例如：

```java
final String rule;
final int color;
final long timestamp;
```

然后在 `OverlayView` 中根据这些字段绘制不同样式。

## 推荐长期结构

如果后续规则越来越多，建议新增：

```text
SafetyRuleEngine.java
```

示例结构：

```java
final class SafetyRuleEngine {
    boolean isCrowded(List<ModelRunner.PersonBox> people, int width, int height) {
        ...
    }

    boolean isInDangerZone(RectF personBox, int width, int height) {
        ...
    }
}
```

当前 `PersonBox` 是 `ModelRunner` 的私有内部类：

```java
private static final class PersonBox
```

如果要让外部规则类访问它，需要：

1. 将 `PersonBox` 独立成普通类；或
2. 将人员框先转换成 `Detection`；或
3. 将规则方法暂时写在 `ModelRunner.java` 内部。

短期最简单：

```text
直接把人员过密逻辑写在 ModelRunner.java
```

长期更清晰：

```text
抽出 SafetyRuleEngine.java
```

## 自定义模型注意事项

当前自定义模型加载路径：

```java
MainActivity.onCustomModelPicked(Uri uri)
```

用户选择模型后：

1. App 复制文件到内部存储：

```java
copyUriToInternalFile(uri, "custom_detector.onnx")
```

2. 调用：

```java
runner.loadCustomDetector(modelFile)
```

3. 自动切换到：

```java
ModelRunner.DetectorMode.CUSTOM
```

注意事项：

- 只支持 ONNX。
- 模型应为 YOLO 检测模型。
- 当前类别名称仍沿用 COCO 80 类。
- 如果自定义模型类别不是 COCO，需要后续增加“自定义类别名称配置”。

## 常见修改示例

### 修改检测置信度

在 `ModelRunner.java` 中改：

```java
private static final float PERSON_THRESHOLD = 0.35f;
private static final float STANDARD_THRESHOLD = 0.35f;
private static final float CLASS_THRESHOLD = 0.45f;
```

含义：

- `PERSON_THRESHOLD`：人员检测置信度。
- `STANDARD_THRESHOLD`：标准模型和自定义模型检测置信度。
- `CLASS_THRESHOLD`：安全帽/反光衣分类置信度。

### 修改 NMS 阈值

```java
private static final float NMS_THRESHOLD = 0.45f;
```

阈值越低，重复框过滤越强。

### 修改红框/蓝框颜色

在 `OverlayView.java` 中修改：

```java
boxPaint.setColor(detection.violation ? Color.rgb(220, 38, 38) : Color.rgb(37, 99, 235));
```

### 修改显示文字

在生成 `Detection` 时修改 `label`。

例如：

```java
new Detection(person.box, classification.score, violation, "No Helmet")
```

## 建议新增 SafetyRuleEngine 的完整做法

当业务规则越来越多时，不建议把所有规则都继续塞进 `ModelRunner.java`。

更推荐新增：

```text
app/src/main/java/com/example/mobiledetect/SafetyRuleEngine.java
```

它专门负责业务规则判断，例如：

- 人员是否过密
- 人员是否进入危险区域
- 车辆是否靠近人员
- 某个违规是否持续超过 N 秒
- 多种规则的优先级合并

### 第一步：新增规则结果类型

可以新增：

```text
app/src/main/java/com/example/mobiledetect/RuleResult.java
```

示例：

```java
package com.example.mobiledetect;

final class RuleResult {
    final boolean violation;
    final String label;

    RuleResult(boolean violation, String label) {
        this.violation = violation;
        this.label = label;
    }
}
```

后续如果要更复杂，可以继续加字段：

```java
final int color;
final String ruleCode;
final float severity;
```

### 第二步：新增 SafetyRuleEngine

示例：

```java
package com.example.mobiledetect;

import android.graphics.RectF;

import java.util.List;

final class SafetyRuleEngine {
    private static final float CLOSE_DISTANCE_RATIO = 0.12f;
    private static final int NEARBY_PERSON_THRESHOLD = 2;

    RuleResult evaluatePersonCrowding(RectF target, List<RectF> people, int frameWidth, int frameHeight) {
        int nearby = 0;
        float reference = Math.min(frameWidth, frameHeight);

        for (RectF other : people) {
            if (other == target) {
                continue;
            }

            float dx = target.centerX() - other.centerX();
            float dy = target.centerY() - other.centerY();
            float distance = (float) Math.sqrt(dx * dx + dy * dy);

            if (distance < reference * CLOSE_DISTANCE_RATIO) {
                nearby++;
            }
        }

        if (nearby >= NEARBY_PERSON_THRESHOLD) {
            return new RuleResult(true, "Crowded");
        }

        return new RuleResult(false, "");
    }
}
```

### 第三步：让 ModelRunner 调用 SafetyRuleEngine

在 `ModelRunner.java` 中新增成员：

```java
private final SafetyRuleEngine ruleEngine = new SafetyRuleEngine();
```

在 `runSafety()` 中，先把 `people` 转成 `RectF` 列表：

```java
List<RectF> personBoxes = new ArrayList<>();
for (PersonBox person : people) {
    personBoxes.add(person.box);
}
```

然后在循环中调用规则：

```java
for (PersonBox person : people) {
    Classification classification = classify(frame, person.box, mode);

    boolean safetyViolation = classification.violation && classification.score >= CLASS_THRESHOLD;
    RuleResult crowdResult = ruleEngine.evaluatePersonCrowding(
            person.box,
            personBoxes,
            frame.getWidth(),
            frame.getHeight()
    );

    boolean violation = safetyViolation || crowdResult.violation;
    String label = crowdResult.violation ? crowdResult.label : classification.label;

    detections.add(new Detection(person.box, classification.score, violation, label));
}
```

这样 `ModelRunner` 仍然负责模型推理，`SafetyRuleEngine` 负责业务规则。

## 如果要支持危险区域判断

危险区域通常是画面中的一个多边形或矩形区域。

最简单版本可以先用矩形：

```java
private boolean isInDangerZone(RectF personBox, int frameWidth, int frameHeight) {
    RectF dangerZone = new RectF(
            frameWidth * 0.25f,
            frameHeight * 0.55f,
            frameWidth * 0.75f,
            frameHeight * 0.95f
    );

    return RectF.intersects(personBox, dangerZone);
}
```

在 `runSafety()` 中：

```java
boolean dangerViolation = isInDangerZone(person.box, frame.getWidth(), frame.getHeight());
boolean violation = safetyViolation || crowdViolation || dangerViolation;
String label = dangerViolation ? "Danger Zone" : classification.label;
```

如果要画出危险区域，需要修改 `OverlayView`，增加区域绘制逻辑。

推荐新增字段：

```java
private RectF dangerZone;
```

并新增方法：

```java
public void setDangerZone(RectF dangerZone) {
    this.dangerZone = dangerZone;
    invalidate();
}
```

然后在 `onDraw()` 中先画区域，再画检测框。

## 如果要支持“持续 N 秒才报警”

这类逻辑属于跨帧规则，不适合只写在 `runSafety()` 当前帧里。

例如：

- 未戴安全帽持续 3 秒才报警
- 人员进入危险区域超过 5 秒才报警
- 人员过密持续 2 秒才报警

需要保存历史状态。

### 简化实现方式

可以在 `SafetyRuleEngine` 中维护时间：

```java
private long crowdStartTime = 0L;
```

示例：

```java
boolean confirmCrowded(boolean crowdedNow) {
    long now = System.currentTimeMillis();

    if (!crowdedNow) {
        crowdStartTime = 0L;
        return false;
    }

    if (crowdStartTime == 0L) {
        crowdStartTime = now;
        return false;
    }

    return now - crowdStartTime >= 2000L;
}
```

然后：

```java
boolean crowdedNow = isCrowdedByDistance(people, frame.getWidth(), frame.getHeight());
boolean confirmedCrowded = confirmCrowded(crowdedNow);
```

这样只有连续拥挤超过 2 秒才报警。

### 更准确的实现方式

如果要对单个人持续追踪，需要目标跟踪 ID。

当前代码没有目标跟踪，只是逐帧检测。因此同一个人每一帧没有稳定 ID。

如果后续要做稳定的跨帧人员规则，建议新增：

```text
ObjectTracker.java
```

它可以根据检测框 IoU 或中心点距离给目标分配临时 ID。

简单思路：

1. 当前帧检测出多个框。
2. 与上一帧的框做 IoU 匹配。
3. IoU 高于阈值时认为是同一目标。
4. 给目标保留 `trackId`。
5. 针对 `trackId` 保存违规开始时间。

## 如果要支持自定义模型类别名称

当前自定义模型仍然使用：

```java
ModelRunner.COCO_NAMES
```

也就是说类别下拉菜单仍是 COCO 80 类。

如果自定义模型不是 COCO 类别，例如：

```text
0 helmet
1 no_helmet
2 vest
3 no_vest
```

需要增加类别配置功能。

### 简单方案：代码里写死自定义类别

在 `ModelRunner.java` 新增：

```java
static final String[] CUSTOM_NAMES = {
        "helmet",
        "no_helmet",
        "vest",
        "no_vest"
};
```

然后 `MainActivity` 中自定义模型模式下，让类别下拉改用 `CUSTOM_NAMES`。

这需要把 `Spinner` 的 adapter 保存成成员变量，切换模型时替换数据源。

### 更通用方案：选择 names.txt

允许用户选择一个文本文件：

```text
helmet
no_helmet
vest
no_vest
```

App 读取后更新类别列表。

大致步骤：

1. 新增一个“选择类别文件”按钮。
2. 用系统文件选择器选择 `.txt`。
3. 一行一个类别名读取成 `String[]`。
4. 更新 `standardClassSpinner` 的 adapter。
5. `runCustom()` 使用新的类别数量。

此时 `runCustom()` 不应该继续写死：

```java
COCO_NAMES.length
```

而应该传入当前类别数组长度。

## 如果要支持不同输入尺寸的模型

当前代码固定：

```java
private static final int DETECTOR_SIZE = 640;
private static final int CLASSIFIER_SIZE = 320;
```

这意味着：

- 检测模型按 `640x640` 输入。
- 分类模型按 `320x320` 输入。

如果自定义模型是 `1280x1280` 或 `320x320`，当前代码仍会传 `640x640`，可能导致模型加载能成功，但推理时报输入尺寸错误。

要支持不同尺寸，有两种方案。

### 方案一：约定所有检测模型都导出为 640

这是最简单稳定的方案。

导出时使用：

```bash
yolo export model=xxx.pt format=onnx imgsz=640
```

当前 App 就是按这个方式设计的。

### 方案二：读取 ONNX 输入 shape

在 `loadCustomDetector()` 中读取输入 shape。

ONNX Runtime 可以拿到输入信息：

```java
customDetector.getInputInfo()
```

然后解析出：

```text
[1, 3, H, W]
```

再把自定义模型的输入尺寸保存下来。

这样 `runCustom()` 需要使用自定义尺寸：

```java
Letterbox letterbox = letterbox(frame, customDetectorSize);
bitmapToTensor(letterbox.bitmap, customDetectorSize);
```

这会让 `detectObjects()` 多一个 `inputSize` 参数。

## 如果要支持不同 YOLO 输出格式

当前 `detectObjects()` 假设输出是 YOLO11/YOLOv8 常见格式：

```text
[1, classes + 4, anchors]
```

或整理后：

```text
[anchors, classes + 4]
```

每行格式：

```text
cx, cy, w, h, class0_score, class1_score, ...
```

如果你的模型输出是下面这种格式：

```text
x1, y1, x2, y2, score, classId
```

就不能直接用当前 `detectObjects()`。

需要新增另一个后处理方法，例如：

```java
private List<PersonBox> detectObjectsXyxy(...)
```

并在自定义模型加载时选择对应解析方式。

后续可以增加一个下拉菜单：

```text
模型输出格式：
- YOLO cxcywh
- XYXY score class
```

然后 `runCustom()` 根据选择调用不同解析方法。

## 调试模型输出的建议

如果自定义模型加载成功但没有检测框，优先检查：

1. 输入尺寸是否是 `640x640`。
2. 输出格式是否是 YOLO11/YOLOv8 检测格式。
3. 类别 ID 是否选对。
4. 置信度阈值是否太高。
5. 模型是否已经包含 NMS。

可以临时降低阈值：

```java
private static final float STANDARD_THRESHOLD = 0.15f;
```

如果降低阈值后出现大量框，说明模型能跑，只是阈值或类别选择问题。

如果仍没有框，可能是：

- 输入尺寸不匹配。
- 输出格式不匹配。
- 前处理归一化不一致。
- 模型不是检测模型。

## 推荐的修改顺序

如果以后要加一个新规则，建议按这个顺序做：

1. 先确认检测框是否正确。
2. 在 `ModelRunner.runSafety()` 中拿到所需目标框。
3. 写一个独立的规则方法，例如 `isCrowdedByDistance()`。
4. 将规则结果合并到 `Detection.violation`。
5. 修改 `label` 显示规则名称。
6. 如需不同颜色，再扩展 `Detection`。
7. 如需跨帧，再抽出 `SafetyRuleEngine` 保存状态。
8. 最后再改 UI，增加开关、阈值输入等配置。

## 当前代码中最适合改的入口总结

| 需求 | 修改位置 |
| --- | --- |
| 修改人员检测阈值 | `ModelRunner.PERSON_THRESHOLD` |
| 修改标准/自定义模型阈值 | `ModelRunner.STANDARD_THRESHOLD` |
| 修改分类阈值 | `ModelRunner.CLASS_THRESHOLD` |
| 修改 NMS | `ModelRunner.NMS_THRESHOLD` |
| 增加人员过密判断 | `ModelRunner.runSafety()` |
| 增加危险区域判断 | `ModelRunner.runSafety()` + `OverlayView` |
| 增加自定义类别 | `MainActivity` + `ModelRunner.runCustom()` |
| 增加跨帧报警 | 新增 `SafetyRuleEngine` 或 `ObjectTracker` |
| 修改框颜色 | `OverlayView.onDraw()` |
| 修改侧边栏设置项 | `MainActivity.buildSettingsDrawer()` |
| 修改自定义模型加载 | `MainActivity.onCustomModelPicked()` + `ModelRunner.loadCustomDetector()` |

## 最小可行的人员过密改法

如果现在只想最快加上“人员过密红框”，建议先不要新建类，直接在 `ModelRunner.java` 中做。

步骤：

1. 在常量区域加入：

```java
private static final float CLOSE_DISTANCE_RATIO = 0.12f;
private static final int NEARBY_PERSON_THRESHOLD = 2;
```

2. 在 `ModelRunner.java` 中加入：

```java
private boolean isPersonInCrowd(PersonBox target, List<PersonBox> people, int frameWidth, int frameHeight) {
    int nearby = 0;
    float reference = Math.min(frameWidth, frameHeight);

    for (PersonBox other : people) {
        if (other == target) {
            continue;
        }

        float dx = target.box.centerX() - other.box.centerX();
        float dy = target.box.centerY() - other.box.centerY();
        float distance = (float) Math.sqrt(dx * dx + dy * dy);

        if (distance < reference * CLOSE_DISTANCE_RATIO) {
            nearby++;
        }
    }

    return nearby >= NEARBY_PERSON_THRESHOLD;
}
```

3. 修改 `runSafety()` 中的循环：

```java
for (PersonBox person : people) {
    Classification classification = classify(frame, person.box, mode);
    boolean safetyViolation = classification.violation && classification.score >= CLASS_THRESHOLD;
    boolean crowdViolation = isPersonInCrowd(person, people, frame.getWidth(), frame.getHeight());

    boolean violation = safetyViolation || crowdViolation;
    String label = crowdViolation ? "Crowded" : classification.label;

    detections.add(new Detection(person.box, classification.score, violation, label));
}
```

这个改法影响面最小，适合先验证业务效果。
