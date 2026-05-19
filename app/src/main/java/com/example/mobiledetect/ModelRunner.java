package com.example.mobiledetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

final class ModelRunner implements AutoCloseable {
    enum ClassifierMode {
        HELMET, VEST
    }

    enum DetectorMode {
        SAFETY, STANDARD, SEGMENT, OBB, POSE, CUSTOM
    }

    private static final int DETECTOR_SIZE = 640;
    private static final int CLASSIFIER_SIZE = 320;
    private static final int PERSON_CLASS = 4;
    private static final float PERSON_THRESHOLD = 0.35f;
    private static final float STANDARD_THRESHOLD = 0.35f;
    private static final float NMS_THRESHOLD = 0.45f;
    private static final float CLASS_THRESHOLD = 0.45f;
    private static final float GENERIC_CLASS_THRESHOLD = 0.60f;
    static final String[] COCO_NAMES = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
    };
    static final String[] OBB_NAMES = {
            "plane", "ship", "storage tank", "baseball diamond", "tennis court",
            "basketball court", "ground track field", "harbor", "bridge",
            "large vehicle", "small vehicle", "helicopter", "roundabout",
            "soccer ball field", "swimming pool"
    };

    private final OrtEnvironment env;
    private final OrtSession detector;
    private final OrtSession standardDetector;
    private final OrtSession segDetector;
    private final OrtSession obbDetector;
    private final OrtSession poseDetector;
    private final OrtSession genericClassifier;
    private final OrtSession helmetClassifier;
    private final OrtSession vestClassifier;
    private final String detectorInput;
    private final String standardDetectorInput;
    private final String segDetectorInput;
    private final String obbDetectorInput;
    private final String poseDetectorInput;
    private final String genericClassifierInput;
    private final String helmetInput;
    private final String vestInput;
    private final String[] genericClassNames;
    private OrtSession customDetector;
    private String customDetectorInput;

    ModelRunner(Context context) throws Exception {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        detector = createSession(context, options, "person_detector.onnx");
        standardDetector = createSession(context, options, "standard_detector.onnx");
        segDetector = createSession(context, options, "seg_detector.onnx");
        obbDetector = createSession(context, options, "obb_detector.onnx");
        poseDetector = createSession(context, options, "pose_detector.onnx");
        genericClassifier = createSession(context, options, "yolo_classifier.onnx");
        helmetClassifier = createSession(context, options, "helmet_classifier.onnx");
        vestClassifier = createSession(context, options, "vest_classifier.onnx");
        detectorInput = detector.getInputNames().iterator().next();
        standardDetectorInput = standardDetector.getInputNames().iterator().next();
        segDetectorInput = segDetector.getInputNames().iterator().next();
        obbDetectorInput = obbDetector.getInputNames().iterator().next();
        poseDetectorInput = poseDetector.getInputNames().iterator().next();
        genericClassifierInput = genericClassifier.getInputNames().iterator().next();
        helmetInput = helmetClassifier.getInputNames().iterator().next();
        vestInput = vestClassifier.getInputNames().iterator().next();
        genericClassNames = loadLabels(context, "cls_labels.txt");
    }

    synchronized List<Detection> runSafety(Bitmap frame, ClassifierMode mode) throws OrtException {
        Letterbox letterbox = letterbox(frame, DETECTOR_SIZE);
        List<PersonBox> people;
        try {
            people = detectObjects(detector, detectorInput, letterbox, frame.getWidth(), frame.getHeight(),
                    PERSON_CLASS, 5, PERSON_THRESHOLD, "person");
        } finally {
            letterbox.bitmap.recycle();
        }
        List<Detection> detections = new ArrayList<>();
        for (PersonBox person : people) {
            Classification classification = classify(frame, person.box, mode);
            boolean violation = classification.violation && classification.score >= CLASS_THRESHOLD;
            detections.add(new Detection(person.box, classification.score, violation, classification.label));
        }
        return detections;
    }

    synchronized List<Detection> runStandard(Bitmap frame, int classId) throws OrtException {
        int safeClassId = clamp(classId, 0, COCO_NAMES.length - 1);
        Letterbox letterbox = letterbox(frame, DETECTOR_SIZE);
        List<PersonBox> boxes;
        try {
            boxes = detectObjects(standardDetector, standardDetectorInput, letterbox, frame.getWidth(), frame.getHeight(),
                    safeClassId, COCO_NAMES.length, STANDARD_THRESHOLD, COCO_NAMES[safeClassId]);
        } finally {
            letterbox.bitmap.recycle();
        }
        List<Detection> detections = new ArrayList<>();
        for (PersonBox box : boxes) {
            detections.add(new Detection(box.box, box.score, false, box.label));
            Classification classification = classifyGeneric(frame, box.box);
            if (classification.score >= GENERIC_CLASS_THRESHOLD) {
                detections.add(new Detection(insetBox(box.box, 4f), classification.score, false,
                        "cls: " + classification.label, Color.rgb(34, 197, 94)));
            }
        }
        return detections;
    }

    synchronized List<Detection> runSegment(Bitmap frame, int classId) throws OrtException {
        int safeClassId = clamp(classId, 0, COCO_NAMES.length - 1);
        Letterbox letterbox = letterbox(frame, DETECTOR_SIZE);
        try {
            return detectSegments(letterbox, frame.getWidth(), frame.getHeight(), safeClassId);
        } finally {
            letterbox.bitmap.recycle();
        }
    }

    synchronized List<Detection> runObb(Bitmap frame, int classId) throws OrtException {
        int safeClassId = clamp(classId, 0, OBB_NAMES.length - 1);
        Letterbox letterbox = letterbox(frame, DETECTOR_SIZE);
        try {
            return detectObb(letterbox, frame.getWidth(), frame.getHeight(), safeClassId);
        } finally {
            letterbox.bitmap.recycle();
        }
    }

    synchronized List<Detection> runPose(Bitmap frame) throws OrtException {
        Letterbox letterbox = letterbox(frame, DETECTOR_SIZE);
        try {
            return detectPose(letterbox, frame.getWidth(), frame.getHeight());
        } finally {
            letterbox.bitmap.recycle();
        }
    }

    synchronized List<Detection> runCustom(Bitmap frame, int classId) throws OrtException {
        if (customDetector == null || customDetectorInput == null) {
            return Collections.emptyList();
        }
        int safeClassId = clamp(classId, 0, COCO_NAMES.length - 1);
        Letterbox letterbox = letterbox(frame, DETECTOR_SIZE);
        List<PersonBox> boxes;
        try {
            boxes = detectObjects(customDetector, customDetectorInput, letterbox, frame.getWidth(), frame.getHeight(),
                    safeClassId, COCO_NAMES.length, STANDARD_THRESHOLD, COCO_NAMES[safeClassId]);
        } finally {
            letterbox.bitmap.recycle();
        }
        List<Detection> detections = new ArrayList<>();
        for (PersonBox box : boxes) {
            detections.add(new Detection(box.box, box.score, false, box.label));
        }
        return detections;
    }

    synchronized void loadCustomDetector(File modelFile) throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        OrtSession newSession = env.createSession(modelFile.getAbsolutePath(), options);
        String newInput = newSession.getInputNames().iterator().next();
        OrtSession oldSession = customDetector;
        customDetector = newSession;
        customDetectorInput = newInput;
        if (oldSession != null) {
            oldSession.close();
        }
    }

    private List<PersonBox> detectObjects(OrtSession session, String inputName, Letterbox letterbox,
                                          int frameWidth, int frameHeight, int targetClass,
                                          int classCount, float threshold, String label) throws OrtException {
        try (OnnxTensor input = bitmapToTensor(letterbox.bitmap, DETECTOR_SIZE);
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, input))) {
            Object value = result.get(0).getValue();
            float[][] raw = toDetectionRows(value);
            List<PersonBox> candidates = new ArrayList<>();
            for (float[] row : raw) {
                if (row.length < 4 + classCount) {
                    continue;
                }
                float score = row[4 + targetClass];
                if (score < threshold) {
                    continue;
                }
                float cx = row[0];
                float cy = row[1];
                float w = row[2];
                float h = row[3];
                RectF box = new RectF(
                        (cx - w * 0.5f - letterbox.padX) / letterbox.scale,
                        (cy - h * 0.5f - letterbox.padY) / letterbox.scale,
                        (cx + w * 0.5f - letterbox.padX) / letterbox.scale,
                        (cy + h * 0.5f - letterbox.padY) / letterbox.scale
                );
                box.left = Math.max(0, Math.min(frameWidth, box.left));
                box.top = Math.max(0, Math.min(frameHeight, box.top));
                box.right = Math.max(0, Math.min(frameWidth, box.right));
                box.bottom = Math.max(0, Math.min(frameHeight, box.bottom));
                if (box.width() >= 8 && box.height() >= 8) {
                    candidates.add(new PersonBox(box, score, label));
                }
            }
            return nms(candidates);
        }
    }

    private RectF decodeBox(float[] row, Letterbox letterbox, int frameWidth, int frameHeight) {
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        RectF box = new RectF(
                (cx - w * 0.5f - letterbox.padX) / letterbox.scale,
                (cy - h * 0.5f - letterbox.padY) / letterbox.scale,
                (cx + w * 0.5f - letterbox.padX) / letterbox.scale,
                (cy + h * 0.5f - letterbox.padY) / letterbox.scale
        );
        box.left = Math.max(0, Math.min(frameWidth, box.left));
        box.top = Math.max(0, Math.min(frameHeight, box.top));
        box.right = Math.max(0, Math.min(frameWidth, box.right));
        box.bottom = Math.max(0, Math.min(frameHeight, box.bottom));
        return box;
    }

    private List<Detection> detectSegments(Letterbox letterbox, int frameWidth, int frameHeight, int targetClass)
            throws OrtException {
        try (OnnxTensor input = bitmapToTensor(letterbox.bitmap, DETECTOR_SIZE);
             OrtSession.Result result = segDetector.run(Collections.singletonMap(segDetectorInput, input))) {
            float[][] raw = toDetectionRows(result.get(0).getValue());
            float[][][][] proto = (float[][][][]) result.get(1).getValue();
            List<SegmentCandidate> candidates = new ArrayList<>();
            for (float[] row : raw) {
                if (row.length < 116) {
                    continue;
                }
                float score = row[4 + targetClass];
                if (score < STANDARD_THRESHOLD) {
                    continue;
                }
                RectF box = decodeBox(row, letterbox, frameWidth, frameHeight);
                if (box.width() < 8 || box.height() < 8) {
                    continue;
                }
                float[] coeffs = new float[32];
                System.arraycopy(row, 84, coeffs, 0, coeffs.length);
                candidates.add(new SegmentCandidate(box, score, COCO_NAMES[targetClass], coeffs));
            }
            List<SegmentCandidate> kept = nmsSegments(candidates);
            List<Detection> detections = new ArrayList<>();
            for (SegmentCandidate candidate : kept) {
                List<RectF> cells = buildMaskCells(candidate.box, candidate.coeffs, proto[0], letterbox, frameWidth, frameHeight);
                detections.add(new Detection(candidate.box, candidate.score, false, candidate.label, null, null, cells));
            }
            return detections;
        }
    }

    private List<Detection> detectObb(Letterbox letterbox, int frameWidth, int frameHeight, int targetClass)
            throws OrtException {
        try (OnnxTensor input = bitmapToTensor(letterbox.bitmap, DETECTOR_SIZE);
             OrtSession.Result result = obbDetector.run(Collections.singletonMap(obbDetectorInput, input))) {
            float[][] raw = toDetectionRows(result.get(0).getValue());
            List<Detection> candidates = new ArrayList<>();
            for (float[] row : raw) {
                if (row.length < 20) {
                    continue;
                }
                float score = row[4 + targetClass];
                if (score < STANDARD_THRESHOLD) {
                    continue;
                }
                float angle = row[19];
                PointF[] polygon = decodeObbPolygon(row[0], row[1], row[2], row[3], angle, letterbox, frameWidth, frameHeight);
                RectF bounds = boundsOf(polygon, frameWidth, frameHeight);
                if (bounds.width() < 8 || bounds.height() < 8) {
                    continue;
                }
                candidates.add(new Detection(bounds, score, false, OBB_NAMES[targetClass], polygon, null, null));
            }
            return nmsDetections(candidates);
        }
    }

    private List<Detection> detectPose(Letterbox letterbox, int frameWidth, int frameHeight) throws OrtException {
        try (OnnxTensor input = bitmapToTensor(letterbox.bitmap, DETECTOR_SIZE);
             OrtSession.Result result = poseDetector.run(Collections.singletonMap(poseDetectorInput, input))) {
            float[][] raw = toDetectionRows(result.get(0).getValue());
            List<Detection> candidates = new ArrayList<>();
            for (float[] row : raw) {
                if (row.length < 56) {
                    continue;
                }
                float score = row[4];
                if (score < STANDARD_THRESHOLD) {
                    continue;
                }
                RectF box = decodeBox(row, letterbox, frameWidth, frameHeight);
                PointF[] keypoints = new PointF[17];
                for (int i = 0; i < keypoints.length; i++) {
                    int offset = 5 + i * 3;
                    float confidence = row[offset + 2];
                    if (confidence < 0.35f) {
                        continue;
                    }
                    float x = (row[offset] - letterbox.padX) / letterbox.scale;
                    float y = (row[offset + 1] - letterbox.padY) / letterbox.scale;
                    keypoints[i] = new PointF(clamp(x, 0, frameWidth), clamp(y, 0, frameHeight));
                }
                candidates.add(new Detection(box, score, false, "person", null, keypoints, null));
            }
            return nmsDetections(candidates);
        }
    }

    private Classification classify(Bitmap frame, RectF box, ClassifierMode mode) throws OrtException {
        Bitmap resized = cropAndResize(frame, box, CLASSIFIER_SIZE);

        OrtSession session = mode == ClassifierMode.HELMET ? helmetClassifier : vestClassifier;
        String inputName = mode == ClassifierMode.HELMET ? helmetInput : vestInput;
        try (OnnxTensor input = bitmapToTensor(resized, CLASSIFIER_SIZE);
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, input))) {
            resized.recycle();
            float[] scores = toClassificationScores(result.get(0).getValue());
            int best = argmax(scores);
            float confidence = confidence(scores, best);
            if (mode == ClassifierMode.HELMET) {
                String[] names = {"Bluehelmet", "Nohelmet", "Otherhelmet", "Redhelmet", "Unclear"};
                return new Classification(names[best], confidence, best == 1);
            }
            String[] names = {"Novest", "Other", "Vest"};
            return new Classification(names[best], confidence, best == 0);
        }
    }

    private Classification classifyGeneric(Bitmap frame, RectF box) throws OrtException {
        Bitmap resized = cropAndResize(frame, box, CLASSIFIER_SIZE);
        try (OnnxTensor input = bitmapToTensor(resized, CLASSIFIER_SIZE);
             OrtSession.Result result = genericClassifier.run(Collections.singletonMap(genericClassifierInput, input))) {
            resized.recycle();
            float[] scores = toClassificationScores(result.get(0).getValue());
            int best = argmax(scores);
            float confidence = confidence(scores, best);
            String label = best >= 0 && best < genericClassNames.length ? genericClassNames[best] : "class_" + best;
            return new Classification(label, confidence, false);
        }
    }

    private static Bitmap cropAndResize(Bitmap frame, RectF box, int size) {
        int left = clamp((int) box.left, 0, frame.getWidth() - 1);
        int top = clamp((int) box.top, 0, frame.getHeight() - 1);
        int right = clamp((int) box.right, left + 1, frame.getWidth());
        int bottom = clamp((int) box.bottom, top + 1, frame.getHeight());
        Bitmap crop = Bitmap.createBitmap(frame, left, top, right - left, bottom - top);
        Bitmap resized = Bitmap.createScaledBitmap(crop, size, size, true);
        crop.recycle();
        return resized;
    }

    private static RectF insetBox(RectF box, float pixels) {
        RectF inset = new RectF(box);
        inset.inset(pixels, pixels);
        if (inset.width() <= 0 || inset.height() <= 0) {
            return new RectF(box);
        }
        return inset;
    }

    private static OnnxTensor bitmapToTensor(Bitmap bitmap, int size) throws OrtException {
        int[] pixels = new int[size * size];
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size);
        FloatBuffer buffer = FloatBuffer.allocate(3 * size * size);
        int plane = size * size;
        for (int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            buffer.put(i, Color.red(color) / 255f);
            buffer.put(plane + i, Color.green(color) / 255f);
            buffer.put(plane * 2 + i, Color.blue(color) / 255f);
        }
        buffer.rewind();
        long[] shape = {1, 3, size, size};
        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, shape);
    }

    private static Letterbox letterbox(Bitmap source, int size) {
        float scale = Math.min(size / (float) source.getWidth(), size / (float) source.getHeight());
        int resizedW = Math.round(source.getWidth() * scale);
        int resizedH = Math.round(source.getHeight() * scale);
        Bitmap resized = Bitmap.createScaledBitmap(source, resizedW, resizedH, true);
        Bitmap output = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        canvas.drawColor(Color.rgb(114, 114, 114));
        float padX = (size - resizedW) * 0.5f;
        float padY = (size - resizedH) * 0.5f;
        canvas.drawBitmap(resized, padX, padY, new Paint(Paint.FILTER_BITMAP_FLAG));
        resized.recycle();
        return new Letterbox(output, scale, padX, padY);
    }

    private static float[][] toDetectionRows(Object value) {
        float[][][] tensor = (float[][][]) value;
        int dim1 = tensor[0].length;
        int dim2 = tensor[0][0].length;
        if (dim1 <= dim2) {
            float[][] rows = new float[dim2][dim1];
            for (int j = 0; j < dim2; j++) {
                for (int i = 0; i < dim1; i++) {
                    rows[j][i] = tensor[0][i][j];
                }
            }
            return rows;
        }
        return tensor[0];
    }

    private static float[] toClassificationScores(Object value) {
        if (value instanceof float[][]) {
            return ((float[][]) value)[0];
        }
        return (float[]) value;
    }

    private static List<PersonBox> nms(List<PersonBox> boxes) {
        boxes.sort((a, b) -> Float.compare(b.score, a.score));
        List<PersonBox> kept = new ArrayList<>();
        boolean[] removed = new boolean[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            if (removed[i]) {
                continue;
            }
            PersonBox current = boxes.get(i);
            kept.add(current);
            for (int j = i + 1; j < boxes.size(); j++) {
                if (!removed[j] && iou(current.box, boxes.get(j).box) > NMS_THRESHOLD) {
                    removed[j] = true;
                }
            }
        }
        return kept;
    }

    private static List<SegmentCandidate> nmsSegments(List<SegmentCandidate> boxes) {
        boxes.sort((a, b) -> Float.compare(b.score, a.score));
        List<SegmentCandidate> kept = new ArrayList<>();
        boolean[] removed = new boolean[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            if (removed[i]) {
                continue;
            }
            SegmentCandidate current = boxes.get(i);
            kept.add(current);
            for (int j = i + 1; j < boxes.size(); j++) {
                if (!removed[j] && iou(current.box, boxes.get(j).box) > NMS_THRESHOLD) {
                    removed[j] = true;
                }
            }
        }
        return kept;
    }

    private static List<Detection> nmsDetections(List<Detection> boxes) {
        boxes.sort((a, b) -> Float.compare(b.score, a.score));
        List<Detection> kept = new ArrayList<>();
        boolean[] removed = new boolean[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            if (removed[i]) {
                continue;
            }
            Detection current = boxes.get(i);
            kept.add(current);
            for (int j = i + 1; j < boxes.size(); j++) {
                if (!removed[j] && iou(current.box, boxes.get(j).box) > NMS_THRESHOLD) {
                    removed[j] = true;
                }
            }
        }
        return kept;
    }

    private static PointF[] decodeObbPolygon(float cx, float cy, float w, float h, float angle,
                                             Letterbox letterbox, int frameWidth, int frameHeight) {
        float cos = (float) Math.cos(angle);
        float sin = (float) Math.sin(angle);
        float hw = w * 0.5f;
        float hh = h * 0.5f;
        float[][] corners = {{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}};
        PointF[] polygon = new PointF[4];
        for (int i = 0; i < corners.length; i++) {
            float x = cx + corners[i][0] * cos - corners[i][1] * sin;
            float y = cy + corners[i][0] * sin + corners[i][1] * cos;
            x = (x - letterbox.padX) / letterbox.scale;
            y = (y - letterbox.padY) / letterbox.scale;
            polygon[i] = new PointF(clamp(x, 0, frameWidth), clamp(y, 0, frameHeight));
        }
        return polygon;
    }

    private static RectF boundsOf(PointF[] points, int frameWidth, int frameHeight) {
        float left = frameWidth;
        float top = frameHeight;
        float right = 0f;
        float bottom = 0f;
        for (PointF point : points) {
            left = Math.min(left, point.x);
            top = Math.min(top, point.y);
            right = Math.max(right, point.x);
            bottom = Math.max(bottom, point.y);
        }
        return new RectF(left, top, right, bottom);
    }

    private static List<RectF> buildMaskCells(RectF box, float[] coeffs, float[][][] proto,
                                              Letterbox letterbox, int frameWidth, int frameHeight) {
        List<RectF> cells = new ArrayList<>();
        int channels = proto.length;
        int maskH = proto[0].length;
        int maskW = proto[0][0].length;
        int step = 4;
        for (int my = 0; my < maskH; my += step) {
            for (int mx = 0; mx < maskW; mx += step) {
                float modelX = (mx + step * 0.5f) * DETECTOR_SIZE / (float) maskW;
                float modelY = (my + step * 0.5f) * DETECTOR_SIZE / (float) maskH;
                float imageX = (modelX - letterbox.padX) / letterbox.scale;
                float imageY = (modelY - letterbox.padY) / letterbox.scale;
                if (!box.contains(imageX, imageY)) {
                    continue;
                }
                float value = 0f;
                for (int c = 0; c < channels && c < coeffs.length; c++) {
                    value += coeffs[c] * proto[c][my][mx];
                }
                if (sigmoid(value) < 0.5f) {
                    continue;
                }
                float cellW = step * DETECTOR_SIZE / (float) maskW / letterbox.scale;
                float cellH = step * DETECTOR_SIZE / (float) maskH / letterbox.scale;
                cells.add(new RectF(
                        clamp(imageX - cellW * 0.5f, 0, frameWidth),
                        clamp(imageY - cellH * 0.5f, 0, frameHeight),
                        clamp(imageX + cellW * 0.5f, 0, frameWidth),
                        clamp(imageY + cellH * 0.5f, 0, frameHeight)
                ));
            }
        }
        return cells;
    }

    private static float iou(RectF a, RectF b) {
        float left = Math.max(a.left, b.left);
        float top = Math.max(a.top, b.top);
        float right = Math.min(a.right, b.right);
        float bottom = Math.min(a.bottom, b.bottom);
        float intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
        float union = a.width() * a.height() + b.width() * b.height() - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    private static float confidence(float[] values, int index) {
        float sum = 0f;
        boolean probabilities = true;
        for (float value : values) {
            sum += value;
            probabilities = probabilities && value >= 0f && value <= 1f;
        }
        if (probabilities && Math.abs(sum - 1f) < 0.05f) {
            return values[index];
        }
        float max = values[argmax(values)];
        double expSum = 0.0;
        for (float value : values) {
            expSum += Math.exp(value - max);
        }
        return (float) (Math.exp(values[index] - max) / expSum);
    }

    private static float sigmoid(float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private OrtSession createSession(Context context, OrtSession.SessionOptions options, String name) throws Exception {
        File model = assetFile(context, name);
        try {
            return env.createSession(model.getAbsolutePath(), options);
        } catch (OrtException e) {
            if (model.exists() && !model.delete()) {
                throw new IllegalStateException("Cannot replace invalid cached model "
                        + model.getAbsolutePath() + ": " + e.getMessage(), e);
            }
            model = assetFile(context, name);
            try {
                return env.createSession(model.getAbsolutePath(), options);
            } catch (OrtException retryError) {
                throw new IllegalStateException("Failed to load " + name
                        + " (" + model.length() + " bytes): " + retryError.getMessage(), retryError);
            }
        }
    }

    private static File assetFile(Context context, String name) throws Exception {
        File file = new File(context.getFilesDir(), name);
        long assetLength = context.getAssets().openFd(name).getLength();
        if (file.exists() && file.length() == assetLength) {
            return file;
        }
        if (file.exists() && !file.delete()) {
            throw new IllegalStateException("Cannot delete stale model file: " + file.getAbsolutePath());
        }
        try (InputStream input = context.getAssets().open(name);
             OutputStream output = new FileOutputStream(file)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            output.flush();
        }
        if (file.length() != assetLength) {
            throw new IllegalStateException("Model copy is incomplete for " + name
                    + ": expected " + assetLength + " bytes, got " + file.length());
        }
        return file;
    }

    private static String[] loadLabels(Context context, String name) throws Exception {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(name)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    labels.add(line.trim());
                }
            }
        }
        return labels.toArray(new String[0]);
    }

    @Override
    public void close() throws Exception {
        detector.close();
        standardDetector.close();
        segDetector.close();
        obbDetector.close();
        poseDetector.close();
        genericClassifier.close();
        if (customDetector != null) {
            customDetector.close();
        }
        helmetClassifier.close();
        vestClassifier.close();
        env.close();
    }

    private static final class PersonBox {
        final RectF box;
        final float score;
        final String label;

        PersonBox(RectF box, float score, String label) {
            this.box = box;
            this.score = score;
            this.label = label;
        }
    }

    private static final class SegmentCandidate {
        final RectF box;
        final float score;
        final String label;
        final float[] coeffs;

        SegmentCandidate(RectF box, float score, String label, float[] coeffs) {
            this.box = box;
            this.score = score;
            this.label = label;
            this.coeffs = coeffs;
        }
    }

    private static final class Letterbox {
        final Bitmap bitmap;
        final float scale;
        final float padX;
        final float padY;

        Letterbox(Bitmap bitmap, float scale, float padX, float padY) {
            this.bitmap = bitmap;
            this.scale = scale;
            this.padX = padX;
            this.padY = padY;
        }
    }

    private static final class Classification {
        final String label;
        final float score;
        final boolean violation;

        Classification(String label, float score, boolean violation) {
            this.label = label;
            this.score = score;
            this.violation = violation;
        }
    }
}
