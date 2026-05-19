package com.example.mobiledetect;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.util.Size;
import android.view.Gravity;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.ZoomState;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends ComponentActivity {
    private static final int CAMERA_PERMISSION = 10;

    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final ActivityResultLauncher<String[]> modelPicker = registerForActivityResult(
            new ActivityResultContracts.OpenDocument(),
            this::onCustomModelPicked
    );

    private PreviewView previewView;
    private OverlayView overlayView;
    private FrameLayout settingsDrawer;
    private LinearLayout drawerContent;
    private TextView drawerToggle;
    private TextView statusView;
    private TextView zoomView;
    private TextView customModelButton;
    private Spinner detectorSpinner;
    private Spinner safetyModeSpinner;
    private Spinner classSpinner;
    private ArrayAdapter<String> classAdapter;
    private ScaleGestureDetector scaleGestureDetector;
    private volatile ModelRunner modelRunner;
    private volatile ModelRunner.DetectorMode detectorMode = ModelRunner.DetectorMode.SAFETY;
    private volatile ModelRunner.ClassifierMode classifierMode = ModelRunner.ClassifierMode.HELMET;
    private volatile int selectedClassId = 0;
    private volatile boolean customModelLoaded = false;
    private int lensFacing = CameraSelector.LENS_FACING_BACK;
    private Camera currentCamera;
    private float currentZoomRatio = 1f;
    private int drawerWidth;
    private int drawerHandleWidth;
    private boolean drawerOpen = false;
    private long lastFrameTime;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        loadModels();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
        }
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.rgb(5, 16, 34));

        previewView = new PreviewView(this);
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        overlayView = new OverlayView(this);

        scaleGestureDetector = new ScaleGestureDetector(this, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override
            public boolean onScale(@NonNull ScaleGestureDetector detector) {
                applyZoom(detector.getScaleFactor());
                return true;
            }
        });
        previewView.setOnTouchListener((view, event) -> {
            scaleGestureDetector.onTouchEvent(event);
            return true;
        });

        root.addView(previewView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        root.addView(overlayView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        buildSettingsDrawer();
        root.addView(settingsDrawer, drawerParams());

        zoomView = new TextView(this);
        zoomView.setTextColor(Color.WHITE);
        zoomView.setTextSize(15f);
        zoomView.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        zoomView.setGravity(Gravity.CENTER);
        zoomView.setPadding(dp(18), dp(10), dp(18), dp(10));
        zoomView.setBackground(zoomBackground());
        updateZoomText(1f);

        FrameLayout.LayoutParams zoomParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        zoomParams.gravity = Gravity.BOTTOM | Gravity.END;
        zoomParams.setMargins(0, 0, dp(22), dp(28));
        root.addView(zoomView, zoomParams);

        setContentView(root);
        settingsDrawer.post(() -> setDrawerOpen(false, false));
        updateModeControls();
    }

    private void buildSettingsDrawer() {
        drawerWidth = dp(322);
        drawerHandleWidth = dp(52);
        settingsDrawer = new FrameLayout(this);

        drawerContent = new LinearLayout(this);
        drawerContent.setOrientation(LinearLayout.VERTICAL);
        drawerContent.setPadding(dp(22), dp(24), dp(70), dp(24));
        drawerContent.setBackground(drawerBackground());

        TextView title = new TextView(this);
        title.setText("智能安全检测");
        title.setTextColor(Color.WHITE);
        title.setTextSize(20f);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);

        TextView subtitle = new TextView(this);
        subtitle.setText("Detect / Seg / OBB / Pose");
        subtitle.setTextColor(0xFFBFDBFE);
        subtitle.setTextSize(12f);
        subtitle.setPadding(0, dp(2), 0, dp(12));

        Spinner cameraSpinner = createSpinner(new String[]{"后置主摄像头", "前置摄像头"});
        detectorSpinner = createSpinner(new String[]{
                "人员安全模型", "标准检测模型", "实例分割模型", "旋转框 OBB 模型", "姿态 Pose 模型", "自定义检测模型"
        });
        safetyModeSpinner = createSpinner(new String[]{"安全帽检测", "反光衣检测"});
        classAdapter = createAdapter(ModelRunner.COCO_NAMES);
        classSpinner = new Spinner(this);
        classSpinner.setAdapter(classAdapter);
        classSpinner.setPadding(dp(12), dp(4), dp(12), dp(4));
        classSpinner.setBackground(spinnerBackground());
        customModelButton = actionButton("选择 ONNX 检测模型");
        customModelButton.setOnClickListener(view -> modelPicker.launch(new String[]{"*/*"}));

        cameraSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                int selectedLens = position == 0 ? CameraSelector.LENS_FACING_BACK : CameraSelector.LENS_FACING_FRONT;
                if (lensFacing != selectedLens) {
                    lensFacing = selectedLens;
                    clearOverlay();
                    if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                            == PackageManager.PERMISSION_GRANTED) {
                        startCamera();
                    }
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        detectorSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                ModelRunner.DetectorMode nextMode = modeForPosition(position);
                if (nextMode == ModelRunner.DetectorMode.CUSTOM && !customModelLoaded) {
                    statusView.setText("请先选择 ONNX 检测模型");
                    detectorSpinner.setSelection(positionForMode(detectorMode));
                    return;
                }
                detectorMode = nextMode;
                selectedClassId = 0;
                updateClassAdapter();
                updateModeControls();
                clearOverlay();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        safetyModeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                classifierMode = position == 0 ? ModelRunner.ClassifierMode.HELMET : ModelRunner.ClassifierMode.VEST;
                clearOverlay();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        classSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                selectedClassId = position;
                clearOverlay();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        drawerContent.addView(title, matchWrapParams());
        drawerContent.addView(subtitle, matchWrapParams());
        drawerContent.addView(label("摄像头"));
        drawerContent.addView(cameraSpinner, matchWrapParams());
        drawerContent.addView(label("检测模型"));
        drawerContent.addView(detectorSpinner, matchWrapParams());
        drawerContent.addView(customModelButton, matchWrapParams());
        drawerContent.addView(label("安全检测类型"));
        drawerContent.addView(safetyModeSpinner, matchWrapParams());
        drawerContent.addView(label("检测类别"));
        drawerContent.addView(classSpinner, matchWrapParams());

        statusView = new TextView(this);
        statusView.setTextColor(0xFFDDEBFF);
        statusView.setTextSize(14f);
        statusView.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        statusView.setText("正在加载模型...");
        statusView.setPadding(dp(18), dp(14), dp(18), dp(14));
        statusView.setBackground(statusBackground());
        drawerContent.addView(statusView, matchWrapParams());

        settingsDrawer.addView(drawerContent, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        drawerToggle = new TextView(this);
        drawerToggle.setText("设置");
        drawerToggle.setTextColor(Color.WHITE);
        drawerToggle.setTextSize(15f);
        drawerToggle.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        drawerToggle.setGravity(Gravity.CENTER);
        drawerToggle.setBackground(drawerHandleBackground());
        drawerToggle.setOnClickListener(view -> setDrawerOpen(!drawerOpen, true));

        FrameLayout.LayoutParams handleParams = new FrameLayout.LayoutParams(drawerHandleWidth, dp(118));
        handleParams.gravity = Gravity.END | Gravity.TOP;
        handleParams.setMargins(0, dp(28), 0, 0);
        settingsDrawer.addView(drawerToggle, handleParams);
    }

    private ModelRunner.DetectorMode modeForPosition(int position) {
        switch (position) {
            case 1:
                return ModelRunner.DetectorMode.STANDARD;
            case 2:
                return ModelRunner.DetectorMode.SEGMENT;
            case 3:
                return ModelRunner.DetectorMode.OBB;
            case 4:
                return ModelRunner.DetectorMode.POSE;
            case 5:
                return ModelRunner.DetectorMode.CUSTOM;
            default:
                return ModelRunner.DetectorMode.SAFETY;
        }
    }

    private int positionForMode(ModelRunner.DetectorMode mode) {
        switch (mode) {
            case STANDARD:
                return 1;
            case SEGMENT:
                return 2;
            case OBB:
                return 3;
            case POSE:
                return 4;
            case CUSTOM:
                return 5;
            default:
                return 0;
        }
    }

    private void updateClassAdapter() {
        String[] names = detectorMode == ModelRunner.DetectorMode.OBB ? ModelRunner.OBB_NAMES : ModelRunner.COCO_NAMES;
        classAdapter.clear();
        classAdapter.addAll(names);
        classAdapter.notifyDataSetChanged();
        classSpinner.setSelection(0);
    }

    private FrameLayout.LayoutParams drawerParams() {
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(drawerWidth, ViewGroup.LayoutParams.MATCH_PARENT);
        params.gravity = Gravity.START | Gravity.TOP;
        return params;
    }

    private void setDrawerOpen(boolean open, boolean animate) {
        drawerOpen = open;
        float target = open ? 0f : -drawerWidth + drawerHandleWidth;
        drawerToggle.setText(open ? "收起" : "设置");
        if (open) {
            drawerContent.setVisibility(View.VISIBLE);
            drawerContent.setAlpha(1f);
        }
        if (animate) {
            settingsDrawer.animate()
                    .translationX(target)
                    .setDuration(220)
                    .withEndAction(() -> {
                        if (!drawerOpen) {
                            drawerContent.setVisibility(View.INVISIBLE);
                        }
                    })
                    .start();
        } else {
            settingsDrawer.setTranslationX(target);
            drawerContent.setVisibility(open ? View.VISIBLE : View.INVISIBLE);
        }
    }

    private Spinner createSpinner(String[] items) {
        Spinner spinner = new Spinner(this);
        spinner.setAdapter(createAdapter(items));
        spinner.setPadding(dp(12), dp(4), dp(12), dp(4));
        spinner.setBackground(spinnerBackground());
        return spinner;
    }

    private ArrayAdapter<String> createAdapter(String[] items) {
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, items);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        return adapter;
    }

    private TextView label(String text) {
        TextView label = new TextView(this);
        label.setText(text);
        label.setTextColor(0xFF93C5FD);
        label.setTextSize(12f);
        label.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        label.setPadding(dp(2), dp(10), 0, dp(4));
        return label;
    }

    private TextView actionButton(String text) {
        TextView button = new TextView(this);
        button.setText(text);
        button.setTextColor(Color.WHITE);
        button.setTextSize(14f);
        button.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        button.setGravity(Gravity.CENTER);
        button.setPadding(dp(14), dp(10), dp(14), dp(10));
        button.setBackground(actionButtonBackground());
        return button;
    }

    private LinearLayout.LayoutParams matchWrapParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.setMargins(0, 0, 0, dp(6));
        return params;
    }

    private GradientDrawable drawerBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                new int[]{0xF20F2A55, 0xF2071B38});
        drawable.setCornerRadii(new float[]{0, 0, dp(28), dp(28), dp(28), dp(28), 0, 0});
        drawable.setStroke(dp(1), 0x6659A8FF);
        return drawable;
    }

    private GradientDrawable drawerHandleBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                new int[]{0xFF2563EB, 0xFF0EA5E9});
        drawable.setCornerRadii(new float[]{0, 0, dp(18), dp(18), dp(18), dp(18), 0, 0});
        drawable.setStroke(dp(1), 0xAAE0F2FE);
        return drawable;
    }

    private GradientDrawable spinnerBackground() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(0xEEEBF4FF);
        drawable.setCornerRadius(dp(18));
        drawable.setStroke(dp(2), 0xFF60A5FA);
        return drawable;
    }

    private GradientDrawable actionButtonBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                new int[]{0xFF2563EB, 0xFF0EA5E9});
        drawable.setCornerRadius(dp(18));
        drawable.setStroke(dp(1), 0xAAE0F2FE);
        return drawable;
    }

    private GradientDrawable statusBackground() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(0x772563EB);
        drawable.setCornerRadius(dp(18));
        drawable.setStroke(dp(1), 0x8893C5FD);
        return drawable;
    }

    private GradientDrawable zoomBackground() {
        GradientDrawable drawable = new GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                new int[]{0xDD1D4ED8, 0xDD0EA5E9});
        drawable.setCornerRadius(dp(28));
        drawable.setStroke(dp(1), 0xAAE0F2FE);
        return drawable;
    }

    private void updateModeControls() {
        boolean safetyMode = detectorMode == ModelRunner.DetectorMode.SAFETY;
        boolean poseMode = detectorMode == ModelRunner.DetectorMode.POSE;
        safetyModeSpinner.setEnabled(safetyMode);
        safetyModeSpinner.setAlpha(safetyMode ? 1f : 0.45f);
        classSpinner.setEnabled(!safetyMode && !poseMode);
        classSpinner.setAlpha(!safetyMode && !poseMode ? 1f : 0.45f);
        customModelButton.setEnabled(modelRunner != null);
        customModelButton.setAlpha(modelRunner == null ? 0.45f : 1f);
    }

    private void clearOverlay() {
        overlayView.setDetections(new ArrayList<>(), 1, 1);
    }

    private void loadModels() {
        inferenceExecutor.execute(() -> {
            try {
                modelRunner = new ModelRunner(getApplicationContext());
                runOnUiThread(() -> {
                    statusView.setText("模型已加载");
                    updateModeControls();
                });
            } catch (Exception e) {
                runOnUiThread(() -> statusView.setText("模型加载失败: " + e.getMessage()));
            }
        });
    }

    private void onCustomModelPicked(Uri uri) {
        if (uri == null) {
            return;
        }
        ModelRunner runner = modelRunner;
        if (runner == null) {
            statusView.setText("基础模型尚未加载完成");
            return;
        }
        statusView.setText("正在加载自定义模型...");
        inferenceExecutor.execute(() -> {
            try {
                File modelFile = copyUriToInternalFile(uri, "custom_detector.onnx");
                runner.loadCustomDetector(modelFile);
                customModelLoaded = true;
                detectorMode = ModelRunner.DetectorMode.CUSTOM;
                runOnUiThread(() -> {
                    detectorSpinner.setSelection(positionForMode(ModelRunner.DetectorMode.CUSTOM));
                    statusView.setText("自定义模型已加载");
                    clearOverlay();
                    updateModeControls();
                });
            } catch (Exception e) {
                runOnUiThread(() -> statusView.setText("自定义模型加载失败: " + e.getMessage()));
            }
        });
    }

    private File copyUriToInternalFile(Uri uri, String fileName) throws Exception {
        File outputFile = new File(getFilesDir(), fileName);
        File tempFile = new File(getFilesDir(), fileName + ".tmp");
        try (InputStream input = getContentResolver().openInputStream(uri);
             OutputStream output = new FileOutputStream(tempFile)) {
            if (input == null) {
                throw new IllegalStateException("无法打开选择的文件");
            }
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            output.flush();
        }
        if (tempFile.length() <= 0) {
            throw new IllegalStateException("选择的模型文件为空");
        }
        if (outputFile.exists() && !outputFile.delete()) {
            throw new IllegalStateException("无法替换旧的自定义模型");
        }
        if (!tempFile.renameTo(outputFile)) {
            throw new IllegalStateException("无法保存自定义模型");
        }
        return outputFile;
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = future.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ResolutionSelector resolutionSelector = new ResolutionSelector.Builder()
                        .setResolutionStrategy(new ResolutionStrategy(
                                new Size(640, 480),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                        .build();
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(resolutionSelector)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                analysis.setAnalyzer(inferenceExecutor, this::analyzeFrame);

                CameraSelector selector = new CameraSelector.Builder()
                        .requireLensFacing(lensFacing)
                        .build();
                if (!cameraProvider.hasCamera(selector)) {
                    statusView.setText("当前设备没有该摄像头");
                    return;
                }

                cameraProvider.unbindAll();
                currentCamera = cameraProvider.bindToLifecycle(this, selector, preview, analysis);
                overlayView.setMirrorHorizontally(lensFacing == CameraSelector.LENS_FACING_FRONT);
                resetZoom();
            } catch (Exception e) {
                statusView.setText("相机启动失败: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void applyZoom(float scaleFactor) {
        Camera camera = currentCamera;
        if (camera == null) {
            return;
        }
        ZoomState zoomState = camera.getCameraInfo().getZoomState().getValue();
        if (zoomState == null) {
            return;
        }
        float target = currentZoomRatio * scaleFactor;
        target = clamp(target, zoomState.getMinZoomRatio(), zoomState.getMaxZoomRatio());
        currentZoomRatio = target;
        camera.getCameraControl().setZoomRatio(target);
        updateZoomText(target);
    }

    private void resetZoom() {
        Camera camera = currentCamera;
        currentZoomRatio = 1f;
        if (camera != null) {
            ZoomState zoomState = camera.getCameraInfo().getZoomState().getValue();
            if (zoomState != null) {
                currentZoomRatio = clamp(1f, zoomState.getMinZoomRatio(), zoomState.getMaxZoomRatio());
            }
            camera.getCameraControl().setZoomRatio(currentZoomRatio);
        }
        updateZoomText(currentZoomRatio);
    }

    private void updateZoomText(float zoomRatio) {
        if (zoomView != null) {
            zoomView.setText(String.format(Locale.US, "%.1fx", zoomRatio));
        }
    }

    private void analyzeFrame(ImageProxy image) {
        ModelRunner runner = modelRunner;
        long now = System.currentTimeMillis();
        if (runner == null || now - lastFrameTime < 180 || !running.compareAndSet(false, true)) {
            image.close();
            return;
        }
        lastFrameTime = now;
        Bitmap bitmap = null;
        try {
            bitmap = ImageUtils.imageProxyToBitmap(image);
            ModelRunner.DetectorMode selectedMode = detectorMode;
            int classId = selectedClassId;
            List<Detection> detections;
            switch (selectedMode) {
                case SAFETY:
                    detections = runner.runSafety(bitmap, classifierMode);
                    break;
                case SEGMENT:
                    detections = runner.runSegment(bitmap, classId);
                    break;
                case OBB:
                    detections = runner.runObb(bitmap, classId);
                    break;
                case POSE:
                    detections = runner.runPose(bitmap);
                    break;
                case CUSTOM:
                    detections = runner.runCustom(bitmap, classId);
                    break;
                case STANDARD:
                default:
                    detections = runner.runStandard(bitmap, classId);
                    break;
            }
            int width = bitmap.getWidth();
            int height = bitmap.getHeight();
            runOnUiThread(() -> {
                overlayView.setDetections(detections, width, height);
                statusView.setText(statusText(selectedMode, classId, detections.size()));
            });
        } catch (Exception e) {
            runOnUiThread(() -> statusView.setText("推理失败: " + e.getMessage()));
        } finally {
            if (bitmap != null) {
                bitmap.recycle();
            }
            image.close();
            running.set(false);
        }
    }

    private String statusText(ModelRunner.DetectorMode mode, int classId, int count) {
        if (mode == ModelRunner.DetectorMode.SAFETY) {
            return "人员: " + count;
        }
        if (mode == ModelRunner.DetectorMode.POSE) {
            return "姿态 person: " + count;
        }
        if (mode == ModelRunner.DetectorMode.OBB) {
            return "OBB " + ModelRunner.OBB_NAMES[Math.min(classId, ModelRunner.OBB_NAMES.length - 1)] + ": " + count;
        }
        String prefix = mode == ModelRunner.DetectorMode.SEGMENT ? "分割 "
                : mode == ModelRunner.DetectorMode.CUSTOM ? "自定义 " : "";
        return prefix + ModelRunner.COCO_NAMES[Math.min(classId, ModelRunner.COCO_NAMES.length - 1)] + ": " + count;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            statusView.setText("需要摄像头权限");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        inferenceExecutor.execute(() -> {
            try {
                if (modelRunner != null) {
                    modelRunner.close();
                }
            } catch (Exception ignored) {
            }
        });
        inferenceExecutor.shutdown();
    }
}
