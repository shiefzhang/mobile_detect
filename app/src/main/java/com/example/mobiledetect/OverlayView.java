package com.example.mobiledetect;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class OverlayView extends View {
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private List<Detection> detections = new ArrayList<>();
    private int sourceWidth = 1;
    private int sourceHeight = 1;
    private boolean mirrorHorizontally = false;

    public OverlayView(Context context) {
        super(context);
        init();
    }

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5f);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(34f);
        textPaint.setStyle(Paint.Style.FILL);
    }

    public void setDetections(List<Detection> detections, int sourceWidth, int sourceHeight) {
        this.detections = new ArrayList<>(detections);
        this.sourceWidth = Math.max(1, sourceWidth);
        this.sourceHeight = Math.max(1, sourceHeight);
        invalidate();
    }

    public void setMirrorHorizontally(boolean mirrorHorizontally) {
        this.mirrorHorizontally = mirrorHorizontally;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float scale = Math.max(getWidth() / (float) sourceWidth, getHeight() / (float) sourceHeight);
        float dx = (getWidth() - sourceWidth * scale) * 0.5f;
        float dy = (getHeight() - sourceHeight * scale) * 0.5f;

        for (Detection detection : detections) {
            boxPaint.setColor(detection.violation ? Color.rgb(220, 38, 38) : Color.rgb(37, 99, 235));
            float left = detection.box.left * scale + dx;
            float right = detection.box.right * scale + dx;
            if (mirrorHorizontally) {
                float mirroredLeft = getWidth() - right;
                right = getWidth() - left;
                left = mirroredLeft;
            }
            RectF r = new RectF(left, detection.box.top * scale + dy, right, detection.box.bottom * scale + dy);
            canvas.drawRect(r, boxPaint);
            String text = String.format(Locale.US, "%s %.0f%%", detection.label, detection.score * 100f);
            float textY = Math.max(36f, r.top - 10f);
            canvas.drawText(text, r.left + 4f, textY, textPaint);
        }
    }
}
