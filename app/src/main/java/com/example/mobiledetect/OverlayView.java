package com.example.mobiledetect;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class OverlayView extends View {
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pointPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
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
        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setColor(Color.argb(70, 37, 99, 235));
        pointPaint.setStyle(Paint.Style.FILL);
        pointPaint.setColor(Color.rgb(125, 211, 252));
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
            drawMaskCells(canvas, detection, scale, dx, dy);
            drawPolygon(canvas, detection, scale, dx, dy);
            drawKeypoints(canvas, detection, scale, dx, dy);
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

    private void drawMaskCells(Canvas canvas, Detection detection, float scale, float dx, float dy) {
        if (detection.maskCells.isEmpty()) {
            return;
        }
        fillPaint.setColor(Color.argb(65, 14, 165, 233));
        for (RectF cell : detection.maskCells) {
            RectF mapped = mapRect(cell, scale, dx, dy);
            canvas.drawRect(mapped, fillPaint);
        }
    }

    private void drawPolygon(Canvas canvas, Detection detection, float scale, float dx, float dy) {
        if (detection.polygon == null || detection.polygon.length < 3) {
            return;
        }
        Path path = new Path();
        PointF first = mapPoint(detection.polygon[0], scale, dx, dy);
        path.moveTo(first.x, first.y);
        for (int i = 1; i < detection.polygon.length; i++) {
            PointF p = mapPoint(detection.polygon[i], scale, dx, dy);
            path.lineTo(p.x, p.y);
        }
        path.close();
        fillPaint.setColor(Color.argb(45, 37, 99, 235));
        canvas.drawPath(path, fillPaint);
        canvas.drawPath(path, boxPaint);
    }

    private void drawKeypoints(Canvas canvas, Detection detection, float scale, float dx, float dy) {
        if (detection.keypoints == null || detection.keypoints.length == 0) {
            return;
        }
        int[][] skeleton = {
                {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 6},
                {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15},
                {12, 14}, {14, 16}, {0, 1}, {0, 2}, {1, 3}, {2, 4}
        };
        boxPaint.setColor(Color.rgb(14, 165, 233));
        for (int[] edge : skeleton) {
            if (edge[0] >= detection.keypoints.length || edge[1] >= detection.keypoints.length) {
                continue;
            }
            PointF a = detection.keypoints[edge[0]];
            PointF b = detection.keypoints[edge[1]];
            if (a == null || b == null) {
                continue;
            }
            PointF ma = mapPoint(a, scale, dx, dy);
            PointF mb = mapPoint(b, scale, dx, dy);
            canvas.drawLine(ma.x, ma.y, mb.x, mb.y, boxPaint);
        }
        pointPaint.setColor(Color.rgb(191, 219, 254));
        for (PointF keypoint : detection.keypoints) {
            if (keypoint == null) {
                continue;
            }
            PointF mapped = mapPoint(keypoint, scale, dx, dy);
            canvas.drawCircle(mapped.x, mapped.y, 6f, pointPaint);
        }
    }

    private RectF mapRect(RectF source, float scale, float dx, float dy) {
        float left = source.left * scale + dx;
        float right = source.right * scale + dx;
        if (mirrorHorizontally) {
            float mirroredLeft = getWidth() - right;
            right = getWidth() - left;
            left = mirroredLeft;
        }
        return new RectF(left, source.top * scale + dy, right, source.bottom * scale + dy);
    }

    private PointF mapPoint(PointF source, float scale, float dx, float dy) {
        float x = source.x * scale + dx;
        if (mirrorHorizontally) {
            x = getWidth() - x;
        }
        return new PointF(x, source.y * scale + dy);
    }
}
