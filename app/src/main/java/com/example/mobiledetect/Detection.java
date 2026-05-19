package com.example.mobiledetect;

import android.graphics.PointF;
import android.graphics.Color;
import android.graphics.RectF;

import java.util.Collections;
import java.util.List;

final class Detection {
    final RectF box;
    final float score;
    final boolean violation;
    final String label;
    final int color;
    final PointF[] polygon;
    final PointF[] keypoints;
    final List<RectF> maskCells;

    Detection(RectF box, float score, boolean violation, String label) {
        this(box, score, violation, label,
                violation ? Color.rgb(220, 38, 38) : Color.rgb(37, 99, 235),
                null, null, Collections.emptyList());
    }

    Detection(RectF box, float score, boolean violation, String label, int color) {
        this(box, score, violation, label, color, null, null, Collections.emptyList());
    }

    Detection(RectF box, float score, boolean violation, String label,
              PointF[] polygon, PointF[] keypoints, List<RectF> maskCells) {
        this(box, score, violation, label,
                violation ? Color.rgb(220, 38, 38) : Color.rgb(37, 99, 235),
                polygon, keypoints, maskCells);
    }

    Detection(RectF box, float score, boolean violation, String label, int color,
              PointF[] polygon, PointF[] keypoints, List<RectF> maskCells) {
        this.box = box;
        this.score = score;
        this.violation = violation;
        this.label = label;
        this.color = color;
        this.polygon = polygon;
        this.keypoints = keypoints;
        this.maskCells = maskCells == null ? Collections.emptyList() : maskCells;
    }
}
