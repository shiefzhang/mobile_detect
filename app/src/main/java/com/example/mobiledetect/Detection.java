package com.example.mobiledetect;

import android.graphics.RectF;

final class Detection {
    final RectF box;
    final float score;
    final boolean violation;
    final String label;

    Detection(RectF box, float score, boolean violation, String label) {
        this.box = box;
        this.score = score;
        this.violation = violation;
        this.label = label;
    }
}
