package com.example.mobiledetect;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

final class ImageUtils {
    private ImageUtils() {
    }

    static Bitmap imageProxyToBitmap(ImageProxy image) {
        byte[] nv21 = yuv420ToNv21(image);
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, image.getWidth(), image.getHeight()), 85, out);
        Bitmap bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size());
        int rotation = image.getImageInfo().getRotationDegrees();
        if (rotation == 0) {
            return bitmap;
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        bitmap.recycle();
        return rotated;
    }

    private static byte[] yuv420ToNv21(ImageProxy image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * height / 2;
        byte[] nv21 = new byte[ySize + uvSize];

        ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
        ImageProxy.PlaneProxy uPlane = image.getPlanes()[1];
        ImageProxy.PlaneProxy vPlane = image.getPlanes()[2];
        copyPlane(yPlane.getBuffer(), yPlane.getRowStride(), yPlane.getPixelStride(), width, height, nv21, 0, 1);
        copyPlane(vPlane.getBuffer(), vPlane.getRowStride(), vPlane.getPixelStride(), width / 2, height / 2, nv21, ySize, 2);
        copyPlane(uPlane.getBuffer(), uPlane.getRowStride(), uPlane.getPixelStride(), width / 2, height / 2, nv21, ySize + 1, 2);
        return nv21;
    }

    private static void copyPlane(ByteBuffer buffer, int rowStride, int pixelStride, int width, int height,
                                  byte[] output, int offset, int outputPixelStride) {
        int outputOffset = offset;
        buffer.rewind();
        for (int y = 0; y < height; y++) {
            int rowStart = y * rowStride;
            for (int x = 0; x < width; x++) {
                int index = rowStart + x * pixelStride;
                if (index < buffer.limit()) {
                    output[outputOffset] = buffer.get(index);
                }
                outputOffset += outputPixelStride;
            }
        }
    }
}
