package com.tridi.audience;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceHolder;
import java.util.Arrays;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

final class CameraController implements SurfaceHolder.Callback {
    // Physical reference selected from test photo 3. The EMEET is mounted on
    // the left side at the vertical centre of the advertisement. Its raw
    // landscape frame has the viewer's head on the left, so AI and evidence
    // are normalized 90 degrees clockwise before any inference is used.
    static final int CAMERA_CLOCKWISE_ROTATION = 90;

    interface Listener {
        void onVision(float[] result, byte[] eventJpeg, long capturedAtMs);
        void onPreview(int[] argb, int width, int height, long capturedAtMs);
        void onCameraStatus(String status, boolean error);
    }

    private final Context context;
    private final SurfaceHolder holder;
    private final long engine;
    private final Listener listener;
    private final AtomicBoolean processing = new AtomicBoolean(false);
    private HandlerThread cameraThread, visionThread;
    private Handler cameraHandler, visionHandler;
    private CameraDevice device;
    private CameraCaptureSession session;
    private ImageReader reader;
    private volatile boolean running;
    private volatile long lastFrameMs;
    private volatile long lastPreviewMs;
    private volatile boolean previewOutputEnabled;
    private int reopenCount;
    private int profileIndex;
    // Keep the exact stream combinations proven on the BTV B11 Camera2 LEGACY
    // provider. The EMEET accepts mixed 1080p sessions but returns torn
    // green/magenta buffers instead of reporting configuration failure.
    private static final int[][] PROFILES = {
            {1280,720,1280,720},
            {1280,720,640,360},
            {640,480,640,360}
    };

    CameraController(Context context, SurfaceHolder holder, long engine, Listener listener) {
        this.context = context; this.holder = holder; this.engine = engine; this.listener = listener;
    }

    void start() {
        if (running) return;
        running = true;
        cameraThread = new HandlerThread("Tridi-Camera");
        visionThread = new HandlerThread("Tridi-NCNN");
        cameraThread.start(); visionThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
        visionHandler = new Handler(visionThread.getLooper());
        holder.addCallback(this);
        if (holder.getSurface().isValid()) cameraHandler.post(this::openCamera);
        cameraHandler.postDelayed(watchdog, 2500L);
    }

    void setPreviewOutputEnabled(boolean enabled) {
        previewOutputEnabled = enabled;
    }

    void stop() {
        running = false;
        holder.removeCallback(this);
        if (cameraHandler != null) cameraHandler.removeCallbacksAndMessages(null);
        if (visionHandler != null) visionHandler.removeCallbacksAndMessages(null);
        closeCamera();
        if (cameraThread != null) cameraThread.quitSafely();
        if (visionThread != null) visionThread.quitSafely();
        cameraThread = visionThread = null; cameraHandler = visionHandler = null;
    }

    @Override public void surfaceCreated(SurfaceHolder h) { if (running && cameraHandler != null) cameraHandler.post(this::openCamera); }
    @Override public void surfaceChanged(SurfaceHolder h, int format, int width, int height) {}
    @Override public void surfaceDestroyed(SurfaceHolder h) { if (cameraHandler != null) cameraHandler.post(this::closeCamera); }

    private final Runnable watchdog = new Runnable() {
        @Override public void run() {
            if (!running || cameraHandler == null) return;
            long silent = SystemClock.elapsedRealtime() - lastFrameMs;
            if (device != null && lastFrameMs > 0 && silent > 5000L) {
                listener.onCameraStatus("Reconectando câmera USB…", true);
                profileIndex = (profileIndex + 1) % PROFILES.length;
                closeCamera();
                cameraHandler.postDelayed(CameraController.this::openCamera, 900L);
            } else if (device == null && holder.getSurface().isValid()) {
                openCamera();
            }
            cameraHandler.postDelayed(this, 2500L);
        }
    };

    @SuppressLint("MissingPermission")
    private void openCamera() {
        if (!running || device != null || !holder.getSurface().isValid()) return;
        try {
            CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
            String selected = selectExternalCamera(manager);
            if (selected == null) throw new CameraAccessException(CameraAccessException.CAMERA_ERROR, "Nenhuma câmera");
            listener.onCameraStatus("Abrindo câmera USB…", false);
            manager.openCamera(selected, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice camera) {
                    if (!running) { camera.close(); return; }
                    device = camera; reopenCount = 0;
                    configureSession(camera);
                }
                @Override public void onDisconnected(CameraDevice camera) { camera.close(); device = null; scheduleReopen("Câmera USB desconectada"); }
                @Override public void onError(CameraDevice camera, int error) { camera.close(); device = null; scheduleReopen("Erro da câmera " + error); }
            }, cameraHandler);
        } catch (Exception e) { scheduleReopen("Aguardando câmera USB"); }
    }

    private String selectExternalCamera(CameraManager manager) throws CameraAccessException {
        String fallback = null;
        for (String id : manager.getCameraIdList()) {
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
            Integer facing = c.get(CameraCharacteristics.LENS_FACING);
            StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map == null || map.getOutputSizes(ImageFormat.YUV_420_888) == null) continue;
            if (fallback == null) fallback = id;
            if (facing != null && facing == CameraCharacteristics.LENS_FACING_EXTERNAL) return id;
        }
        return fallback;
    }

    private void configureSession(CameraDevice camera) {
        try {
            CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
            CameraCharacteristics c = manager.getCameraCharacteristics(camera.getId());
            StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            int[] profile = PROFILES[profileIndex % PROFILES.length];
            Size previewSize = choose(map.getOutputSizes(SurfaceHolder.class), profile[0], profile[1]);
            Size analysisSize = choose(map.getOutputSizes(ImageFormat.YUV_420_888), profile[2], profile[3]);
            holder.setFixedSize(previewSize.getWidth(), previewSize.getHeight());
            reader = ImageReader.newInstance(analysisSize.getWidth(), analysisSize.getHeight(), ImageFormat.YUV_420_888, 3);
            reader.setOnImageAvailableListener(this::onImage, cameraHandler);
            List<Surface> outputs = Arrays.asList(holder.getSurface(), reader.getSurface());
            camera.createCaptureSession(outputs, new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession s) {
                    if (!running || device == null) { s.close(); return; }
                    session = s;
                    try {
                        CaptureRequest.Builder b = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        b.addTarget(holder.getSurface()); b.addTarget(reader.getSurface());
                        b.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                        int[] afModes = c.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES);
                        int af = contains(afModes, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
                                ? CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
                                : (contains(afModes, CaptureRequest.CONTROL_AF_MODE_AUTO)
                                ? CaptureRequest.CONTROL_AF_MODE_AUTO : CaptureRequest.CONTROL_AF_MODE_OFF);
                        b.set(CaptureRequest.CONTROL_AF_MODE, af);
                        b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON);
                        s.setRepeatingRequest(b.build(), null, cameraHandler);
                        lastFrameMs = SystemClock.elapsedRealtime();
                        listener.onCameraStatus("Câmera ativa • IA " + analysisSize.getWidth() + "×" + analysisSize.getHeight()
                                + " • preview " + previewSize.getWidth() + "×" + previewSize.getHeight(), false);
                    } catch (Exception e) {
                        profileIndex = (profileIndex + 1) % PROFILES.length;
                        scheduleReopen("Falha ao iniciar vídeo");
                    }
                }
                @Override public void onConfigureFailed(CameraCaptureSession s) {
                    profileIndex = (profileIndex + 1) % PROFILES.length;
                    scheduleReopen("Reconfigurando perfil da câmera");
                }
            }, cameraHandler);
        } catch (Exception e) {
            profileIndex = (profileIndex + 1) % PROFILES.length;
            scheduleReopen("Formato da webcam incompatível");
        }
    }

    private static Size choose(Size[] sizes, int wantedW, int wantedH) {
        if (sizes == null || sizes.length == 0) return new Size(wantedW, wantedH);
        return Collections.min(Arrays.asList(sizes), Comparator.comparingLong(s ->
                Math.abs((long) s.getWidth() - wantedW) * 4L + Math.abs((long) s.getHeight() - wantedH) * 4L
                        + Math.abs((long) s.getWidth() * wantedH - (long) wantedW * s.getHeight())));
    }

    private static boolean contains(int[] values, int wanted) {
        if (values == null) return false;
        for (int value : values) if (value == wanted) return true;
        return false;
    }

    private void onImage(ImageReader source) {
        Image image = null;
        try {
            image = source.acquireLatestImage();
            if (image == null || !running) return;
            lastFrameMs = SystemClock.elapsedRealtime();

            // CONFIG preview must never wait for NCNN/GazeNet. Camera callbacks
            // run on the camera thread; the expensive vision frame is handed
            // to the dedicated vision thread below. A smaller 240px upright
            // copy at ~11 FPS costs roughly the same pixel budget as the old
            // 400px / 3 FPS path, but looks dramatically smoother.
            long now = SystemClock.elapsedRealtime();
            if (previewOutputEnabled && now - lastPreviewMs >= 90L) {
                PreviewFrame preview = buildUprightPreview(image, CAMERA_CLOCKWISE_ROTATION, 240);
                if (preview != null) {
                    lastPreviewMs = now;
                    listener.onPreview(preview.argb, preview.width, preview.height,
                            System.currentTimeMillis());
                }
            }

            // Keep at most one native inference in flight. Crucially, do not
            // block the ImageReader callback while NCNN is running: newer
            // camera frames can still feed the preview and old analysis frames
            // are dropped by acquireLatestImage().
            if (!processing.compareAndSet(false, true) || visionHandler == null) return;
            final Image visionImage = image;
            image = null; // ownership moves to the vision thread
            visionHandler.post(() -> processVisionImage(visionImage));
        } catch (Throwable t) {
            listener.onCameraStatus("Recuperando quadro da câmera…", true);
        } finally {
            if (image != null) image.close();
        }
    }

    private void processVisionImage(Image image) {
        try {
            if (!running || image == null) return;
            Image.Plane[] p = image.getPlanes();
            float[] result = NativeBridge.processYuv420(engine,
                    p[0].getBuffer(), p[0].getBuffer().position(), p[0].getRowStride(),
                    p[1].getBuffer(), p[1].getBuffer().position(), p[1].getRowStride(), p[1].getPixelStride(),
                    p[2].getBuffer(), p[2].getBuffer().position(), p[2].getRowStride(), p[2].getPixelStride(),
                    image.getWidth(), image.getHeight(), CAMERA_CLOCKWISE_ROTATION, System.nanoTime());
            byte[] eventJpeg = needsEvidenceJpeg(result)
                    ? encodeRotatedJpeg(image, 92, CAMERA_CLOCKWISE_ROTATION) : null;
            listener.onVision(result, eventJpeg, System.currentTimeMillis());
        } catch (Throwable t) {
            listener.onCameraStatus("Recuperando análise da câmera…", true);
        } finally {
            try { if (image != null) image.close(); } catch (Throwable ignored) {}
            processing.set(false);
        }
    }

    private static boolean needsEvidenceJpeg(float[] result) {
        if (result == null || result.length < VisionResult.HEADER) return false;
        int count = Math.max(0, Math.round(result[4]));
        for (int i = 0; i < count; i++) {
            int p = VisionResult.offset(i);
            if (p + VisionResult.ATTENTION_STREAK >= result.length) break;
            int flags = Math.round(result[p + VisionResult.EVENT_FLAGS]);
            int attentionEvaluation = Math.round(result[p + VisionResult.ATTENTION_EVALUATION]);
            // Only the winning pseudo-impression frame (or another real event)
            // requests a JPEG. Rejected frames remain numeric observations;
            // compressing the whole temporary list would stall the ARMv7 box.
            if (attentionEvaluation > 0
                    || result[p + VisionResult.NEW_IMPRESSION] > .5f
                    || (flags & (1 | 2 | 4 | 8)) != 0) return true;
        }
        return false;
    }

    private static final class PreviewFrame {
        final int[] argb;
        final int width;
        final int height;

        PreviewFrame(int[] argb, int width, int height) {
            this.argb = argb;
            this.width = width;
            this.height = height;
        }
    }

    /**
     * Builds the CONFIG image from the exact normalized YUV planes consumed by
     * NCNN. The physical SurfaceView remains only as the vendor Camera2 output;
     * none of its compositor pixels are used by the visible preview.
     */
    private static PreviewFrame buildUprightPreview(Image image, int clockwiseRotation,
                                                     int maxLongSide) {
        try {
            int sourceWidth = image.getWidth();
            int sourceHeight = image.getHeight();
            int rotation = ((clockwiseRotation % 360) + 360) % 360;
            int logicalWidth = (rotation == 90 || rotation == 270) ? sourceHeight : sourceWidth;
            int logicalHeight = (rotation == 90 || rotation == 270) ? sourceWidth : sourceHeight;
            float scale = Math.min(1f, (float) maxLongSide / Math.max(logicalWidth, logicalHeight));
            int width = Math.max(90, Math.round(logicalWidth * scale));
            int height = Math.max(90, Math.round(logicalHeight * scale));
            int[] pixels = new int[width * height];
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer y = planes[0].getBuffer();
            ByteBuffer u = planes[1].getBuffer();
            ByteBuffer v = planes[2].getBuffer();
            int yp = y.position(), up = u.position(), vp = v.position();
            int yStride = planes[0].getRowStride();
            int uStride = planes[1].getRowStride(), vStride = planes[2].getRowStride();
            int uPixel = planes[1].getPixelStride(), vPixel = planes[2].getPixelStride();
            for (int row = 0; row < height; row++) {
                int logicalY = Math.min(logicalHeight - 1,
                        (int) (((row + .5f) * logicalHeight) / height));
                for (int col = 0; col < width; col++) {
                    int logicalX = Math.min(logicalWidth - 1,
                            (int) (((col + .5f) * logicalWidth) / width));
                    int sourceX;
                    int sourceY;
                    if (rotation == 90) {
                        sourceX = logicalY;
                        sourceY = sourceHeight - 1 - logicalX;
                    } else if (rotation == 180) {
                        sourceX = sourceWidth - 1 - logicalX;
                        sourceY = sourceHeight - 1 - logicalY;
                    } else if (rotation == 270) {
                        sourceX = sourceWidth - 1 - logicalY;
                        sourceY = logicalX;
                    } else {
                        sourceX = logicalX;
                        sourceY = logicalY;
                    }
                    int yy = (y.get(yp + sourceY * yStride + sourceX) & 255) - 16;
                    int chromaX = sourceX >> 1, chromaY = sourceY >> 1;
                    int uu = (u.get(up + chromaY * uStride + chromaX * uPixel) & 255) - 128;
                    int vv = (v.get(vp + chromaY * vStride + chromaX * vPixel) & 255) - 128;
                    int c = Math.max(0, yy);
                    int red = clampRgb((298 * c + 409 * vv + 128) >> 8);
                    int green = clampRgb((298 * c - 100 * uu - 208 * vv + 128) >> 8);
                    int blue = clampRgb((298 * c + 516 * uu + 128) >> 8);
                    pixels[row * width + col] = 0xFF000000 | (red << 16) | (green << 8) | blue;
                }
            }
            return new PreviewFrame(pixels, width, height);
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static int clampRgb(int value) {
        return Math.max(0, Math.min(255, value));
    }

    private static byte[] encodeRotatedJpeg(Image image, int quality, int clockwiseRotation) {
        try {
            int sourceWidth = image.getWidth(), sourceHeight = image.getHeight();
            int rotation = ((clockwiseRotation % 360) + 360) % 360;
            int width = (rotation == 90 || rotation == 270) ? sourceHeight : sourceWidth;
            int height = (rotation == 90 || rotation == 270) ? sourceWidth : sourceHeight;
            Image.Plane[] planes = image.getPlanes();
            byte[] nv21 = new byte[width * height * 3 / 2];
            ByteBuffer y = planes[0].getBuffer(), u = planes[1].getBuffer(), v = planes[2].getBuffer();
            int yp = y.position(), up = u.position(), vp = v.position();
            int yStride = planes[0].getRowStride();
            int uStride = planes[1].getRowStride(), vStride = planes[2].getRowStride();
            int uPixel = planes[1].getPixelStride(), vPixel = planes[2].getPixelStride();
            int out = 0;
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    int sourceX;
                    int sourceY;
                    if (rotation == 90) {
                        sourceX = row;
                        sourceY = sourceHeight - 1 - col;
                    } else if (rotation == 180) {
                        sourceX = sourceWidth - 1 - col;
                        sourceY = sourceHeight - 1 - row;
                    } else if (rotation == 270) {
                        sourceX = sourceWidth - 1 - row;
                        sourceY = col;
                    } else {
                        sourceX = col;
                        sourceY = row;
                    }
                    nv21[out++] = y.get(yp + sourceY * yStride + sourceX);
                }
            }
            for (int row = 0; row < height / 2; row++) {
                for (int col = 0; col < width / 2; col++) {
                    int logicalX = col * 2;
                    int logicalY = row * 2;
                    int sourceX;
                    int sourceY;
                    if (rotation == 90) {
                        sourceX = logicalY;
                        sourceY = sourceHeight - 1 - logicalX;
                    } else if (rotation == 180) {
                        sourceX = sourceWidth - 1 - logicalX;
                        sourceY = sourceHeight - 1 - logicalY;
                    } else if (rotation == 270) {
                        sourceX = sourceWidth - 1 - logicalY;
                        sourceY = logicalX;
                    } else {
                        sourceX = logicalX;
                        sourceY = logicalY;
                    }
                    int chromaX = Math.max(0, Math.min(sourceWidth / 2 - 1, sourceX / 2));
                    int chromaY = Math.max(0, Math.min(sourceHeight / 2 - 1, sourceY / 2));
                    nv21[out++] = v.get(vp + chromaY * vStride + chromaX * vPixel);
                    nv21[out++] = u.get(up + chromaY * uStride + chromaX * uPixel);
                }
            }
            ByteArrayOutputStream bytes = new ByteArrayOutputStream(width * height / 5);
            boolean ok = new YuvImage(nv21, ImageFormat.NV21, width, height, null)
                    .compressToJpeg(new Rect(0, 0, width, height), quality, bytes);
            return ok ? bytes.toByteArray() : null;
        } catch (Throwable ignored) { return null; }
    }

    private void scheduleReopen(String message) {
        listener.onCameraStatus(message, true);
        closeCamera();
        if (running && cameraHandler != null) {
            long delay = Math.min(5000L, 700L + (++reopenCount * 450L));
            cameraHandler.postDelayed(this::openCamera, delay);
        }
    }

    private void closeCamera() {
        processing.set(false);
        try { if (session != null) session.close(); } catch (Exception ignored) {}
        try { if (device != null) device.close(); } catch (Exception ignored) {}
        try { if (reader != null) reader.close(); } catch (Exception ignored) {}
        session = null; device = null; reader = null; lastFrameMs = 0; lastPreviewMs = 0;
    }
}