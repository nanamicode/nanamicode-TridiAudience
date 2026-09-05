package com.tridi.audience;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Single-window kiosk UI.
 *
 * Android TV firmwares commonly ignore Activity portrait requests. The whole
 * application therefore lives in one rotatable stage: advertisements, CONFIG,
 * camera preview and overlay always use the same selected orientation.
 *
 * The permanent SurfaceView behind the advertisement keeps the validated
 * Camera2 preview + YUV session alive. No raw USB authorization is used.
 */
public final class MainActivity extends Activity
        implements PlaylistPlayer.Listener, SurfaceHolder.Callback {
    private static final int PERMISSION_REQUEST = 51;
    private static final String PREFS = "tridi_state";
    // The physical stream remains the proven 16:9 Camera2 profile. The image
    // presented to AI and CONFIG is upright after the lateral camera rotation.
    private static final int CAMERA_LOGICAL_ASPECT_WIDTH = 9;
    private static final int CAMERA_LOGICAL_ASPECT_HEIGHT = 16;
    private static final int CAMERA_PHYSICAL_ASPECT_WIDTH = 16;
    private static final int CAMERA_PHYSICAL_ASPECT_HEIGHT = 9;
    // Stage rotation remains independent of the TV firmware orientation policy.
    private static final String PREF_ORIENTATION_LEGACY = "display_orientation_stage_v2";
    private static final String PREF_ROTATION = "display_rotation_stage_v3";
    private static final String PREF_ROOT_ROTATION = "root_rotation_available_v1";

    private final android.os.Handler ui = new android.os.Handler();
    private final ExecutorService rotationWorker = Executors.newSingleThreadExecutor();
    private PlaylistPlayer playlistPlayer;
    private FrameLayout displayHost;
    private FrameLayout orientationStage;
    private FrameLayout playerRoot;
    private FrameLayout configRoot;
    private LinearLayout settingsPanel;
    private SurfaceView cameraPreview;
    private CameraFrameView cameraFrameView;
    private OverlayView overlay;
    private TextView countsText;
    private TextView aiStatusText;
    private TextView serverStatusText;
    private TextView ipText;
    private TextView playerStatusText;
    private TextView calibrationStatusText;
    private TextView calibrationTarget;
    private Button aiButton;
    private Button calibrationButton;
    private Button serverButton;
    private Button orientationButton;
    private Button filesButton;
    private Button closeButton;
    private FileManagerView fileManagerView;
    private boolean resumed;
    private boolean configShowing;
    private boolean fileManagerShowing;
    private int displayRotation;
    private boolean previewAttached;
    private boolean rootRotationPending;
    private Bitmap normalizedPreviewBitmap;
    private long normalizedPreviewAt;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        displayRotation = loadDisplayRotation();
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
                | WindowManager.LayoutParams.FLAG_FULLSCREEN);
        hideSystemUi();

        displayHost = new FrameLayout(this);
        displayHost.setBackgroundColor(Color.BLACK);
        orientationStage = new FrameLayout(this);
        orientationStage.setBackgroundColor(Color.BLACK);
        orientationStage.setClipChildren(false);
        displayHost.addView(orientationStage,
                new FrameLayout.LayoutParams(-1, -1, Gravity.CENTER));

        createPlayerView();
        createSettingsView();
        setContentView(displayHost);

        displayHost.addOnLayoutChangeListener((view, left, top, right, bottom,
                                                oldLeft, oldTop, oldRight, oldBottom) -> {
            if ((right - left) != (oldRight - oldLeft)
                    || (bottom - top) != (oldBottom - oldTop)) {
                applyDisplayMode();
            }
        });

        playerRoot.requestFocus();
        requestNeededPermissions();
        ui.post(this::applyDisplayMode);
        ui.post(snapshotUpdater);
    }

    private void createPlayerView() {
        // Real, permanent preview surface.  It stays alive behind the TextureView
        // while advertisements play and becomes visible in CONFIG.
        cameraPreview = new SurfaceView(this);
        cameraPreview.setZOrderOnTop(false);
        cameraPreview.setZOrderMediaOverlay(false);
        cameraPreview.getHolder().setFormat(PixelFormat.OPAQUE);
        cameraPreview.getHolder().setFixedSize(1280, 720);
        cameraPreview.getHolder().addCallback(this);
        orientationStage.addView(cameraPreview, new FrameLayout.LayoutParams(-1, -1));

        playerRoot = new FrameLayout(this);
        playerRoot.setBackgroundColor(Color.BLACK);
        playerRoot.setFocusable(true);
        playerRoot.setFocusableInTouchMode(true);
        playerRoot.setClickable(true);
        playerRoot.setOnClickListener(view -> showSettings());

        TextureView videoSurface = new TextureView(this);
        videoSurface.setOpaque(true);
        playerRoot.addView(videoSurface, new FrameLayout.LayoutParams(-1, -1));

        playerStatusText = text(21, Color.WHITE);
        playerStatusText.setGravity(Gravity.CENTER);
        playerStatusText.setBackgroundColor(0xC0000000);
        playerStatusText.setPadding(24, 18, 24, 18);
        playerStatusText.setVisibility(View.GONE);
        FrameLayout.LayoutParams statusParams = new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM);
        statusParams.setMargins(18, 18, 18, 36);
        playerRoot.addView(playerStatusText, statusParams);

        orientationStage.addView(playerRoot, new FrameLayout.LayoutParams(-1, -1));
        playlistPlayer = new PlaylistPlayer(this, videoSurface, this);
    }

    private void createSettingsView() {
        configRoot = new FrameLayout(this);
        // This opaque layer fully covers the physical 16:9 SurfaceView. The
        // visible preview below is rebuilt from the same upright YUV image
        // consumed by NCNN, preventing compositor mosaics on Android TV.
        configRoot.setBackgroundColor(Color.BLACK);
        configRoot.setClipChildren(false);
        configRoot.setVisibility(View.GONE);

        cameraFrameView = new CameraFrameView(this);
        configRoot.addView(cameraFrameView, new FrameLayout.LayoutParams(-1, -1));

        overlay = new OverlayView(this);
        configRoot.addView(overlay, new FrameLayout.LayoutParams(-1, -1));

        calibrationTarget = text(23, Color.WHITE);
        calibrationTarget.setText("⊕  OLHE PARA ESTE PONTO");
        calibrationTarget.setGravity(Gravity.CENTER);
        calibrationTarget.setBackgroundColor(0xD0000000);
        calibrationTarget.setVisibility(View.GONE);
        FrameLayout.LayoutParams targetParams = new FrameLayout.LayoutParams(
                dp(310), dp(64), Gravity.CENTER);
        configRoot.addView(calibrationTarget, targetParams);

        settingsPanel = buildSettingsPanel();
        settingsPanel.addOnLayoutChangeListener((view, left, top, right, bottom,
                                                  oldLeft, oldTop, oldRight, oldBottom) -> {
            if ((right - left) != (oldRight - oldLeft)
                    || (bottom - top) != (oldBottom - oldTop)) {
                applyCameraFitLayout();
            }
        });
        configRoot.addView(settingsPanel);
        orientationStage.addView(configRoot, new FrameLayout.LayoutParams(-1, -1));

        fileManagerView = new FileManagerView(this, this::hideFileManager);
        fileManagerView.setVisibility(View.GONE);
        orientationStage.addView(fileManagerView, new FrameLayout.LayoutParams(-1, -1));
        applyConfigPanelLayout();
    }

    @Override protected void onResume() {
        super.onResume();
        resumed = true;
        hideSystemUi();
        if (canControlSystemRotation()) applySystemRotation();
        else if (rootRotationAvailable()) applyRootSystemRotation(false);
        ui.postDelayed(this::applyDisplayMode, 220L);
        attachPreviewIfReady();
        AudienceService.setNormalizedPreviewEnabled(configShowing && !fileManagerShowing);
        if (hasPermissions()) {
            ensureFolders();
            startAudienceService(AudienceService.ACTION_START);
            if (!configShowing) playlistPlayer.start();
        }
    }

    @Override protected void onPause() {
        resumed = false;
        AudienceService.setNormalizedPreviewEnabled(false);
        playlistPlayer.stop();
        super.onPause();
    }

    @Override protected void onDestroy() {
        ui.removeCallbacksAndMessages(null);
        rotationWorker.shutdownNow();
        if (fileManagerView != null) fileManagerView.shutdown();
        AudienceService.setNormalizedPreviewEnabled(false);
        detachPreview();
        playlistPlayer.destroy();
        super.onDestroy();
    }

    @Override public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) hideSystemUi();
    }

    @Override public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        hideSystemUi();
        ui.post(this::applyDisplayMode);
        ui.postDelayed(this::applyDisplayMode, 220L);
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grants) {
        super.onRequestPermissionsResult(requestCode, permissions, grants);
        if (requestCode != PERMISSION_REQUEST) return;
        restoreKioskUi();
        if (hasPermissions()) {
            ensureFolders();
            attachPreviewIfReady();
            startAudienceService(AudienceService.ACTION_START);
            if (!configShowing) playlistPlayer.start();
        } else {
            playerStatusText.setVisibility(View.VISIBLE);
            playerStatusText.setTextColor(0xFFFF8A80);
            playerStatusText.setText("Permita câmera e armazenamento para o totem funcionar.");
        }
    }

    @Override public void onPlayerStatus(String status, boolean error) {
        runOnUiThread(() -> {
            if (!error) {
                playerStatusText.setVisibility(View.GONE);
                return;
            }
            playerStatusText.setText(status);
            playerStatusText.setTextColor(0xFFFF8A80);
            playerStatusText.setVisibility(View.VISIBLE);
            ui.removeCallbacks(hidePlayerError);
            ui.postDelayed(hidePlayerError, 8000L);
        });
    }

    private final Runnable hidePlayerError = () -> playerStatusText.setVisibility(View.GONE);

    @Override public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (!configShowing && isConfigKey(keyCode)) {
            showSettings();
            return true;
        }
        if (fileManagerShowing && keyCode == KeyEvent.KEYCODE_BACK) {
            if (!fileManagerView.handleBack()) hideFileManager();
            return true;
        }
        if (configShowing && keyCode == KeyEvent.KEYCODE_BACK) {
            hideSettings();
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }

    private static boolean isConfigKey(int keyCode) {
        return keyCode == KeyEvent.KEYCODE_DPAD_CENTER
                || keyCode == KeyEvent.KEYCODE_ENTER
                || keyCode == KeyEvent.KEYCODE_NUMPAD_ENTER
                || keyCode == KeyEvent.KEYCODE_BUTTON_A
                || keyCode == KeyEvent.KEYCODE_MENU
                || keyCode == KeyEvent.KEYCODE_SETTINGS
                || keyCode == KeyEvent.KEYCODE_F1;
    }

    private LinearLayout buildSettingsPanel() {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setGravity(Gravity.CENTER_HORIZONTAL);
        panel.setPadding(dp(14), dp(10), dp(14), dp(10));
        GradientDrawable panelBackground = new GradientDrawable();
        panelBackground.setColor(0xE7101820);
        panelBackground.setCornerRadius(dp(14));
        panel.setBackground(panelBackground);

        countsText = text(20, 0xFFB2FF59);
        countsText.setGravity(Gravity.CENTER);
        panel.addView(countsText, row());

        aiStatusText = text(16, 0xFFCFD8DC);
        aiStatusText.setGravity(Gravity.CENTER);
        panel.addView(aiStatusText, row());

        calibrationStatusText = text(15, 0xFFFFD740);
        calibrationStatusText.setGravity(Gravity.CENTER);
        panel.addView(calibrationStatusText, row());

        aiButton = button("DESLIGAR IA");
        aiButton.setOnClickListener(view -> startAudienceService(AudienceService.ACTION_TOGGLE_AI));
        panel.addView(aiButton, row());

        calibrationButton = button("MVP LATERAL ATIVO • SEM CALIBRAÇÃO");
        calibrationButton.setEnabled(false);
        panel.addView(calibrationButton, row());

        serverStatusText = text(15, 0xFF90CAF9);
        serverStatusText.setGravity(Gravity.CENTER);
        panel.addView(serverStatusText, row());

        ipText = text(17, Color.WHITE);
        ipText.setGravity(Gravity.CENTER);
        // Status text must never capture the TV remote's DPAD focus.
        ipText.setFocusable(false);
        ipText.setFocusableInTouchMode(false);
        ipText.setClickable(false);
        ipText.setLongClickable(false);
        panel.addView(ipText, row());

        serverButton = button("LIGAR SERVIDOR WEB");
        serverButton.setOnClickListener(view -> startAudienceService(AudienceService.ACTION_TOGGLE_SERVER));
        panel.addView(serverButton, row());

        orientationButton = button("ORIENTAÇÃO DA TELA");
        orientationButton.setOnClickListener(view -> toggleOrientation());
        panel.addView(orientationButton, row());

        filesButton = button("GERENCIADOR DE ARQUIVOS");
        filesButton.setOnClickListener(view -> showFileManager());
        panel.addView(filesButton, row());

        closeButton = button("VOLTAR AOS ANÚNCIOS");
        closeButton.setOnClickListener(view -> hideSettings());
        panel.addView(closeButton, row());
        configureButtonFocusOrder();
        return panel;
    }

    private void startTargetCalibration() {
        Toast.makeText(this,
                "O alvo lateral já está ativo. Não existe calibração manual nesta versão.",
                Toast.LENGTH_LONG).show();
    }

    private void showSettings() {
        if (configShowing) return;
        playlistPlayer.stop();
        playerStatusText.setVisibility(View.GONE);
        configShowing = true;
        AudienceService.setNormalizedPreviewEnabled(true);
        playerRoot.setVisibility(View.GONE);
        configRoot.setVisibility(View.VISIBLE);
        updateSnapshot();
        applyCameraFitLayout();
        attachPreviewIfReady();
        ui.postDelayed(() -> {
            applyCameraFitLayout();
            attachPreviewIfReady();
            aiButton.requestFocus();
        }, 160L);
    }

    private void hideSettings() {
        if (!configShowing) return;
        if (AudienceService.snapshot().calibrationRunning) {
            Toast.makeText(this, "Conclua os 21 segundos da calibração antes de sair.",
                    Toast.LENGTH_SHORT).show();
            return;
        }
        fileManagerShowing = false;
        if (fileManagerView != null) fileManagerView.setVisibility(View.GONE);
        configShowing = false;
        AudienceService.setNormalizedPreviewEnabled(false);
        configRoot.setVisibility(View.GONE);
        playerRoot.setVisibility(View.VISIBLE);
        applyCameraFitLayout();
        playerRoot.requestFocus();
        hideSystemUi();
        if (resumed && hasPermissions()) ui.postDelayed(playlistPlayer::start, 180L);
    }

    private void showFileManager() {
        if (!configShowing || fileManagerView == null) return;
        if (AudienceService.snapshot().calibrationRunning) return;
        fileManagerShowing = true;
        AudienceService.setNormalizedPreviewEnabled(false);
        configRoot.setVisibility(View.GONE);
        fileManagerView.setVisibility(View.VISIBLE);
        fileManagerView.bringToFront();
        fileManagerView.openHome();
        hideSystemUi();
    }

    private void hideFileManager() {
        if (!fileManagerShowing) return;
        fileManagerShowing = false;
        AudienceService.setNormalizedPreviewEnabled(true);
        fileManagerView.setVisibility(View.GONE);
        configRoot.setVisibility(View.VISIBLE);
        applyCameraFitLayout();
        updateSnapshot();
        ui.postDelayed(() -> filesButton.requestFocus(), 80L);
        hideSystemUi();
    }

    private void attachPreviewIfReady() {
        if (previewAttached) return;
        SurfaceHolder holder = cameraPreview.getHolder();
        if (holder.getSurface() != null && holder.getSurface().isValid()) {
            previewAttached = true;
            AudienceService.attachCameraPreview(holder);
        }
    }

    private void detachPreview() {
        if (!previewAttached || cameraPreview == null) return;
        previewAttached = false;
        AudienceService.detachCameraPreview(cameraPreview.getHolder());
    }

    @Override public void surfaceCreated(SurfaceHolder holder) {
        attachPreviewIfReady();
    }

    @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {}

    @Override public void surfaceDestroyed(SurfaceHolder holder) {
        detachPreview();
    }

    private final Runnable snapshotUpdater = new Runnable() {
        @Override public void run() {
            updateSnapshot();
            if (configShowing) {
                overlay.setVision(AudienceService.latestVision());
                updateNormalizedCameraFrame();
            }
            ui.postDelayed(this, configShowing ? 70L : 1000L);
        }
    };

    private void updateNormalizedCameraFrame() {
        if (!configShowing || fileManagerShowing || cameraFrameView == null) return;
        AudienceService.PreviewFrame preview = AudienceService.latestPreview();
        if (preview == null || preview.capturedAt == normalizedPreviewAt) return;
        if (normalizedPreviewBitmap == null
                || normalizedPreviewBitmap.getWidth() != preview.width
                || normalizedPreviewBitmap.getHeight() != preview.height) {
            normalizedPreviewBitmap = Bitmap.createBitmap(
                    preview.width, preview.height, Bitmap.Config.ARGB_8888);
        }
        normalizedPreviewBitmap.setPixels(preview.argb, 0, preview.width,
                0, 0, preview.width, preview.height);
        normalizedPreviewAt = preview.capturedAt;
        cameraFrameView.setFrame(normalizedPreviewBitmap);
    }

    private void updateSnapshot() {
        AudienceService.Snapshot snapshot = AudienceService.snapshot();
        if (countsText == null) return;
        countsText.setText("ALCANCE " + snapshot.reach + "   •   IMPRESSÕES " + snapshot.impressions
                + "\nMASC. " + snapshot.male + "   •   FEM. " + snapshot.female);
        aiStatusText.setText(snapshot.aiStatus);
        calibrationStatusText.setText(snapshot.calibrationStatus);
        calibrationStatusText.setTextColor(snapshot.calibrationRunning
                ? 0xFFFFD740 : (snapshot.calibrationReady ? 0xFFB2FF59 : 0xFFFF8A80));
        calibrationTarget.setVisibility(snapshot.calibrationRunning ? View.VISIBLE : View.GONE);
        aiButton.setText(snapshot.aiEnabled ? "DESLIGAR IA" : "LIGAR IA");
        calibrationButton.setText("MVP LATERAL ATIVO • SEM CALIBRAÇÃO");
        calibrationButton.setEnabled(false);
        aiButton.setEnabled(!snapshot.calibrationRunning);
        orientationButton.setEnabled(!snapshot.calibrationRunning);
        filesButton.setEnabled(!snapshot.calibrationRunning);
        closeButton.setEnabled(!snapshot.calibrationRunning);
        serverStatusText.setText(snapshot.serverStatus + "   •   FILA " + snapshot.queued);
        ipText.setText(snapshot.serverEnabled ? snapshot.address : "Servidor web desligado");
        serverButton.setText(snapshot.serverEnabled ? "DESLIGAR SERVIDOR WEB" : "LIGAR SERVIDOR WEB");
        orientationButton.setText("TELA: " + orientationLabel()
                + (canControlSystemRotation()
                ? "  •  APP/CÂMERA/SISTEMA"
                : (rootRotationAvailable()
                ? "  •  APP/CÂMERA/SISTEMA ROOT"
                : "  •  APP/CÂMERA • TENTAR SISTEMA")));
        boolean cameraError = !snapshot.aiStatus.startsWith("IA ativa")
                && !snapshot.aiStatus.contains("Abrindo")
                && !snapshot.aiStatus.contains("Inicializando");
        aiStatusText.setTextColor(cameraError ? 0xFFFF8A80 : 0xFFCFD8DC);
        overlay.setStatus(snapshot.aiStatus, cameraError);
    }

    private void startAudienceService(String action) {
        Intent service = new Intent(this, AudienceService.class).setAction(action);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) startForegroundService(service);
        else startService(service);
    }

    private int loadDisplayRotation() {
        android.content.SharedPreferences preferences = getSharedPreferences(PREFS, MODE_PRIVATE);
        if (preferences.contains(PREF_ROTATION)) {
            return normalizeRotation(preferences.getInt(PREF_ROTATION, 0));
        }
        // Preserve the user's selection when updating from the two-position build.
        String saved = preferences.getString(PREF_ORIENTATION_LEGACY, "auto");
        if ("portrait".equals(saved) || "vertical".equals(saved)) return 90;
        if ("landscape".equals(saved) || "horizontal".equals(saved)) return 0;
        return getResources().getDisplayMetrics().heightPixels
                > getResources().getDisplayMetrics().widthPixels ? 90 : 0;
    }

    private void toggleOrientation() {
        displayRotation = normalizeRotation(displayRotation + 90);
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putInt(PREF_ROTATION, displayRotation)
                .apply();
        boolean systemApplied = applySystemRotation();
        applyDisplayMode();
        applyConfigPanelLayout();
        updateSnapshot();
        orientationButton.requestFocus();
        if (!systemApplied) {
            if (rootRotationAvailable()) applyRootSystemRotation(false);
            else requestSystemRotationPermission();
        }
    }

    private static int normalizeRotation(int value) {
        int normalized = value % 360;
        if (normalized < 0) normalized += 360;
        return (normalized / 90) * 90;
    }

    private boolean isVerticalMode() {
        return displayRotation == 90 || displayRotation == 270;
    }

    private String orientationLabel() {
        if (displayRotation == 90) return "VERTICAL 90°";
        if (displayRotation == 180) return "HORIZONTAL INVERTIDA 180°";
        if (displayRotation == 270) return "VERTICAL INVERTIDA 270°";
        return "HORIZONTAL 0°";
    }

    /**
     * The Android display performs as much of the rotation as the TV firmware
     * allows. The stage supplies only the missing residual rotation, so the app
     * remains usable even on boxes that ignore USER_ROTATION.
     */
    private void applyDisplayMode() {
        if (displayHost == null || orientationStage == null) return;
        int hostWidth = displayHost.getWidth();
        int hostHeight = displayHost.getHeight();
        if (hostWidth <= 0 || hostHeight <= 0) {
            displayHost.post(this::applyDisplayMode);
            return;
        }

        final int residualRotation = residualDisplayRotation();
        final boolean rotatedStage = residualRotation == 90 || residualRotation == 270;
        final float rotation = residualRotation;
        FrameLayout.LayoutParams params = rotatedStage
                ? new FrameLayout.LayoutParams(hostHeight, hostWidth, Gravity.CENTER)
                : new FrameLayout.LayoutParams(hostWidth, hostHeight, Gravity.CENTER);
        orientationStage.setLayoutParams(params);
        orientationStage.setRotation(rotation);
        orientationStage.post(() -> {
            orientationStage.setPivotX(orientationStage.getWidth() / 2f);
            orientationStage.setPivotY(orientationStage.getHeight() / 2f);
            orientationStage.setRotation(rotation);
            applyCameraFitLayout();
            applyConfigPanelLayout();
            if (playlistPlayer != null) playlistPlayer.refreshLayout();
        });
    }

    /**
     * Keep the complete camera frame visible without touching Camera2 or NCNN.
     * The validated 1280x720 SurfaceView is never detached or reconfigured.
     * SurfaceView content is a separate compositor layer on many TV boxes. It
     * remains behind an opaque hierarchy solely to satisfy Camera2. The visible
     * CameraFrameView receives an already upright copy of the analysis YUV,
     * so preview, overlay and native inference share one coordinate system.
     */
    private void applyCameraFitLayout() {
        if (orientationStage == null || cameraPreview == null || overlay == null) return;
        int stageWidth = orientationStage.getWidth();
        int stageHeight = orientationStage.getHeight();
        if (stageWidth <= 0 || stageHeight <= 0) {
            orientationStage.post(this::applyCameraFitLayout);
            return;
        }
        boolean verticalMode = isVerticalMode();

        // In CONFIG reserve a separate area for the controls. The complete
        // frame stays visible and the panel no longer hides its right/bottom.
        int contentLeft = 0;
        int contentTop = 0;
        int availableWidth = stageWidth;
        int availableHeight = stageHeight;
        if (configShowing && settingsPanel != null) {
            if (verticalMode) {
                int panelHeight = settingsPanel.getHeight();
                if (panelHeight <= 0) panelHeight = settingsPanel.getMeasuredHeight();
                if (panelHeight <= 0) panelHeight = dp(430);
                availableHeight = Math.max(dp(180),
                        stageHeight - panelHeight - dp(34));
            } else {
                int panelWidth = settingsPanel.getWidth();
                if (panelWidth <= 0) panelWidth = settingsPanel.getMeasuredWidth();
                if (panelWidth <= 0) panelWidth = dp(460);
                availableWidth = Math.max(dp(240),
                        stageWidth - panelWidth - dp(36));
            }
        }

        int logicalAspectWidth = CAMERA_LOGICAL_ASPECT_WIDTH;
        int logicalAspectHeight = CAMERA_LOGICAL_ASPECT_HEIGHT;
        int fittedWidth = availableWidth;
        int fittedHeight = Math.round((float) fittedWidth
                * logicalAspectHeight / logicalAspectWidth);
        if (fittedHeight > availableHeight) {
            fittedHeight = availableHeight;
            fittedWidth = Math.round((float) fittedHeight
                    * logicalAspectWidth / logicalAspectHeight);
        }

        int fittedLeft = contentLeft + (availableWidth - fittedWidth) / 2;
        int fittedTop = contentTop + (availableHeight - fittedHeight) / 2;

        // Keep the real SurfaceView in the Camera2 stream's proven 16:9 shape.
        // The opaque CONFIG/player layers hide it; changing or removing this
        // vendor-required output can freeze the USB camera on the BTV B11.
        int cameraWidth = availableWidth;
        int cameraHeight = Math.round((float) cameraWidth
                * CAMERA_PHYSICAL_ASPECT_HEIGHT / CAMERA_PHYSICAL_ASPECT_WIDTH);
        if (cameraHeight > availableHeight) {
            cameraHeight = availableHeight;
            cameraWidth = Math.round((float) cameraHeight
                    * CAMERA_PHYSICAL_ASPECT_WIDTH / CAMERA_PHYSICAL_ASPECT_HEIGHT);
        }
        FrameLayout.LayoutParams cameraParams = new FrameLayout.LayoutParams(
                cameraWidth, cameraHeight, Gravity.TOP | Gravity.START);
        cameraParams.leftMargin = contentLeft + (availableWidth - cameraWidth) / 2;
        cameraParams.topMargin = contentTop + (availableHeight - cameraHeight) / 2;
        cameraPreview.setLayoutParams(cameraParams);
        cameraPreview.setPivotX(cameraWidth / 2f);
        cameraPreview.setPivotY(cameraHeight / 2f);
        cameraPreview.setRotation(0f);

        FrameLayout.LayoutParams frameParams = new FrameLayout.LayoutParams(
                fittedWidth, fittedHeight, Gravity.TOP | Gravity.START);
        frameParams.leftMargin = fittedLeft;
        frameParams.topMargin = fittedTop;
        cameraFrameView.setLayoutParams(frameParams);
        // Pixels from AudienceService.PreviewFrame are already normalized.
        cameraFrameView.setFrameRotation(0);

        FrameLayout.LayoutParams overlayParams = new FrameLayout.LayoutParams(
                fittedWidth, fittedHeight, Gravity.TOP | Gravity.START);
        overlayParams.leftMargin = fittedLeft;
        overlayParams.topMargin = fittedTop;
        overlay.setLayoutParams(overlayParams);
        // CameraFrameView and OverlayView are ordinary children of the same
        // rotatable stage, so both inherit the layout angle together.
        overlay.setFrameRotation(0);
    }

    private void applyConfigPanelLayout() {
        if (configRoot == null || settingsPanel == null) return;
        boolean verticalMode = isVerticalMode();
        FrameLayout.LayoutParams panelParams = verticalMode
                ? new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM)
                : new FrameLayout.LayoutParams(dp(460), -2, Gravity.END | Gravity.CENTER_VERTICAL);
        panelParams.setMargins(dp(12), dp(10), dp(12), dp(14));
        settingsPanel.setLayoutParams(panelParams);
        settingsPanel.post(this::applyCameraFitLayout);
    }

    private void configureButtonFocusOrder() {
        Button[] buttons = {aiButton, calibrationButton, serverButton,
                orientationButton, filesButton, closeButton};
        for (Button button : buttons) button.setId(View.generateViewId());
        for (int i = 0; i < buttons.length; i++) {
            Button current = buttons[i];
            Button previous = buttons[(i + buttons.length - 1) % buttons.length];
            Button next = buttons[(i + 1) % buttons.length];
            current.setNextFocusUpId(previous.getId());
            current.setNextFocusDownId(next.getId());
            current.setNextFocusForwardId(next.getId());
        }
    }

    private boolean canControlSystemRotation() {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.M || Settings.System.canWrite(this);
    }

    /** Use Android's supported user-rotation setting; no root or shell access. */
    private boolean applySystemRotation() {
        if (!canControlSystemRotation()) return false;
        try {
            Settings.System.putInt(getContentResolver(),
                    Settings.System.ACCELEROMETER_ROTATION, 0);
            boolean written = Settings.System.putInt(getContentResolver(),
                    Settings.System.USER_ROTATION, rotationSetting(displayRotation));
            if (written) {
                ui.postDelayed(this::applyDisplayMode, 220L);
                ui.postDelayed(this::applyDisplayMode, 650L);
            }
            return written;
        } catch (SecurityException exception) {
            return false;
        }
    }

    private void requestSystemRotationPermission() {
        Toast.makeText(this,
                "Autorize ‘Modificar configurações do sistema’ para girar toda a TV Box.",
                Toast.LENGTH_LONG).show();
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_VISIBLE);
        try {
            Intent permission = new Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS,
                    Uri.parse("package:" + getPackageName()));
            startActivity(permission);
        } catch (ActivityNotFoundException exception) {
            try {
                startActivity(new Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS));
            } catch (ActivityNotFoundException ignored) {
                Toast.makeText(this,
                        "O firmware removeu a autorização normal. Se aparecer uma janela "
                                + "de Superusuário, toque em PERMITIR.",
                        Toast.LENGTH_LONG).show();
                restoreKioskUi();
                applyRootSystemRotation(true);
            }
        }
    }

    private boolean rootRotationAvailable() {
        return getSharedPreferences(PREFS, MODE_PRIVATE)
                .getBoolean(PREF_ROOT_ROTATION, false);
    }

    /** Optional fallback for rooted TV boxes whose vendor removed WRITE_SETTINGS UI. */
    private void applyRootSystemRotation(boolean openDisplaySettingsOnFailure) {
        if (rootRotationPending) return;
        rootRotationPending = true;
        final int wantedDegrees = displayRotation;
        final int wantedSetting = rotationSetting(wantedDegrees);
        rotationWorker.execute(() -> {
            boolean accepted = false;
            Process process = null;
            try {
                String command = "settings put system accelerometer_rotation 0; "
                        + "settings put system user_rotation " + wantedSetting + "; "
                        + "wm user-rotation lock " + wantedSetting
                        + " >/dev/null 2>&1 || wm set-user-rotation lock " + wantedSetting
                        + " >/dev/null 2>&1 || true";
                process = new ProcessBuilder("su", "-c", command)
                        .redirectErrorStream(true).start();
                boolean finished = process.waitFor(45L, TimeUnit.SECONDS);
                accepted = finished && process.exitValue() == 0;
                if (!finished) process.destroy();
            } catch (Exception ignored) {
                if (process != null) process.destroy();
            }
            final boolean rootAccepted = accepted;
            runOnUiThread(() -> {
                if (isFinishing() || isDestroyed()) return;
                rootRotationPending = false;
                if (rootAccepted) {
                    getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                            .putBoolean(PREF_ROOT_ROTATION, true).apply();
                    ui.postDelayed(this::applyDisplayMode, 350L);
                    ui.postDelayed(() -> reportRootRotationResult(wantedDegrees), 1400L);
                } else {
                    getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                            .putBoolean(PREF_ROOT_ROTATION, false).apply();
                    Toast.makeText(this,
                            "A TV Box não concedeu acesso root. Vou abrir as configurações de tela.",
                            Toast.LENGTH_LONG).show();
                    if (openDisplaySettingsOnFailure) openDisplaySettings();
                }
                updateSnapshot();
            });
        });
    }

    private void reportRootRotationResult(int wantedDegrees) {
        applyDisplayMode();
        boolean displayChanged = currentSystemRotation() == wantedDegrees;
        Toast.makeText(this, displayChanged
                        ? "Rotação aplicada ao sistema inteiro."
                        : "O acesso foi concedido, mas este firmware ainda ignorou a rotação global. "
                        + "O aplicativo e a câmera continuarão no ângulo correto.",
                Toast.LENGTH_LONG).show();
    }

    private void openDisplaySettings() {
        try {
            startActivity(new Intent(Settings.ACTION_DISPLAY_SETTINGS));
        } catch (ActivityNotFoundException exception) {
            Toast.makeText(this,
                    "O firmware também não possui uma tela de rotação. Fora do aplicativo, "
                            + "isso só pode ser liberado modificando o firmware.",
                    Toast.LENGTH_LONG).show();
            restoreKioskUi();
        }
    }

    private int residualDisplayRotation() {
        return normalizeRotation(displayRotation - currentSystemRotation());
    }

    private int currentSystemRotation() {
        int rotation = getWindowManager().getDefaultDisplay().getRotation();
        if (rotation == Surface.ROTATION_90) return 90;
        if (rotation == Surface.ROTATION_180) return 180;
        if (rotation == Surface.ROTATION_270) return 270;
        return 0;
    }

    private static int rotationSetting(int degrees) {
        if (degrees == 90) return Surface.ROTATION_90;
        if (degrees == 180) return Surface.ROTATION_180;
        if (degrees == 270) return Surface.ROTATION_270;
        return Surface.ROTATION_0;
    }

    private void restoreKioskUi() {
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        hideSystemUi();
    }

    private void requestNeededPermissions() {
        List<String> missing = new ArrayList<>();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.CAMERA);
        }
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.READ_EXTERNAL_STORAGE);
        }
        if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            missing.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        if (!missing.isEmpty()) {
            // The only authorization flow is Android's ordinary runtime CAMERA
            // permission. It is requested once; Camera2 owns the USB webcam.
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_VISIBLE);
            requestPermissions(missing.toArray(new String[0]), PERMISSION_REQUEST);
        }
    }

    private boolean hasPermissions() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    private void ensureFolders() {
        new File(Environment.getExternalStorageDirectory(), "TRIDI_VIDEOS").mkdirs();
    }

    private void hideSystemUi() {
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    private TextView text(int size, int color) {
        TextView view = new TextView(this);
        view.setTextSize(size);
        view.setTextColor(color);
        view.setPadding(dp(8), dp(3), dp(8), dp(3));
        return view;
    }

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextSize(16);
        button.setTextColor(Color.WHITE);
        button.setFocusable(true);
        button.setFocusableInTouchMode(true);
        button.setMinHeight(dp(46));
        button.setPadding(dp(14), dp(6), dp(14), dp(6));
        applyButtonBackground(button, false);
        button.setOnFocusChangeListener((view, focused) -> applyButtonBackground((Button) view, focused));
        return button;
    }

    private void applyButtonBackground(Button button, boolean focused) {
        GradientDrawable background = new GradientDrawable();
        background.setColor(focused ? 0xFF00838F : 0xFF006064);
        background.setCornerRadius(dp(8));
        background.setStroke(focused ? dp(4) : dp(1), focused ? Color.WHITE : 0xFF4DB6AC);
        button.setBackground(background);
        button.setElevation(focused ? dp(8) : dp(2));
    }

    private LinearLayout.LayoutParams row() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.setMargins(0, dp(2), 0, dp(2));
        return params;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}