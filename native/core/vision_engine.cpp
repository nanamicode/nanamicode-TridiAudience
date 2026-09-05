// Offline audience face/gender inference for Android. No image is retained.
// SCRFD decoding is derived from Tencent ncnn's BSD-3 sample implementation.
// v1.14.6 keeps every classification tied to the current YUV frame: raw
// landmarks are used for attention and alignment while smoothed landmarks are
// used only to make the preview rectangle stable.
#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#include "net.h"
#include "cpu.h"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "TridiVision", __VA_ARGS__)

namespace {

struct Point { float x = 0, y = 0; };
struct Rect { float x = 0, y = 0, w = 0, h = 0; };
struct Face { Rect box; Point lm[5]; float score = 0; };

static float clampf(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }
static int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }
static float area(const Rect& r) { return std::max(0.f, r.w) * std::max(0.f, r.h); }
static float iou(const Rect& a, const Rect& b) {
    const float x0 = std::max(a.x, b.x), y0 = std::max(a.y, b.y);
    const float x1 = std::min(a.x + a.w, b.x + b.w), y1 = std::min(a.y + a.h, b.y + b.h);
    const float inter = std::max(0.f, x1 - x0) * std::max(0.f, y1 - y0);
    return inter / std::max(1.f, area(a) + area(b) - inter);
}

struct YuvFrame {
    const uint8_t *y, *u, *v;
    // width/height are the upright logical coordinates consumed by every
    // detector, landmark, crop and tracker. sourceWidth/sourceHeight retain
    // the physical Camera2 plane layout.
    int width, height, sourceWidth, sourceHeight;
    int yStride, uStride, vStride, uPixel, vPixel, clockwiseRotation;

    void sourceCoordinate(int px, int py, int& sourceX, int& sourceY) const {
        px = clampi(px, 0, width - 1); py = clampi(py, 0, height - 1);
        if (clockwiseRotation == 90) {
            sourceX = py;
            sourceY = sourceHeight - 1 - px;
        } else if (clockwiseRotation == 180) {
            sourceX = sourceWidth - 1 - px;
            sourceY = sourceHeight - 1 - py;
        } else if (clockwiseRotation == 270) {
            sourceX = sourceWidth - 1 - py;
            sourceY = px;
        } else {
            sourceX = px;
            sourceY = py;
        }
        sourceX = clampi(sourceX, 0, sourceWidth - 1);
        sourceY = clampi(sourceY, 0, sourceHeight - 1);
    }

    void rgbAt(int px, int py, uint8_t* out) const {
        int sourceX, sourceY;
        sourceCoordinate(px, py, sourceX, sourceY);
        int yy = y[sourceY * yStride + sourceX] & 255;
        int uvx = sourceX >> 1, uvy = sourceY >> 1;
        int uu = (u[uvy * uStride + uvx * uPixel] & 255) - 128;
        int vv = (v[uvy * vStride + uvx * vPixel] & 255) - 128;
        int c = std::max(0, yy - 16);
        out[0] = (uint8_t)clampi((298 * c + 409 * vv + 128) >> 8, 0, 255);
        out[1] = (uint8_t)clampi((298 * c - 100 * uu - 208 * vv + 128) >> 8, 0, 255);
        out[2] = (uint8_t)clampi((298 * c + 516 * uu + 128) >> 8, 0, 255);
    }

    void rgbBilinear(float fx, float fy, uint8_t* out) const {
        int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
        float dx = fx - x0, dy = fy - y0;
        uint8_t c00[3], c10[3], c01[3], c11[3];
        rgbAt(x0, y0, c00); rgbAt(x0 + 1, y0, c10);
        rgbAt(x0, y0 + 1, c01); rgbAt(x0 + 1, y0 + 1, c11);
        for (int k = 0; k < 3; ++k) {
            float a = c00[k] + dx * (c10[k] - c00[k]);
            float b = c01[k] + dx * (c11[k] - c01[k]);
            out[k] = (uint8_t)clampi((int)std::lround(a + dy * (b - a)), 0, 255);
        }
    }

    void resizeRgb(int longSide, std::vector<uint8_t>& dst, int& outW, int& outH) const {
        float s = (float)longSide / std::max(width, height);
        outW = std::max(32, (int)std::lround(width * s));
        outH = std::max(32, (int)std::lround(height * s));
        dst.resize((size_t)outW * outH * 3);
        for (int j = 0; j < outH; ++j) {
            float sy = (j + .5f) * height / outH - .5f;
            for (int i = 0; i < outW; ++i) {
                float sx = (i + .5f) * width / outW - .5f;
                rgbBilinear(sx, sy, &dst[((size_t)j * outW + i) * 3]);
            }
        }
    }
};

static void generateProposals(int baseSize, int stride, const ncnn::Mat& scores,
                              const ncnn::Mat& boxes, const ncnn::Mat& kps,
                              float threshold, std::vector<Face>& out) {
    const int w = scores.w, h = scores.h;
    for (int q = 0; q < 2; ++q) {
        const float size = baseSize * (q + 1.f);
        const float ax0 = -size * .5f, ay0 = -size * .5f;
        const ncnn::Mat score = scores.channel(q);
        const ncnn::Mat bbox = boxes.channel_range(q * 4, 4);
        const ncnn::Mat lm = kps.empty() ? ncnn::Mat() : kps.channel_range(q * 10, 10);
        for (int iy = 0; iy < h; ++iy) for (int ix = 0; ix < w; ++ix) {
            int idx = iy * w + ix;
            float prob = score[idx];
            if (prob < threshold) continue;
            float cx = ax0 + ix * stride + size * .5f;
            float cy = ay0 + iy * stride + size * .5f;
            Face f;
            float x0 = cx - bbox.channel(0)[idx] * stride;
            float y0 = cy - bbox.channel(1)[idx] * stride;
            float x1 = cx + bbox.channel(2)[idx] * stride;
            float y1 = cy + bbox.channel(3)[idx] * stride;
            f.box = {x0, y0, x1 - x0 + 1.f, y1 - y0 + 1.f}; f.score = prob;
            for (int p = 0; p < 5 && !lm.empty(); ++p) {
                f.lm[p].x = cx + lm.channel(p * 2)[idx] * stride;
                f.lm[p].y = cy + lm.channel(p * 2 + 1)[idx] * stride;
            }
            out.push_back(f);
        }
    }
}

class Scrfd {
public:
    bool load(AAssetManager* assets) {
        net.opt.num_threads = 2;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = false;
        return net.load_param(assets, "scrfd_500m_kps-opt2.param") == 0 &&
               net.load_model(assets, "scrfd_500m_kps-opt2.bin") == 0;
    }

    std::vector<Face> detect(const YuvFrame& frame, int longSide, float threshold) {
        int sw, sh;
        frame.resizeRgb(longSide, rgbScratch, sw, sh);
        ncnn::Mat input = ncnn::Mat::from_pixels(rgbScratch.data(), ncnn::Mat::PIXEL_RGB, sw, sh);
        int wp = (sw + 31) / 32 * 32 - sw, hp = (sh + 31) / 32 * 32 - sh;
        ncnn::Mat padded;
        ncnn::copy_make_border(input, padded, hp / 2, hp - hp / 2, wp / 2, wp - wp / 2,
                               ncnn::BORDER_CONSTANT, 0.f);
        const float mean[3] = {127.5f, 127.5f, 127.5f};
        const float norm[3] = {1.f / 128, 1.f / 128, 1.f / 128};
        padded.substract_mean_normalize(mean, norm);
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input.1", padded);
        std::vector<Face> proposals; proposals.reserve(96);
        const int strides[3] = {8, 16, 32}, bases[3] = {16, 64, 256};
        for (int z = 0; z < 3; ++z) {
            ncnn::Mat s, b, k;
            std::string suffix = std::to_string(strides[z]);
            if (ex.extract(("score_" + suffix).c_str(), s) != 0) continue;
            ex.extract(("bbox_" + suffix).c_str(), b);
            ex.extract(("kps_" + suffix).c_str(), k);
            generateProposals(bases[z], strides[z], s, b, k, threshold, proposals);
        }
        std::sort(proposals.begin(), proposals.end(), [](const Face& a, const Face& b){ return a.score > b.score; });
        std::vector<Face> result; result.reserve(10);
        for (const Face& raw : proposals) {
            bool keep = true;
            for (const Face& old : result) if (iou(raw.box, old.box) > .40f) { keep = false; break; }
            if (!keep) continue;
            Face f = raw;
            float sx = (float)frame.width / sw, sy = (float)frame.height / sh;
            f.box.x = (raw.box.x - wp / 2.f) * sx; f.box.y = (raw.box.y - hp / 2.f) * sy;
            f.box.w = raw.box.w * sx; f.box.h = raw.box.h * sy;
            f.box.x = clampf(f.box.x, 0, frame.width - 1.f);
            f.box.y = clampf(f.box.y, 0, frame.height - 1.f);
            f.box.w = clampf(f.box.w, 1, frame.width - f.box.x);
            f.box.h = clampf(f.box.h, 1, frame.height - f.box.y);
            for (int p = 0; p < 5; ++p) {
                f.lm[p].x = clampf((raw.lm[p].x - wp / 2.f) * sx, 0, frame.width - 1.f);
                f.lm[p].y = clampf((raw.lm[p].y - hp / 2.f) * sy, 0, frame.height - 1.f);
            }
            if (f.box.w >= 20 && f.box.h >= 20) result.push_back(f);
            if (result.size() >= 10) break;
        }
        return result;
    }
private:
    ncnn::Net net;
    std::vector<uint8_t> rgbScratch;
};

struct AttentionResult {
    bool accepted = false;
    bool geometryValid = false;
    float score = 0;
    float pitchRatio = 0;
    float signedYawEye = 0;
    float signedYawMouth = 0;
    float faceScale = 0;
    float rollDegrees = 0;
    float expectedYawEye = 0;
    float expectedYawMouth = 0;
    float targetResidual = 9;
};

struct AttentionCalibration {
    bool enabled = false;
    float eyeIntercept = 0;
    float eyeSlope = 0;
    float mouthIntercept = 0;
    float mouthSlope = 0;
    float eyeTolerance = .11f;
    float mouthTolerance = .15f;
};

// SCRFD provides five stable landmarks rather than iris gaze. Every detected
// face is treated as a pseudo-impression for the lifetime of its track. This
// filter answers one binary question for each available frame: did the head
// point toward the physical display at least once? Body position, image x/y,
// apparent face size and walking direction are deliberately absent. The first
// positive frame becomes the evidence frame; negative frames never reset the
// track or require a consecutive pose sequence.
static AttentionResult evaluateAttention(const Face& f, int frameWidth,
                                         const AttentionCalibration& calibration) {
    AttentionResult result;
    const Point &le=f.lm[0], &re=f.lm[1], &nose=f.lm[2], &lm=f.lm[3], &rm=f.lm[4];
    float eye = std::hypot(re.x - le.x, re.y - le.y);
    float mouth = std::hypot(rm.x - lm.x, rm.y - lm.y);
    if (eye < 8 || mouth < 4 || re.x <= le.x || rm.x <= lm.x) return result;
    float eyeMidX=(le.x+re.x)*.5f, eyeMidY=(le.y+re.y)*.5f;
    float mouthMidX=(lm.x+rm.x)*.5f, mouthMidY=(lm.y+rm.y)*.5f;
    float eyeNose=nose.y-eyeMidY, noseMouth=mouthMidY-nose.y;
    if(eyeNose<=eye*.10f||noseMouth<=eye*.10f)return result;
    float roll = std::fabs(std::atan2(re.y - le.y, re.x - le.x) * 57.29578f);
    float eyeRatio = eye / std::max(1.f, f.box.w);
    float yawEye=(nose.x-eyeMidX)/std::max(1.f,eye);
    float yawMouth=(nose.x-mouthMidX)/std::max(1.f,mouth);
    float pitch=eyeNose/std::max(1.f,noseMouth);
    float faceScale=f.box.w/std::max(1.f,(float)frameWidth);
    result.pitchRatio=pitch;
    result.signedYawEye=yawEye;
    result.signedYawMouth=yawMouth;
    result.faceScale=faceScale;
    result.rollDegrees=roll;
    result.geometryValid=roll<22.f&&eyeRatio>.20f&&eyeRatio<.70f&&
                         pitch>.58f&&pitch<1.58f&&f.score>=.48f;
    if(!result.geometryValid||!calibration.enabled)return result;

    // v1.20.5: target-band instead of "any turn toward the screen side".
    // The camera is physically left of the display. The centre of the ad is
    // represented by the empirically stable five-landmark pose used by the
    // lateral mount. A bounded residual rejects both direct camera looks and
    // people who simply turn too far to the side.
    float targetSign=calibration.eyeIntercept<0.f?-1.f:1.f;
    float expectedEye=targetSign*.12f;
    float expectedMouth=targetSign*.16f;
    float residualEye=std::fabs(yawEye-expectedEye);
    float residualMouth=std::fabs(yawMouth-expectedMouth);
    result.expectedYawEye=expectedEye;
    result.expectedYawMouth=expectedMouth;
    result.targetResidual=std::max(residualEye/.085f,residualMouth/.120f);

    // Keep a strict vertical head-pose gate. This runs only for impression
    // confirmation; gender/tracking continue to use the broader geometryValid.
    bool verticalTarget=pitch>.75f&&pitch<1.25f;
    if(!verticalTarget)return result;

    // Precision is intentionally asymmetric: at the far edge where the camera
    // and screen become visually indistinguishable, the observation is rejected
    // rather than treating a lens look as an ad impression.
    if(residualEye>.085f||residualMouth>.120f)return result;

    result.score=1.f;
    result.accepted=true;
    return result;
}

static bool align128(const YuvFrame& frame, const Face& f, std::vector<uint8_t>& rgb) {
    static const float ref112[10] = {38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f,
                                      71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f};
    float mx=0,my=0,mu=0,mv=0;
    for (int i=0;i<5;++i) { mx+=f.lm[i].x; my+=f.lm[i].y; mu+=ref112[i*2]*128.f/112.f; mv+=ref112[i*2+1]*128.f/112.f; }
    mx/=5; my/=5; mu/=5; mv/=5;
    float numA=0,numB=0,den=0;
    for (int i=0;i<5;++i) {
        float x=f.lm[i].x-mx,y=f.lm[i].y-my,u=ref112[i*2]*128.f/112.f-mu,v=ref112[i*2+1]*128.f/112.f-mv;
        numA += x*u + y*v; numB += x*v - y*u; den += x*x + y*y;
    }
    if (den < 4) return false;
    float a=numA/den,b=numB/den,tx=mu-a*mx+b*my,ty=mv-b*mx-a*my;
    float inv=a*a+b*b; if (inv < 1e-6f) return false;
    rgb.resize(128*128*3);
    for (int y=0;y<128;++y) for(int x=0;x<128;++x) {
        float u=x+.5f-tx,v=y+.5f-ty;
        float sx=(a*u+b*v)/inv, sy=(-b*u+a*v)/inv;
        frame.rgbBilinear(sx, sy, &rgb[((size_t)y*128+x)*3]);
    }
    return true;
}

// Lightweight crop quality estimate. It prevents dark, blown-out or severely
// blurred autofocus transitions from entering the temporal gender vote.
static float cropVisualQuality(const std::vector<uint8_t>& rgb) {
    if(rgb.size()<128u*128u*3u)return 0.f;
    double sum=0,squares=0,gradients=0;int count=0,gradientCount=0;
    auto luminance=[&](int x,int y){
        const uint8_t* p=&rgb[((size_t)y*128+x)*3];
        return (77*(int)p[0]+150*(int)p[1]+29*(int)p[2])>>8;
    };
    for(int y=2;y<126;y+=2)for(int x=2;x<126;x+=2){
        int l=luminance(x,y);sum+=l;squares+=(double)l*l;++count;
        gradients+=std::abs(l-luminance(x-2,y))+std::abs(l-luminance(x,y-2));
        gradientCount+=2;
    }
    if(!count||!gradientCount)return 0.f;
    float mean=(float)(sum/count);
    float variance=(float)std::max(0.0,squares/count-(double)mean*mean);
    float deviation=std::sqrt(variance);
    float gradient=(float)(gradients/gradientCount);
    if(mean<24.f||mean>235.f||deviation<10.f||gradient<2.5f)return 0.f;
    float exposure=clampf(1.f-std::fabs(mean-128.f)/108.f,.12f,1.f);
    float contrast=clampf(deviation/46.f,.18f,1.f);
    float sharpness=clampf(gradient/20.f,.15f,1.f);
    return exposure*(.35f+.65f*contrast)*(.30f+.70f*sharpness);
}


// Quality gate dedicated to gender sampling. Do not reuse the legacy
// attention/gaze geometry here: apparent gender is independent of where the
// person is looking. This gate only rejects crops whose 5-point geometry is
// too distorted to align reliably.
static float genderPoseQuality(const Face& f) {
    const Point &le=f.lm[0], &re=f.lm[1], &nose=f.lm[2], &lm=f.lm[3], &rm=f.lm[4];
    float eye=std::hypot(re.x-le.x,re.y-le.y);
    float mouth=std::hypot(rm.x-lm.x,rm.y-lm.y);
    if(eye<12.f||mouth<6.f||re.x<=le.x||rm.x<=lm.x)return 0.f;
    float eyeMidX=(le.x+re.x)*.5f,eyeMidY=(le.y+re.y)*.5f;
    float mouthMidX=(lm.x+rm.x)*.5f,mouthMidY=(lm.y+rm.y)*.5f;
    float upper=nose.y-eyeMidY,lower=mouthMidY-nose.y;
    if(upper<=eye*.08f||lower<=eye*.08f)return 0.f;
    float roll=std::fabs(std::atan2(re.y-le.y,re.x-le.x)*57.29578f);
    float yawEye=std::fabs(nose.x-eyeMidX)/std::max(1.f,eye);
    float yawMouth=std::fabs(nose.x-mouthMidX)/std::max(1.f,mouth);
    float pitch=upper/std::max(1.f,lower);
    if(roll>30.f||yawEye>.42f||yawMouth>.52f||pitch<.48f||pitch>1.85f)return 0.f;
    float qRoll=clampf(1.f-roll/28.f,.12f,1.f);
    float qYaw=clampf(1.f-std::max(yawEye/.38f,yawMouth/.48f),.12f,1.f);
    float qPitch=clampf(1.f-std::fabs(std::log(pitch/.98f))/.75f,.15f,1.f);
    return qRoll*qYaw*qPitch;
}

class GenderNet {
public:
    bool load(AAssetManager* assets) {
        net.opt.num_threads = 1; net.opt.use_vulkan_compute = false;
        net.opt.use_int8_inference = true; net.opt.use_fp16_storage = false;
        return net.load_param(assets, "fastface_mnv3_large_128_int8.param") == 0 &&
               net.load_model(assets, "fastface_mnv3_large_128_int8.bin") == 0;
    }
    float maleProbability(const std::vector<uint8_t>& rgb) {
        return maleProbabilityRaw(rgb);
    }
    // Same FastFace MobileNetV3 INT8. Only uncertain crops receive a mirrored
    // second opinion; strong self-disagreement abstains instead of forcing a label.
    float maleProbabilityStable(const std::vector<uint8_t>& rgb, float* flipAgreement=nullptr) {
        float p0=maleProbabilityRaw(rgb);
        if(p0<0.f){if(flipAgreement)*flipAgreement=0.f;return p0;}
        if(p0<=.26f||p0>=.74f){if(flipAgreement)*flipAgreement=1.f;return p0;}
        mirrorScratch.resize(rgb.size());
        for(int y=0;y<128;++y)for(int x=0;x<128;++x){
            const uint8_t* src=&rgb[((size_t)y*128+x)*3];
            uint8_t* dst=&mirrorScratch[((size_t)y*128+(127-x))*3];
            dst[0]=src[0];dst[1]=src[1];dst[2]=src[2];
        }
        float p1=maleProbabilityRaw(mirrorScratch);
        if(p1<0.f){if(flipAgreement)*flipAgreement=0.f;return p0;}
        float delta=std::fabs(p0-p1);
        float agreement=1.f-clampf(delta/.30f,0.f,1.f);
        if(flipAgreement)*flipAgreement=agreement;
        if(delta>.28f)return -.2f;
        return .5f*(p0+p1);
    }
private:
    float maleProbabilityRaw(const std::vector<uint8_t>& rgb) {
        ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data(), ncnn::Mat::PIXEL_RGB, 128, 128);
        const float mean[3] = {123.675f,116.28f,103.53f};
        const float norm[3] = {.0171247538f,.0175070028f,.0174291939f};
        in.substract_mean_normalize(mean,norm);
        ncnn::Extractor ex=net.create_extractor(); ex.set_light_mode(true); ex.input("in0",in);
        ncnn::Mat out; if(ex.extract("out0",out)!=0 || out.total()<2) return -.1f;
        float m=std::max(out[0],out[1]),e0=std::exp(out[0]-m),e1=std::exp(out[1]-m);
        return e1/(e0+e1);
    }
    ncnn::Net net;
    std::vector<uint8_t> mirrorScratch;
};

struct PersonDetection { Rect box; float score=0; };

class PersonDetector {
public:
    bool load(AAssetManager* assets) {
        net.opt.num_threads = 2;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = false;
        return net.load_param(assets, "nanodet_m_320.param") == 0 &&
               net.load_model(assets, "nanodet_m_320.bin") == 0;
    }

    std::vector<PersonDetection> detect(const YuvFrame& frame) {
        int sw=0, sh=0;
        // A 720x1280 logical frame would become only 180 pixels wide at 320.
        // Preserve enough horizontal face/body detail for the portrait EMEET
        // while keeping the NanoDet graph and output decoding unchanged.
        frame.resizeRgb(320, rgbScratch, sw, sh);
        ncnn::Mat input=ncnn::Mat::from_pixels(rgbScratch.data(),ncnn::Mat::PIXEL_RGB2BGR,sw,sh);
        int wp=(sw+31)/32*32-sw,hp=(sh+31)/32*32-sh;
        ncnn::Mat padded;
        ncnn::copy_make_border(input,padded,hp/2,hp-hp/2,wp/2,wp-wp/2,ncnn::BORDER_CONSTANT,0.f);
        const float mean[3]={103.53f,116.28f,123.675f};
        const float norm[3]={1.f/57.375f,1.f/57.12f,1.f/58.395f};
        padded.substract_mean_normalize(mean,norm);
        ncnn::Extractor ex=net.create_extractor();
        ex.input("input.1",padded);
        std::vector<PersonDetection> proposals; proposals.reserve(96);
        for(int stride:{8,16,32}) {
            ncnn::Mat cls,dis; std::string suffix=std::to_string(stride);
            if(ex.extract(("cls_pred_stride_"+suffix).c_str(),cls)!=0)continue;
            if(ex.extract(("dis_pred_stride_"+suffix).c_str(),dis)!=0)continue;
            int nx=padded.w/stride,ny=cls.h/std::max(1,nx),bins=dis.w/4;
            for(int iy=0;iy<ny;++iy)for(int ix=0;ix<nx;++ix){
                int idx=iy*nx+ix;float score=cls.row(idx)[0];
                if(score<.36f)continue;
                float d[4]={};const float* row=dis.row(idx);
                for(int k=0;k<4;++k){
                    const float* q=row+k*bins;float m=q[0];
                    for(int z=1;z<bins;++z)m=std::max(m,q[z]);
                    float sum=0,num=0;
                    for(int z=0;z<bins;++z){float e=std::exp(q[z]-m);sum+=e;num+=z*e;}
                    d[k]=(sum>0?num/sum:0)*stride;
                }
                float cx=(ix+.5f)*stride,cy=(iy+.5f)*stride;
                float sx=(float)frame.width/sw,sy=(float)frame.height/sh;
                float x0=(cx-d[0]-wp/2.f)*sx,y0=(cy-d[1]-hp/2.f)*sy;
                float x1=(cx+d[2]-wp/2.f)*sx,y1=(cy+d[3]-hp/2.f)*sy;
                x0=clampf(x0,0,frame.width-1.f);y0=clampf(y0,0,frame.height-1.f);
                x1=clampf(x1,0,frame.width-1.f);y1=clampf(y1,0,frame.height-1.f);
                if(x1-x0>=18 && y1-y0>=35)proposals.push_back({{x0,y0,x1-x0,y1-y0},score});
            }
        }
        std::sort(proposals.begin(),proposals.end(),[](const PersonDetection&a,const PersonDetection&b){return a.score>b.score;});
        std::vector<PersonDetection> result; result.reserve(12);
        for(const auto& p:proposals){
            // NanoDet occasionally emits two partially overlapping person boxes for one body.
            // A slightly stricter NMS prevents both boxes from becoming independent sessions.
            bool keep=true;for(const auto& old:result)if(iou(p.box,old.box)>.38f){keep=false;break;}
            if(keep)result.push_back(p);if(result.size()>=12)break;
        }
        return result;
    }
private:
    ncnn::Net net;
    std::vector<uint8_t> rgbScratch;
};

struct WeightedGender { float probability=0, weight=0; };

struct PersonTrack {
    int id=0;
    Rect body;
    float bodyScore=0, vx=0, vy=0;
    Face face;
    bool hasFace=false, bodyMatched=false, faceMatched=false;
    int bodyHits=0, faceHits=0, frontalHits=0, targetHits=0;
    int64_t firstBody=0,firstFace=0,lastBody=0,lastFace=0,lastGender=0,frontalStart=0;
    int64_t attentionTs=0,targetWindowStart=0,targetLast=0;
    float attentionScore=0,attentionSum=0;
    bool reachCounted=false,impressionCounted=false,newImpression=false;
    bool motionConfirmed=false,nearEdge=false,exitLikely=false;
    float originBodyCx=0,originBodyCy=0,originBodyArea=0;
    float bodyTravel=0,maxBodyShift=0,maxBodyScale=0;
    int eventFlags=0;
    // Transient result for this exact input frame. It is reset before each
    // process() call and exported so Android can preserve precisely the
    // accepted images used by the temporal impression decision.
    int attentionEvaluation=0;
    bool attentionGeometryValid=false;
    float signedYawEye=0,signedYawMouth=0,faceScale=0,pitchRatio=0,rollDegrees=0;
    float expectedYawEye=0,expectedYawMouth=0,targetResidual=9;
    int gender=-1, attributedGender=-1;
    float genderConfidence=0,attributedQuality=0,maxGenderQuality=0;
    std::vector<WeightedGender> samples;
};

class Engine {
public:
    bool load(AAssetManager* assets) {
        ncnn::set_cpu_powersave(0); ncnn::set_omp_num_threads(2);
        tracks.reserve(24);
        genderAlignedScratch.reserve(128*128*3);
        ready = detector.load(assets) && people.load(assets) && gender.load(assets);
        return ready;
    }

    void setAttentionCalibration(bool enabled, float eyeIntercept, float eyeSlope,
                                 float mouthIntercept, float mouthSlope,
                                 float eyeTolerance, float mouthTolerance) {
        std::lock_guard<std::mutex> guard(lock);
        attentionCalibration.enabled=enabled;
        attentionCalibration.eyeIntercept=clampf(eyeIntercept,-1.f,1.f);
        attentionCalibration.eyeSlope=clampf(eyeSlope,-12.f,12.f);
        attentionCalibration.mouthIntercept=clampf(mouthIntercept,-1.f,1.f);
        attentionCalibration.mouthSlope=clampf(mouthSlope,-12.f,12.f);
        attentionCalibration.eyeTolerance=clampf(eyeTolerance,.075f,.18f);
        attentionCalibration.mouthTolerance=clampf(mouthTolerance,.10f,.24f);
        // A target change invalidates only the temporal attention streak. Face
        // tracking and the proven gender vote remain untouched.
        for(auto& t:tracks){
            t.frontalHits=0;t.frontalStart=0;t.attentionSum=0;
            t.targetHits=0;t.targetWindowStart=0;t.targetLast=0;
        }
    }

    std::vector<float> process(const YuvFrame& frame, int64_t timestamp) {
        std::lock_guard<std::mutex> guard(lock);
        if (!ready) return {0,(float)frame.width,(float)frame.height,0,0,0};
        auto started=std::chrono::steady_clock::now();
        ++frameNo;
        // Preserve the validated scan hierarchy while adapting cadence when
        // actual ARMv7 frame cost rises, so detector bursts do not starve preview.
        int lightEvery=3,mediumEvery=18,maxEvery=54,bodyEvery=12;
        if(emaProcessMs>72.f){lightEvery=5;mediumEvery=30;maxEvery=90;bodyEvery=15;}
        else if(emaProcessMs>48.f){lightEvery=4;mediumEvery=24;maxEvery=72;bodyEvery=12;}
        int mode=(frameNo==1||frameNo%maxEvery==0)?3:
                 (frameNo%mediumEvery==0?2:(frameNo%lightEvery==0?1:0));
        bool bodyScan=(frameNo==1||frameNo%bodyEvery==0);
        for(auto& t:tracks){t.newImpression=false;t.eventFlags=0;t.attentionEvaluation=0;t.bodyMatched=false;t.faceMatched=false;}
        if(bodyScan){associateBodies(people.detect(frame),frame,timestamp);mergeDuplicateTracks(timestamp);}
        if(mode){
            int target;
            if(frame.height>frame.width)
                target=mode==3?896:(mode==2?576:320);
            else
                target=mode==3?1024:(mode==2?640:320);
            float threshold=mode==3?.44f:(mode==2?.49f:.56f);
            associateFaces(detector.detect(frame,target,threshold),frame,timestamp);
            mergeDuplicateTracks(timestamp);
        }
        tracks.erase(std::remove_if(tracks.begin(),tracks.end(),[&](const PersonTrack& t){
            int64_t unseen=timestamp-std::max(t.lastBody,t.lastFace);
            int64_t limit=(t.exitLikely&&t.motionConfirmed)?1200000000LL:
                          ((t.reachCounted||t.impressionCounted)?6000000000LL:3000000000LL);
            return unseen>limit;
        }),tracks.end());
        int visible=0;
        for(const auto& t:tracks)if(shouldOutput(t,timestamp)||t.newImpression||t.eventFlags)++visible;
        std::vector<float> out; out.reserve(6+(size_t)visible*24);
        out={1,(float)frame.width,(float)frame.height,(float)mode,(float)visible,0};
        for(auto& t:tracks){
            if(!shouldOutput(t,timestamp)&&!t.newImpression&&!t.eventFlags)continue;
            bool freshFace=t.hasFace&&timestamp-t.lastFace<750000000LL;
            const Rect& box=freshFace?t.face.box:t.body;
            float detectorScore=freshFace?t.face.score:-std::max(.01f,t.bodyScore);
            bool attentive=freshFace&&timestamp-t.attentionTs<650000000LL&&
                           t.attentionEvaluation>0&&t.attentionScore>=.66f;
            out.insert(out.end(),{(float)t.id,box.x,box.y,box.w,box.h,(float)t.gender,
                                  t.genderConfidence,attentive?1.f:0.f,
                                  t.newImpression?1.f:0.f,detectorScore,(float)t.samples.size(),(float)t.eventFlags,
                                  (float)t.attentionEvaluation,t.attentionScore,(float)t.frontalHits,
                                  t.signedYawEye,t.signedYawMouth,t.faceScale,t.pitchRatio,t.rollDegrees,
                                  t.expectedYawEye,t.expectedYawMouth,t.targetResidual,
                                  t.attentionGeometryValid?1.f:0.f});
        }
        out[5]=(float)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count();
        emaProcessMs=.88f*emaProcessMs+.12f*out[5];
        return out;
    }
private:
    static bool isVisible(const PersonTrack& t,int64_t ts){return ts-std::max(t.lastBody,t.lastFace)<850000000LL;}
    static bool shouldOutput(const PersonTrack& t,int64_t ts){
        bool freshFace=t.hasFace&&ts-t.lastFace<750000000LL;
        return isVisible(t,ts)&&(t.reachCounted||freshFace);
    }

    static Rect syntheticBody(const Face& f,const YuvFrame& frame){
        Rect r{f.box.x-1.4f*f.box.w,f.box.y-.35f*f.box.h,3.8f*f.box.w,6.3f*f.box.h};
        r.x=clampf(r.x,0,frame.width-1.f);r.y=clampf(r.y,0,frame.height-1.f);
        r.w=clampf(r.w,1,frame.width-r.x);r.h=clampf(r.h,1,frame.height-r.y);return r;
    }

    static void ensureReach(PersonTrack& t){if(!t.reachCounted){t.reachCounted=true;t.eventFlags|=1;}}

    static void initBodyMotion(PersonTrack& t,const Rect& body,const YuvFrame& frame,int64_t ts){
        t.firstBody=ts;
        t.originBodyCx=body.x+body.w*.5f;t.originBodyCy=body.y+body.h*.5f;
        t.originBodyArea=std::max(1.f,area(body));
        t.nearEdge=body.x<frame.width*.045f||body.y<frame.height*.045f||
                   body.x+body.w>frame.width*.955f||body.y+body.h>frame.height*.955f;
    }

    static void updateBodyMotion(PersonTrack& t,const Rect& detected,const YuvFrame& frame,int64_t ts){
        if(!t.firstBody)initBodyMotion(t,t.body,frame,ts);
        float diag=std::max(1.f,std::hypot((float)frame.width,(float)frame.height));
        float oldCx=t.body.x+t.body.w*.5f,oldCy=t.body.y+t.body.h*.5f;
        float newCx=detected.x+detected.w*.5f,newCy=detected.y+detected.h*.5f;
        float step=std::hypot(newCx-oldCx,newCy-oldCy)/diag;
        // Ignore the normal 3-5 pixel detector jitter of a static rectangle.
        if(step>.0035f)t.bodyTravel+=step-.0035f;
        t.maxBodyShift=std::max(t.maxBodyShift,
                std::hypot(newCx-t.originBodyCx,newCy-t.originBodyCy)/diag);
        t.maxBodyScale=std::max(t.maxBodyScale,
                std::fabs(std::log(std::max(1.f,area(detected))/std::max(1.f,t.originBodyArea))));
        t.nearEdge=detected.x<frame.width*.045f||detected.y<frame.height*.045f||
                   detected.x+detected.w>frame.width*.955f||detected.y+detected.h>frame.height*.955f;
        bool outward=(newCx<frame.width*.09f&&newCx<oldCx-2.f)||
                     (newCx>frame.width*.91f&&newCx>oldCx+2.f)||
                     (newCy<frame.height*.09f&&newCy<oldCy-2.f)||
                     (newCy>frame.height*.91f&&newCy>oldCy+2.f);
        if(t.nearEdge&&outward)t.exitLikely=true;
        else if(!t.nearEdge)t.exitLikely=false;
        if(t.bodyHits>=3&&ts-t.firstBody>=220000000LL&&
           (t.maxBodyShift>=.014f||t.bodyTravel>=.022f||t.maxBodyScale>=.12f)){
            t.motionConfirmed=true;ensureReach(t);
        }
    }

    void associateBodies(const std::vector<PersonDetection>& detections,const YuvFrame& frame,int64_t ts){
        std::vector<char> used(detections.size(),0);
        for(auto& t:tracks){
            int best=-1;float bestScore=.14f;
            for(int i=0;i<(int)detections.size();++i) if(!used[i]){
                Rect predicted=t.body;predicted.x+=t.vx;predicted.y+=t.vy;
                float ov=iou(predicted,detections[i].box);
                float cx=predicted.x+predicted.w*.5f,cy=predicted.y+predicted.h*.5f;
                float dx=detections[i].box.x+detections[i].box.w*.5f-cx;
                float dy=detections[i].box.y+detections[i].box.h*.5f-cy;
                float dist=std::hypot(dx,dy)/std::max(35.f,std::max(predicted.w,predicted.h)*.55f);
                float score=ov+(dist<.90f?.38f:(dist<1.25f?.17f:0.f));
                if(score>bestScore){bestScore=score;best=i;}
            }
            if(best>=0){
                const auto& d=detections[best];used[best]=1;t.bodyMatched=true;
                ++t.bodyHits;updateBodyMotion(t,d.box,frame,ts);
                float oldx=t.body.x,oldy=t.body.y,a=.72f;
                t.body.x=a*d.box.x+(1-a)*t.body.x;t.body.y=a*d.box.y+(1-a)*t.body.y;
                t.body.w=a*d.box.w+(1-a)*t.body.w;t.body.h=a*d.box.h+(1-a)*t.body.h;
                t.vx=t.body.x-oldx;t.vy=t.body.y-oldy;t.bodyScore=d.score;t.lastBody=ts;
            }
        }
        for(int i=0;i<(int)detections.size();++i) if(!used[i]){
            PersonTrack t;t.id=nextId++;t.body=detections[i].box;t.bodyScore=detections[i].score;
            t.lastBody=ts;t.bodyMatched=true;t.bodyHits=1;initBodyMotion(t,t.body,frame,ts);tracks.push_back(t);
        }
    }

    void associateFaces(const std::vector<Face>& detections,const YuvFrame& frame,int64_t ts){
        std::vector<char> assigned(tracks.size(),0);
        for(const Face& d:detections){
            int best=-1;float bestScore=.05f;
            float fx=d.box.x+d.box.w*.5f,fy=d.box.y+d.box.h*.5f;
            for(int j=0;j<(int)tracks.size();++j)if(!assigned[j]){
                auto& t=tracks[j];float score=0;
                Rect expanded{t.body.x-t.body.w*.22f,t.body.y-t.body.h*.20f,t.body.w*1.44f,t.body.h*1.30f};
                bool inside=fx>=expanded.x&&fx<=expanded.x+expanded.w&&fy>=expanded.y&&fy<=expanded.y+expanded.h*.72f;
                if(inside){
                    float hx=t.body.x+t.body.w*.5f,hy=t.body.y+t.body.h*.18f;
                    float dist=std::hypot(fx-hx,fy-hy)/std::max(25.f,t.body.w);
                    score=1.2f-std::min(1.f,dist);
                }
                if(t.hasFace&&ts-t.lastFace<6000000000LL){
                    float ov=iou(t.face.box,d.box);
                    float tcx=t.face.box.x+t.face.box.w*.5f,tcy=t.face.box.y+t.face.box.h*.5f;
                    float fd=std::hypot(fx-tcx,fy-tcy)/std::max(18.f,std::max(t.face.box.w,d.box.w));
                    score=std::max(score,ov*1.8f);
                    if(fd<1.35f)score=std::max(score,.78f-fd*.28f);
                }
                if(score>bestScore){bestScore=score;best=j;}
            }
            if(best<0){
                PersonTrack t;t.id=nextId++;t.body=syntheticBody(d,frame);t.face=d;t.hasFace=true;
                t.firstFace=ts;t.lastFace=ts;t.faceHits=1;t.faceMatched=true;
                initBodyMotion(t,t.body,frame,ts);tracks.push_back(t);assigned.push_back(1);
                updateFaceState(frame,tracks.back(),d,ts);continue;
            }
            assigned[best]=1;auto& t=tracks[best];t.faceMatched=true;
            if(!t.firstFace)t.firstFace=ts;
            ++t.faceHits;
            if(!t.hasFace)t.face=d;
            else{
                float a=.73f;t.face.box.x=a*d.box.x+(1-a)*t.face.box.x;t.face.box.y=a*d.box.y+(1-a)*t.face.box.y;
                t.face.box.w=a*d.box.w+(1-a)*t.face.box.w;t.face.box.h=a*d.box.h+(1-a)*t.face.box.h;
                for(int p=0;p<5;++p){t.face.lm[p].x=a*d.lm[p].x+(1-a)*t.face.lm[p].x;t.face.lm[p].y=a*d.lm[p].y+(1-a)*t.face.lm[p].y;}
                t.face.score=d.score;
            }
            t.hasFace=true;t.lastFace=ts;
            if(t.faceHits>=2&&ts-t.firstFace>=120000000LL)ensureReach(t);
            // The raw detection belongs to this exact YUV frame. The smoothed
            // track above is intentionally restricted to drawing/association.
            updateFaceState(frame,t,d,ts);
        }
    }

    void updateFaceState(const YuvFrame& frame,PersonTrack& t,const Face& observed,int64_t ts){
        AttentionResult attention=evaluateAttention(observed,frame.width,attentionCalibration);
        t.attentionTs=ts;t.attentionScore=attention.score;
        t.attentionGeometryValid=attention.geometryValid;
        t.signedYawEye=attention.signedYawEye;t.signedYawMouth=attention.signedYawMouth;
        t.faceScale=attention.faceScale;t.pitchRatio=attention.pitchRatio;
        t.rollDegrees=attention.rollDegrees;t.expectedYawEye=attention.expectedYawEye;
        t.expectedYawMouth=attention.expectedYawMouth;t.targetResidual=attention.targetResidual;
        // Every visible face is a pseudo-impression. frontalHits now records
        // how many analyzable face frames belonged to the track; it is never
        // reset by a negative direction result.
        if(!t.frontalStart)t.frontalStart=ts;
        ++t.frontalHits;
        t.attentionSum=std::max(t.attentionSum,attention.score);
        bool targetCandidate=attention.accepted&&!t.impressionCounted&&observed.score>=.50f;
        if(targetCandidate){
            // Two strict observations inside a short rolling window reject a
            // one-frame landmark glitch while keeping an ordinary glance fast.
            if(!t.targetLast||ts-t.targetLast>900000000LL){
                t.targetHits=0;t.targetWindowStart=ts;
            }
            t.targetLast=ts;
            ++t.targetHits;
        }
        bool winningFrame=targetCandidate;
        // Only the confirmed frame requests JPEG compression. Earlier
        // pseudo/candidate frames remain numeric, preserving ARM32 throughput.
        t.attentionEvaluation=winningFrame?1:(attention.geometryValid?-1:0);
        // Gender is a property of the face crop, not of whether the person is
        // currently looking at the advertisement. Collect its robust vote from
        // every geometrically usable portrait crop, then attribute it only if
        // this same track later becomes a real impression.
        float relative=observed.box.w/std::max(1.f,(float)frame.width);
        float facePixels=std::min(observed.box.w,observed.box.h);
        float poseQuality=genderPoseQuality(observed);
        bool room=t.samples.size()<14;
        bool settled=t.gender>=0&&t.samples.size()>=8&&t.genderConfidence>=.90f;
        int64_t genderInterval=settled?1200000000LL:420000000LL;
        // Same MobileNetV3 INT8 and same 5-point alignment. The improvement is
        // entirely in how it is used: only clear, sufficiently large,
        // independently pose-valid crops enter the vote. This avoids the old
        // accidental dependency on attention.geometryValid and prevents tiny
        // far-away faces/autofocus transitions from biasing the public label.
        if(relative>=.035f&&facePixels>=46.f&&observed.score>=.58f&&
           poseQuality>=.16f&&ts-t.lastGender>=genderInterval){
            auto& aligned=genderAlignedScratch;
            if(align128(frame,observed,aligned)){
                float visual=cropVisualQuality(aligned);
                float pixelQuality=clampf((facePixels-42.f)/86.f,.12f,1.20f);
                float sizeQuality=clampf((relative-.025f)/.080f,.20f,1.18f);
                float quality=pixelQuality*sizeQuality*poseQuality*
                              clampf(observed.score,.55f,1.f)*visual;
                if(!room){
                    float minw=10.f;
                    for(const auto& sample:t.samples)minw=std::min(minw,sample.weight);
                    room=quality>minw*1.12f;
                }
                if(visual>=.14f&&quality>=.020f&&room){
                    float flipAgreement=1.f;
                    float p=gender.maleProbabilityStable(aligned,&flipAgreement);
                    t.lastGender=ts;
                    if(p>=0.f&&flipAgreement>=.22f){
                        quality*=.72f+.28f*flipAgreement;
                        addGenderSample(t,{p,quality});lockVote(t);
                    }
                } else {
                    // Avoid retrying the same low-quality crop on every face scan.
                    t.lastGender=ts;
                }
            }
        }
        // v1.20.7: first strict target frame wins immediately and is the JPEG evidence.
        if(winningFrame){
            t.impressionCounted=true;t.newImpression=true;
        }
        updateAttribution(t);
    }

    static void addGenderSample(PersonTrack& t,WeightedGender sample){
        t.maxGenderQuality=std::max(t.maxGenderQuality,sample.weight);
        if(t.samples.size()<14){t.samples.push_back(sample);return;}
        auto it=std::min_element(t.samples.begin(),t.samples.end(),[](const WeightedGender&a,const WeightedGender&b){return a.weight<b.weight;});
        if(it!=t.samples.end()&&sample.weight>it->weight)*it=sample;
    }

    static void lockVote(PersonTrack& t){
        if(t.samples.size()<4)return;
        float bestWeight=0;for(const auto&s:t.samples)bestWeight=std::max(bestWeight,s.weight);
        std::vector<WeightedGender> good;
        for(const auto&s:t.samples)if(s.weight>=bestWeight*.52f)good.push_back(s);
        if(good.size()<4)good=t.samples;
        std::sort(good.begin(),good.end(),[](const WeightedGender&a,const WeightedGender&b){
            return a.probability<b.probability;
        });
        float weighted=0,weights=0;
        for(const auto&s:good){weighted+=s.probability*s.weight;weights+=s.weight;}
        if(weights<=0)return;
        float half=weights*.5f,walk=0,median=good.back().probability;
        for(const auto&s:good){walk+=s.weight;if(walk>=half){median=s.probability;break;}}
        float mean=weighted/weights;
        float robust=.55f*median+.45f*mean;
        int candidate=robust>=.5f?1:0;
        float agreeWeight=0;
        for(const auto&s:good)if((s.probability>=.5f)==(candidate==1))agreeWeight+=s.weight;
        float agreement=agreeWeight/weights;
        float confidence=std::max(robust,1.f-robust);
        int n=(int)good.size();
        // v1.20.2: prefer Indeterminado over a wrong public label. The model
        // and crops are unchanged; only a stronger temporal consensus can
        // publish Masculino/Feminino.
        bool ready=(n>=5&&confidence>=.91f&&agreement>=.92f)||
                   (n>=8&&confidence>=.84f&&agreement>=.86f)||
                   (n>=11&&confidence>=.79f&&agreement>=.84f);
        if(!ready)return;
        if(t.gender<0){t.gender=candidate;t.genderConfidence=confidence;return;}
        if(candidate==t.gender){t.genderConfidence=std::max(t.genderConfidence,confidence);return;}
        // Hysteresis keeps the public preview stable. An opposite label needs
        // substantially stronger, longer evidence before replacing it.
        if(n>=11&&confidence>=.94f&&agreement>=.94f){
            t.gender=candidate;t.genderConfidence=confidence;
        }
    }

    static void updateAttribution(PersonTrack& t){
        if(!t.impressionCounted||t.gender<0)return;
        if(t.attributedGender<0&&t.genderConfidence>=.68f){
            t.attributedGender=t.gender;t.attributedQuality=t.maxGenderQuality;t.eventFlags|=2;return;
        }
        if(t.attributedGender!=t.gender&&t.genderConfidence>=.84f&&t.maxGenderQuality>t.attributedQuality*1.18f){
            t.attributedGender=t.gender;t.attributedQuality=t.maxGenderQuality;t.eventFlags|=(t.gender==1?4:8);
        }
    }

    static float centerDistance(const Rect& a,const Rect& b){
        float ax=a.x+a.w*.5f,ay=a.y+a.h*.5f,bx=b.x+b.w*.5f,by=b.y+b.h*.5f;
        return std::hypot(ax-bx,ay-by)/std::max(20.f,std::max({a.w,a.h,b.w,b.h}));
    }

    static bool duplicateTracks(const PersonTrack& a,const PersonTrack& b,int64_t ts){
        bool af=a.hasFace&&ts-a.lastFace<1200000000LL;
        bool bf=b.hasFace&&ts-b.lastFace<1200000000LL;
        if(af&&bf){
            float scale=std::max(18.f,std::max(a.face.box.w,b.face.box.w));
            float ax=a.face.box.x+a.face.box.w*.5f,ay=a.face.box.y+a.face.box.h*.5f;
            float bx=b.face.box.x+b.face.box.w*.5f,by=b.face.box.y+b.face.box.h*.5f;
            float fd=std::hypot(ax-bx,ay-by)/scale;
            return iou(a.face.box,b.face.box)>.18f||fd<.48f;
        }
        // Never fuse two simultaneously visible, distinct faces only because their bodies cross.
        if(af!=bf){
            const PersonTrack& ft=af?a:b;const PersonTrack& bt=af?b:a;
            float fx=ft.face.box.x+ft.face.box.w*.5f,fy=ft.face.box.y+ft.face.box.h*.5f;
            Rect upper{bt.body.x-bt.body.w*.16f,bt.body.y-bt.body.h*.12f,bt.body.w*1.32f,bt.body.h*.78f};
            bool headInside=fx>=upper.x&&fx<=upper.x+upper.w&&fy>=upper.y&&fy<=upper.y+upper.h;
            if(headInside&&iou(a.body,b.body)>.16f)return true;
            if(iou(a.body,b.body)<.62f)return false;
        }
        return iou(a.body,b.body)>.52f||(iou(a.body,b.body)>.34f&&centerDistance(a.body,b.body)<.23f);
    }

    static void mergeInto(PersonTrack& a,const PersonTrack& b){
        bool oldReach=(a.reachCounted&&(a.eventFlags&1)==0)||(b.reachCounted&&(b.eventFlags&1)==0);
        bool oldImpression=(a.impressionCounted&&!a.newImpression)||(b.impressionCounted&&!b.newImpression);
        bool oldAttribution=(a.attributedGender>=0&&(a.eventFlags&2)==0)||
                            (b.attributedGender>=0&&(b.eventFlags&2)==0);
        if(b.lastBody>a.lastBody){a.body=b.body;a.bodyScore=b.bodyScore;a.vx=b.vx;a.vy=b.vy;}
        if(b.lastFace>a.lastFace){a.face=b.face;a.hasFace=b.hasFace;}
        a.id=std::min(a.id,b.id);a.firstBody=a.firstBody?std::min(a.firstBody,b.firstBody):b.firstBody;
        a.firstFace=a.firstFace?std::min(a.firstFace,b.firstFace):b.firstFace;
        a.lastBody=std::max(a.lastBody,b.lastBody);a.lastFace=std::max(a.lastFace,b.lastFace);
        a.lastGender=std::max(a.lastGender,b.lastGender);a.bodyHits=std::max(a.bodyHits,b.bodyHits);
        a.faceHits=std::max(a.faceHits,b.faceHits);
        if(b.attentionTs>a.attentionTs){
            a.attentionTs=b.attentionTs;a.attentionScore=b.attentionScore;
            a.frontalHits=b.frontalHits;a.frontalStart=b.frontalStart;
            a.attentionSum=b.attentionSum;
            a.attentionGeometryValid=b.attentionGeometryValid;
            a.signedYawEye=b.signedYawEye;a.signedYawMouth=b.signedYawMouth;
            a.faceScale=b.faceScale;a.pitchRatio=b.pitchRatio;a.rollDegrees=b.rollDegrees;
            a.expectedYawEye=b.expectedYawEye;a.expectedYawMouth=b.expectedYawMouth;
            a.targetResidual=b.targetResidual;
        }
        if(b.targetLast>a.targetLast){
            a.targetHits=b.targetHits;a.targetWindowStart=b.targetWindowStart;
            a.targetLast=b.targetLast;
        }else{
            a.targetHits=std::max(a.targetHits,b.targetHits);
            if(!a.targetWindowStart)a.targetWindowStart=b.targetWindowStart;
            a.targetLast=std::max(a.targetLast,b.targetLast);
        }
        if(b.attentionEvaluation>0||a.attentionEvaluation==0)
            a.attentionEvaluation=b.attentionEvaluation;
        a.bodyMatched|=b.bodyMatched;a.faceMatched|=b.faceMatched;a.hasFace|=b.hasFace;
        a.motionConfirmed|=b.motionConfirmed;a.nearEdge=a.nearEdge&&b.nearEdge;
        a.exitLikely|=b.exitLikely;
        a.bodyTravel=std::max(a.bodyTravel,b.bodyTravel);a.maxBodyShift=std::max(a.maxBodyShift,b.maxBodyShift);
        a.maxBodyScale=std::max(a.maxBodyScale,b.maxBodyScale);
        a.reachCounted|=b.reachCounted;a.impressionCounted|=b.impressionCounted;
        a.newImpression=(a.newImpression||b.newImpression)&&!oldImpression;
        a.eventFlags|=b.eventFlags;if(oldReach)a.eventFlags&=~1;
        if(oldAttribution)a.eventFlags&=~(2|4|8);
        if(b.genderConfidence>a.genderConfidence){a.gender=b.gender;a.genderConfidence=b.genderConfidence;}
        if(b.attributedGender>=0&&(a.attributedGender<0||b.attributedQuality>a.attributedQuality)){
            a.attributedGender=b.attributedGender;a.attributedQuality=b.attributedQuality;
        }
        a.maxGenderQuality=std::max(a.maxGenderQuality,b.maxGenderQuality);
        for(const auto& sample:b.samples)addGenderSample(a,sample);
        lockVote(a);updateAttribution(a);
    }

    void mergeDuplicateTracks(int64_t ts){
        for(size_t i=0;i<tracks.size();++i){
            for(size_t j=i+1;j<tracks.size();){
                if(duplicateTracks(tracks[i],tracks[j],ts)){
                    mergeInto(tracks[i],tracks[j]);tracks.erase(tracks.begin()+j);
                }else ++j;
            }
        }
    }

    Scrfd detector;PersonDetector people;GenderNet gender;std::vector<PersonTrack> tracks;
    std::vector<uint8_t> genderAlignedScratch;
    std::mutex lock;
    AttentionCalibration attentionCalibration;
    int frameNo=0,nextId=1;bool ready=false;
    float emaProcessMs=38.f;
};

static jfloatArray toArray(JNIEnv* env,const std::vector<float>& values){
    jfloatArray a=env->NewFloatArray((jsize)values.size());
    if(a&&!values.empty())env->SetFloatArrayRegion(a,0,(jsize)values.size(),values.data());
    return a;
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_tridi_audience_NativeBridge_create(JNIEnv* env,jclass,jobject assetManager){
    AAssetManager* assets=AAssetManager_fromJava(env,assetManager);if(!assets)return 0;
    std::unique_ptr<Engine> e(new Engine());if(!e->load(assets)){LOGE("model loading failed");return 0;}
    return (jlong)e.release();
}

extern "C" JNIEXPORT void JNICALL
Java_com_tridi_audience_NativeBridge_destroy(JNIEnv*,jclass,jlong handle){delete reinterpret_cast<Engine*>(handle);}

extern "C" JNIEXPORT void JNICALL
Java_com_tridi_audience_NativeBridge_setAttentionCalibration(JNIEnv*,jclass,jlong handle,
    jboolean enabled,jfloat eyeIntercept,jfloat eyeSlope,jfloat mouthIntercept,
    jfloat mouthSlope,jfloat eyeTolerance,jfloat mouthTolerance){
    Engine* e=reinterpret_cast<Engine*>(handle);if(!e)return;
    e->setAttentionCalibration(enabled==JNI_TRUE,eyeIntercept,eyeSlope,mouthIntercept,
                               mouthSlope,eyeTolerance,mouthTolerance);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_tridi_audience_NativeBridge_processYuv420(JNIEnv* env,jclass,jlong handle,
    jobject yb,jint yo,jint ys,jobject ub,jint uo,jint us,jint up,jobject vb,jint vo,jint vs,jint vp,
    jint width,jint height,jint clockwiseRotation,jlong timestamp){
    Engine* e=reinterpret_cast<Engine*>(handle);
    auto* y=(uint8_t*)env->GetDirectBufferAddress(yb);auto* u=(uint8_t*)env->GetDirectBufferAddress(ub);auto* v=(uint8_t*)env->GetDirectBufferAddress(vb);
    if(!e||!y||!u||!v||width<=0||height<=0)return toArray(env,{0,(float)width,(float)height,0,0,0});
    int rotation=((clockwiseRotation%360)+360)%360;
    if(rotation!=90&&rotation!=180&&rotation!=270)rotation=0;
    int logicalWidth=(rotation==90||rotation==270)?height:width;
    int logicalHeight=(rotation==90||rotation==270)?width:height;
    YuvFrame f{y+yo,u+uo,v+vo,logicalWidth,logicalHeight,width,height,
               ys,us,vs,up,vp,rotation};
    return toArray(env,e->process(f,timestamp));
}