#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "net.h"
#include "cpu.h"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,"TridiGaze",__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,"TridiGaze",__VA_ARGS__)

namespace {

enum {
    HEADER=6, STRIDE=24,
    ID=0, X=1, Y=2, WIDTH=3, HEIGHT=4, GENDER=5, GENDER_CONF=6,
    ATTENTIVE=7, NEW_IMPRESSION=8, DETECTOR_SCORE=9, GENDER_SAMPLES=10,
    EVENT_FLAGS=11, ATTENTION_EVALUATION=12, ATTENTION_SCORE=13,
    ATTENTION_STREAK=14, SIGNED_YAW_EYE=15, SIGNED_YAW_MOUTH=16,
    FACE_SCALE=17, PITCH_RATIO=18, ROLL_DEGREES=19,
    EXPECTED_YAW_EYE=20, EXPECTED_YAW_MOUTH=21, TARGET_RESIDUAL=22,
    ATTENTION_GEOMETRY_VALID=23
};

static float clampf(float v,float lo,float hi){return std::max(lo,std::min(v,hi));}
static int clampi(int v,int lo,int hi){return std::max(lo,std::min(v,hi));}

using CoreCreateFn=jlong(*)(JNIEnv*,jclass,jobject);
using CoreDestroyFn=void(*)(JNIEnv*,jclass,jlong);
using CoreSetCalFn=void(*)(JNIEnv*,jclass,jlong,jboolean,jfloat,jfloat,jfloat,jfloat,jfloat,jfloat);
using CoreProcessFn=jfloatArray(*)(JNIEnv*,jclass,jlong,jobject,jint,jint,jobject,jint,jint,jint,
                                   jobject,jint,jint,jint,jint,jint,jint,jlong);

struct CoreApi {
    void* lib=nullptr;
    CoreCreateFn create=nullptr;
    CoreDestroyFn destroy=nullptr;
    CoreSetCalFn setCal=nullptr;
    CoreProcessFn process=nullptr;
    bool load(){
        if(process&&create&&destroy&&setCal)return true;
        if(!lib)lib=dlopen("libtridi_vision_core.so",RTLD_NOW);
        if(!lib){LOGE("dlopen core failed: %s",dlerror());return false;}
        create=(CoreCreateFn)dlsym(lib,"Java_com_tridi_audience_NativeBridge_create");
        destroy=(CoreDestroyFn)dlsym(lib,"Java_com_tridi_audience_NativeBridge_destroy");
        setCal=(CoreSetCalFn)dlsym(lib,"Java_com_tridi_audience_NativeBridge_setAttentionCalibration");
        process=(CoreProcessFn)dlsym(lib,"Java_com_tridi_audience_NativeBridge_processYuv420");
        return create&&destroy&&setCal&&process;
    }
};
static CoreApi gCore;

struct Rect {float x=0,y=0,w=0,h=0;};
struct Point {float x=0,y=0;};

struct Frame {
    const uint8_t *y=nullptr,*u=nullptr,*v=nullptr;
    int yo=0,yrs=0,uo=0,urs=0,ups=0,vo=0,vrs=0,vps=0;
    int sw=0,sh=0,lw=0,lh=0,rot=0;

    void source(int px,int py,int& sx,int& sy) const {
        px=clampi(px,0,lw-1);py=clampi(py,0,lh-1);
        if(rot==90){sx=py;sy=sh-1-px;}
        else if(rot==180){sx=sw-1-px;sy=sh-1-py;}
        else if(rot==270){sx=sw-1-py;sy=px;}
        else {sx=px;sy=py;}
        sx=clampi(sx,0,sw-1);sy=clampi(sy,0,sh-1);
    }
    void rgbAt(int px,int py,uint8_t* out) const {
        int sx,sy;source(px,py,sx,sy);
        int yy=y[yo+sy*yrs+sx]&255;
        int ux=sx>>1,uy=sy>>1;
        int uu=(u[uo+uy*urs+ux*ups]&255)-128;
        int vv=(v[vo+uy*vrs+ux*vps]&255)-128;
        int c=std::max(0,yy-16);
        out[0]=(uint8_t)clampi((298*c+409*vv+128)>>8,0,255);
        out[1]=(uint8_t)clampi((298*c-100*uu-208*vv+128)>>8,0,255);
        out[2]=(uint8_t)clampi((298*c+516*uu+128)>>8,0,255);
    }
    void rgbBilinear(float fx,float fy,uint8_t* out) const {
        int x0=(int)std::floor(fx),y0=(int)std::floor(fy);
        float dx=fx-x0,dy=fy-y0;
        uint8_t c00[3],c10[3],c01[3],c11[3];
        rgbAt(x0,y0,c00);rgbAt(x0+1,y0,c10);rgbAt(x0,y0+1,c01);rgbAt(x0+1,y0+1,c11);
        for(int k=0;k<3;k++){
            float a=c00[k]+dx*(c10[k]-c00[k]),b=c01[k]+dx*(c11[k]-c01[k]);
            out[k]=(uint8_t)clampi((int)std::lround(a+dy*(b-a)),0,255);
        }
    }
    bool crop(const Rect& r,int ow,int oh,std::vector<uint8_t>& dst) const {
        if(r.w<2||r.h<2||r.x<0||r.y<0||r.x+r.w>lw||r.y+r.h>lh)return false;
        dst.resize((size_t)ow*oh*3);
        for(int j=0;j<oh;j++){
            float sy=r.y+(j+.5f)*r.h/oh-.5f;
            for(int i=0;i<ow;i++){
                float sx=r.x+(i+.5f)*r.w/ow-.5f;
                rgbBilinear(sx,sy,&dst[((size_t)j*ow+i)*3]);
            }
        }
        return true;
    }
    bool cropRotated(const Rect& r,int ow,int oh,float angleDeg,std::vector<uint8_t>& dst) const {
        if(r.w<2||r.h<2||r.x<0||r.y<0||r.x+r.w>lw||r.y+r.h>lh)return false;
        const float rad=angleDeg*3.14159265358979323846f/180.f;
        const float cs=std::cos(rad),sn=std::sin(rad);
        const float cx=r.x+r.w*.5f,cy=r.y+r.h*.5f;
        dst.resize((size_t)ow*oh*3);
        for(int j=0;j<oh;j++){
            const float dy=((j+.5f)/oh-.5f)*r.h;
            for(int i=0;i<ow;i++){
                const float dx=((i+.5f)/ow-.5f)*r.w;
                // Same visual rotation convention as OpenCV warpAffine with
                // getRotationMatrix2D(center, +angle).
                const float sx=cx+cs*dx-sn*dy;
                const float sy=cy+sn*dx+cs*dy;
                if(sx<0||sy<0||sx>lw-1||sy>lh-1)return false;
                rgbBilinear(sx,sy,&dst[((size_t)j*ow+i)*3]);
            }
        }
        return true;
    }
};

struct GazeResult {
    bool valid=false,accepted=false;
    float score=0,yaw=0,pitch=0,roll=0,gx=0,gy=0,gz=-1;
    float hitX=0,hitY=0,depth=0,residual=9;
};

class NeuralGaze {
public:
    bool load(AAssetManager* assets){
        config(head);config(lm);config(gaze);
        if(head.load_param(assets,"gaze/head_pose.param")!=0||head.load_model(assets,"gaze/head_pose.bin")!=0)return false;
        if(lm.load_param(assets,"gaze/landmarks35.param")!=0||lm.load_model(assets,"gaze/landmarks35.bin")!=0)return false;
        if(gaze.load_param(assets,"gaze/gaze.param")!=0||gaze.load_model(assets,"gaze/gaze.bin")!=0)return false;
        resolve();
        LOGI("gaze io head=%d ypr=%d/%d/%d comb=%d lm=%d/%d gaze=%d/%d/%d -> %d",
             headIn,yawOut,pitchOut,rollOut,headCombined,lmIn,lmOut,leftIn,rightIn,poseIn,gazeOut);
        return headIn>=0&&lmIn>=0&&lmOut>=0&&leftIn>=0&&rightIn>=0&&poseIn>=0&&gazeOut>=0&&
               (headCombined>=0||(yawOut>=0&&pitchOut>=0&&rollOut>=0));
    }

    GazeResult infer(const Frame& f,const Rect& face,float detector) const {
        GazeResult r;
        if(detector<.50f||face.w<56.f||face.h<56.f)return r;
        std::vector<uint8_t> faceRgb;
        if(!f.crop(face,60,60,faceRgb))return r;
        ncnn::Mat fi=ncnn::Mat::from_pixels(faceRgb.data(),ncnn::Mat::PIXEL_RGB2BGR,60,60);

        {
            ncnn::Extractor ex=head.create_extractor();ex.set_light_mode(true);
            if(ex.input(headIn,fi)!=0)return r;
            if(headCombined>=0){
                ncnn::Mat o;if(ex.extract(headCombined,o)!=0||o.total()<3)return r;
                r.yaw=o[0];r.pitch=o[1];r.roll=o[2];
            }else{
                ncnn::Mat y,p,rr;
                if(ex.extract(yawOut,y)!=0||ex.extract(pitchOut,p)!=0||ex.extract(rollOut,rr)!=0||
                   y.empty()||p.empty()||rr.empty())return r;
                r.yaw=y[0];r.pitch=p[0];r.roll=rr[0];
            }
        }
        if(!std::isfinite(r.yaw)||!std::isfinite(r.pitch)||!std::isfinite(r.roll)||
           std::fabs(r.yaw)>75.f||std::fabs(r.pitch)>55.f||std::fabs(r.roll)>45.f)return r;

        ncnn::Mat landmarks;
        {
            ncnn::Extractor ex=lm.create_extractor();ex.set_light_mode(true);
            if(ex.input(lmIn,fi)!=0||ex.extract(lmOut,landmarks)!=0||landmarks.total()<8)return r;
        }
        Point e[4];
        for(int i=0;i<4;i++){
            float nx=landmarks[2*i],ny=landmarks[2*i+1];
            if(!std::isfinite(nx)||!std::isfinite(ny)||nx<-.1f||nx>1.1f||ny<-.1f||ny>1.1f)return r;
            e[i]={face.x+nx*face.w,face.y+ny*face.h};
        }
        Rect le=eyeBox(e[0],e[1]),re=eyeBox(e[2],e[3]);
        std::vector<uint8_t> lrgb,rrgb;
        // Follow the official OMZ gaze pipeline: roll-align both eye crops,
        // feed roll=0, then rotate the predicted gaze back afterwards.
        if(!f.cropRotated(le,60,60,r.roll,lrgb)||!f.cropRotated(re,60,60,r.roll,rrgb))return r;
        ncnn::Mat li=ncnn::Mat::from_pixels(lrgb.data(),ncnn::Mat::PIXEL_RGB2BGR,60,60);
        ncnn::Mat ri=ncnn::Mat::from_pixels(rrgb.data(),ncnn::Mat::PIXEL_RGB2BGR,60,60);
        ncnn::Mat pose(3);pose[0]=r.yaw;pose[1]=r.pitch;pose[2]=0.f;
        ncnn::Mat gv;
        {
            ncnn::Extractor ex=gaze.create_extractor();ex.set_light_mode(true);
            if(ex.input(leftIn,li)!=0||ex.input(rightIn,ri)!=0||ex.input(poseIn,pose)!=0||
               ex.extract(gazeOut,gv)!=0||gv.total()<3)return r;
        }
        float norm=std::sqrt(gv[0]*gv[0]+gv[1]*gv[1]+gv[2]*gv[2]);
        if(norm<.25f||!std::isfinite(norm))return r;
        r.gx=gv[0]/norm;r.gy=gv[1]/norm;r.gz=gv[2]/norm;
        {
            const float rad=r.roll*3.14159265358979323846f/180.f;
            const float cs=std::cos(rad),sn=std::sin(rad);
            const float gx=r.gx*cs+r.gy*sn;
            const float gy=-r.gx*sn+r.gy*cs;
            r.gx=gx;r.gy=gy;
        }
        // OMZ convention: straight toward camera ~= (0,0,-1).
        if(r.gz>-.16f)return r;

        // EMEET S600 factory profile: 73-degree diagonal FoV at its wide
        // setting. Use the neural eye landmarks to estimate distance from
        // average inter-eye spacing instead of relying on face-box width.
        const float diagonal=std::sqrt((float)f.lw*f.lw+(float)f.lh*f.lh);
        const float focal=diagonal/(2.f*std::tan(73.f*3.14159265358979323846f/360.f));
        const float lcx=(e[0].x+e[1].x)*.5f,lcy=(e[0].y+e[1].y)*.5f;
        const float rcx=(e[2].x+e[3].x)*.5f,rcy=(e[2].y+e[3].y)*.5f;
        const float dx=lcx-rcx,dy=lcy-rcy;
        const float interEyePx=std::sqrt(dx*dx+dy*dy);
        if(interEyePx<18.f)return r;
        const float z=focal*63.f/interEyePx;
        if(z<450.f||z>3600.f)return r;
        const float cx=(lcx+rcx)*.5f,cy=(lcy+rcy)*.5f;
        const float eyeX=(cx-f.lw*.5f)*z/focal;
        const float eyeY=-(cy-f.lh*.5f)*z/focal;
        float t=-z/r.gz;if(t<=0||!std::isfinite(t))return r;
        r.hitX=eyeX+t*r.gx;r.hitY=eyeY+t*r.gy;r.depth=z;

        // Physical Tridi totem: display 320x540 mm; camera about 187.3 mm
        // horizontally from display centre and 70 mm below it, same plane.
        const float screenX=-187.3f,screenY=70.f;
        // Modest interior margin protects the 27 mm camera/display-edge gap.
        const float safeHalfW=136.f,safeHalfH=246.f;
        float nx=std::fabs(r.hitX-screenX)/safeHalfW;
        float ny=std::fabs(r.hitY-screenY)/safeHalfH;
        r.residual=std::max(nx,ny);
        r.valid=true;
        if(nx<=1.f&&ny<=1.f){
            r.accepted=true;
            r.score=clampf(.96f-.20f*r.residual,.72f,.96f);
        }else{
            r.score=clampf(.60f-.35f*std::max(0.f,r.residual-1.f),0.f,.60f);
        }
        return r;
    }

private:
    static void config(ncnn::Net& n){
        n.opt.num_threads=2;n.opt.use_vulkan_compute=false;
        n.opt.use_fp16_storage=false;n.opt.use_fp16_packed=false;n.opt.use_fp16_arithmetic=false;
    }
    static bool has(const char* s,const char* n){return s&&n&&std::strstr(s,n);}
    static int first(const std::vector<int>& v){return v.empty()?-1:v[0];}
    static Rect eyeBox(const Point& a,const Point& b){
        float dx=a.x-b.x,dy=a.y-b.y,size=std::sqrt(dx*dx+dy*dy)*1.8f;
        float mx=(a.x+b.x)*.5f,my=(a.y+b.y)*.5f;
        return {mx-size*.5f,my-size*.5f,size,size};
    }
    void resolve(){
        headIn=first(head.input_indexes());lmIn=first(lm.input_indexes());lmOut=first(lm.output_indexes());
        gazeOut=first(gaze.output_indexes());
        const auto& hn=head.output_names();const auto& hi=head.output_indexes();
        for(size_t i=0;i<hn.size()&&i<hi.size();i++){
            if(has(hn[i],"angle_y")||has(hn[i],"yaw"))yawOut=hi[i];
            else if(has(hn[i],"angle_p")||has(hn[i],"pitch"))pitchOut=hi[i];
            else if(has(hn[i],"angle_r")||has(hn[i],"roll"))rollOut=hi[i];
        }
        if(hi.size()==1)headCombined=hi[0];
        if(hi.size()==3&&(yawOut<0||pitchOut<0||rollOut<0)){
            // PINTO head-pose ONNX keeps OMZ yaw/pitch/roll ordering.
            yawOut=hi[0];pitchOut=hi[1];rollOut=hi[2];
        }
        const auto& gn=gaze.input_names();const auto& gi=gaze.input_indexes();
        for(size_t i=0;i<gn.size()&&i<gi.size();i++){
            if(has(gn[i],"left"))leftIn=gi[i];
            else if(has(gn[i],"right"))rightIn=gi[i];
            else if(has(gn[i],"head")||has(gn[i],"pose"))poseIn=gi[i];
        }
        if(gi.size()>=3){
            if(leftIn<0)leftIn=gi[0];if(rightIn<0)rightIn=gi[1];if(poseIn<0)poseIn=gi[2];
        }
    }
    ncnn::Net head,lm,gaze;
    int headIn=-1,yawOut=-1,pitchOut=-1,rollOut=-1,headCombined=-1;
    int lmIn=-1,lmOut=-1,leftIn=-1,rightIn=-1,poseIn=-1,gazeOut=-1;
};

struct Track {
    bool used=false,confirmed=false;
    int id=0,genderReported=-2;
    jlong lastSeen=0;
};

struct Wrapper {
    jlong core=0;
    NeuralGaze neural;
    Track tracks[256];

    Track* track(int id,jlong ts){
        int free=-1,old=0;jlong oldest=0x7fffffffffffffffLL;
        for(int i=0;i<256;i++){
            if(tracks[i].used&&tracks[i].id==id){tracks[i].lastSeen=ts;return &tracks[i];}
            if(!tracks[i].used&&free<0)free=i;
            if(tracks[i].used&&tracks[i].lastSeen<oldest){oldest=tracks[i].lastSeen;old=i;}
        }
        int i=free>=0?free:old;
        tracks[i]=Track{};tracks[i].used=true;tracks[i].id=id;tracks[i].lastSeen=ts;
        return &tracks[i];
    }
};

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_tridi_audience_NativeBridge_create(JNIEnv* env,jclass cls,jobject assetManager){
    if(!gCore.load())return 0;
    AAssetManager* assets=AAssetManager_fromJava(env,assetManager);if(!assets)return 0;
    std::unique_ptr<Wrapper> w(new Wrapper());
    w->core=gCore.create(env,cls,assetManager);
    if(!w->core||!w->neural.load(assets)){
        if(w->core)gCore.destroy(env,cls,w->core);
        LOGE("create failed: core or neural model");
        return 0;
    }
    // Disable the legacy calibrated/head-landmark impression gate in the core.
    gCore.setCal(env,cls,w->core,JNI_FALSE,0,0,0,0,.11f,.15f);
    return (jlong)w.release();
}

extern "C" JNIEXPORT void JNICALL
Java_com_tridi_audience_NativeBridge_destroy(JNIEnv* env,jclass cls,jlong handle){
    Wrapper* w=(Wrapper*)handle;if(!w)return;
    if(gCore.load()&&w->core)gCore.destroy(env,cls,w->core);
    delete w;
}

extern "C" JNIEXPORT void JNICALL
Java_com_tridi_audience_NativeBridge_setAttentionCalibration(
    JNIEnv* env,jclass cls,jlong handle,jboolean,jfloat,jfloat,jfloat,jfloat,jfloat,jfloat){
    Wrapper* w=(Wrapper*)handle;if(!w||!gCore.load())return;
    // Factory geometry is fixed; user calibration must not move the physical target.
    gCore.setCal(env,cls,w->core,JNI_FALSE,0,0,0,0,.11f,.15f);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_tridi_audience_NativeBridge_processYuv420(
    JNIEnv* env,jclass cls,jlong handle,
    jobject yb,jint yo,jint yrs,jobject ub,jint uo,jint urs,jint ups,
    jobject vb,jint vo,jint vrs,jint vps,jint sw,jint sh,jint rot,jlong ts){
    Wrapper* w=(Wrapper*)handle;
    if(!w||!gCore.load())return nullptr;
    jfloatArray arr=gCore.process(env,cls,w->core,yb,yo,yrs,ub,uo,urs,ups,vb,vo,vrs,vps,sw,sh,rot,ts);
    if(!arr)return arr;
    auto* yp=(uint8_t*)env->GetDirectBufferAddress(yb);
    auto* up=(uint8_t*)env->GetDirectBufferAddress(ub);
    auto* vp=(uint8_t*)env->GetDirectBufferAddress(vb);
    if(!yp||!up||!vp)return arr;

    jsize len=env->GetArrayLength(arr);if(len<HEADER)return arr;
    jboolean copy=JNI_FALSE;jfloat* a=env->GetFloatArrayElements(arr,&copy);if(!a)return arr;
    int lw=(int)std::lround(a[1]),lh=(int)std::lround(a[2]);
    int rr=((rot%360)+360)%360;if(rr!=90&&rr!=180&&rr!=270)rr=0;
    Frame f{yp,up,vp,yo,yrs,uo,urs,ups,vo,vrs,vps,sw,sh,lw,lh,rr};

    int n=(int)std::lround(a[4]);if(n<0)n=0;while(n&&HEADER+n*STRIDE>len)n--;
    for(int k=0;k<n;k++){
        int p=HEADER+k*STRIDE;
        int id=(int)std::lround(a[p+ID]);Track* tr=w->track(id,ts);
        int originalFlags=(int)std::lround(a[p+EVENT_FLAGS]);
        int flags=originalFlags&1; // preserve reach only; neural gate owns impression/attribution.
        int gender=a[p+GENDER]>=.5f?1:(a[p+GENDER]>-.5f?0:-1);

        a[p+ATTENTIVE]=0;a[p+NEW_IMPRESSION]=0;a[p+ATTENTION_EVALUATION]=0;a[p+ATTENTION_STREAK]=0;
        if(tr->confirmed){
            if(tr->genderReported==-1&&gender>=0){flags|=2;tr->genderReported=gender;}
            else if(tr->genderReported==0&&gender==1){flags|=4;tr->genderReported=1;}
            else if(tr->genderReported==1&&gender==0){flags|=8;tr->genderReported=0;}
            a[p+EVENT_FLAGS]=(float)flags;
            continue;
        }

        // Positive detector score means the exported rectangle is the current,
        // fresh SCRFD face rectangle rather than a tracked body rectangle.
        float det=a[p+DETECTOR_SCORE];
        if(det<=0.f){
            a[p+ATTENTION_GEOMETRY_VALID]=0;
            a[p+EVENT_FLAGS]=(float)flags;
            continue;
        }
        Rect face{a[p+X],a[p+Y],a[p+WIDTH],a[p+HEIGHT]};
        GazeResult g=w->neural.infer(f,face,det);
        a[p+ATTENTION_SCORE]=g.score;
        a[p+SIGNED_YAW_EYE]=(std::fabs(g.gz)>.01f)?g.gx/(-g.gz):0.f;
        a[p+SIGNED_YAW_MOUTH]=(std::fabs(g.gz)>.01f)?g.gy/(-g.gz):0.f;
        a[p+FACE_SCALE]=face.w/std::max(1.f,(float)lw);
        a[p+PITCH_RATIO]=g.pitch;
        a[p+ROLL_DEGREES]=std::fabs(g.roll);
        a[p+EXPECTED_YAW_EYE]=g.hitX;
        a[p+EXPECTED_YAW_MOUTH]=g.hitY;
        a[p+TARGET_RESIDUAL]=g.residual;
        a[p+ATTENTION_GEOMETRY_VALID]=g.valid?1.f:0.f;
        a[p+ATTENTION_EVALUATION]=g.valid?-1.f:0.f;

        if(g.accepted){
            // Core invariant: 50 bad frames + 1 true frame = exactly one impression.
            // NEW_IMPRESSION is raised only on this exact current frame, so Java
            // JPEG-encodes and uploads this exact winner.
            tr->confirmed=true;
            a[p+ATTENTIVE]=1.f;a[p+NEW_IMPRESSION]=1.f;
            a[p+ATTENTION_EVALUATION]=1.f;a[p+ATTENTION_STREAK]=1.f;
            if(gender>=0){flags|=2;tr->genderReported=gender;}else tr->genderReported=-1;
        }
        a[p+EVENT_FLAGS]=(float)flags;
    }
    env->ReleaseFloatArrayElements(arr,a,0);
    return arr;
}
