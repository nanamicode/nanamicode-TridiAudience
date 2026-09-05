#!/usr/bin/env python3
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text()

def rep(old,new,label):
    global s
    if old not in s:
        raise SystemExit(f'missing anchor: {label}')
    s=s.replace(old,new,1)

old='''    float maleProbability(const std::vector<uint8_t>& rgb) {\n        ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data(), ncnn::Mat::PIXEL_RGB, 128, 128);\n        const float mean[3] = {123.675f,116.28f,103.53f};\n        const float norm[3] = {.0171247538f,.0175070028f,.0174291939f};\n        in.substract_mean_normalize(mean,norm);\n        ncnn::Extractor ex=net.create_extractor(); ex.input("in0",in);\n        ncnn::Mat out; if(ex.extract("out0",out)!=0 || out.total()<2) return -.1f;\n        float m=std::max(out[0],out[1]),e0=std::exp(out[0]-m),e1=std::exp(out[1]-m);\n        return e1/(e0+e1);\n    }\nprivate: ncnn::Net net;\n};'''
new='''    float maleProbability(const std::vector<uint8_t>& rgb) {\n        return maleProbabilityRaw(rgb);\n    }\n    // Same FastFace MobileNetV3 INT8. Only uncertain crops receive a mirrored\n    // second opinion; strong self-disagreement abstains instead of forcing a label.\n    float maleProbabilityStable(const std::vector<uint8_t>& rgb, float* flipAgreement=nullptr) {\n        float p0=maleProbabilityRaw(rgb);\n        if(p0<0.f){if(flipAgreement)*flipAgreement=0.f;return p0;}\n        if(p0<=.26f||p0>=.74f){if(flipAgreement)*flipAgreement=1.f;return p0;}\n        mirrorScratch.resize(rgb.size());\n        for(int y=0;y<128;++y)for(int x=0;x<128;++x){\n            const uint8_t* src=&rgb[((size_t)y*128+x)*3];\n            uint8_t* dst=&mirrorScratch[((size_t)y*128+(127-x))*3];\n            dst[0]=src[0];dst[1]=src[1];dst[2]=src[2];\n        }\n        float p1=maleProbabilityRaw(mirrorScratch);\n        if(p1<0.f){if(flipAgreement)*flipAgreement=0.f;return p0;}\n        float delta=std::fabs(p0-p1);\n        float agreement=1.f-clampf(delta/.30f,0.f,1.f);\n        if(flipAgreement)*flipAgreement=agreement;\n        if(delta>.28f)return -.2f;\n        return .5f*(p0+p1);\n    }\nprivate:\n    float maleProbabilityRaw(const std::vector<uint8_t>& rgb) {\n        ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data(), ncnn::Mat::PIXEL_RGB, 128, 128);\n        const float mean[3] = {123.675f,116.28f,103.53f};\n        const float norm[3] = {.0171247538f,.0175070028f,.0174291939f};\n        in.substract_mean_normalize(mean,norm);\n        ncnn::Extractor ex=net.create_extractor(); ex.set_light_mode(true); ex.input("in0",in);\n        ncnn::Mat out; if(ex.extract("out0",out)!=0 || out.total()<2) return -.1f;\n        float m=std::max(out[0],out[1]),e0=std::exp(out[0]-m),e1=std::exp(out[1]-m);\n        return e1/(e0+e1);\n    }\n    ncnn::Net net;\n    std::vector<uint8_t> mirrorScratch;\n};'''
rep(old,new,'GenderNet')

old='''        // A light face scan every three delivered frames gives much lower\n        // response latency than the v1.19 five-frame cadence. Medium/max scans\n        // remain periodic for distant faces, while the average pixel budget is\n        // lower than v1.19 to avoid stalling the BTV B11 ARMv7 CPU.\n        int mode=(frameNo==1||frameNo%54==0)?3:(frameNo%18==0?2:(frameNo%3==0?1:0));\n        bool bodyScan=(frameNo==1||frameNo%12==0);'''
new='''        // Preserve the validated scan hierarchy while adapting cadence when\n        // actual ARMv7 frame cost rises, so detector bursts do not starve preview.\n        int lightEvery=3,mediumEvery=18,maxEvery=54,bodyEvery=12;\n        if(emaProcessMs>72.f){lightEvery=5;mediumEvery=30;maxEvery=90;bodyEvery=15;}\n        else if(emaProcessMs>48.f){lightEvery=4;mediumEvery=24;maxEvery=72;bodyEvery=12;}\n        int mode=(frameNo==1||frameNo%maxEvery==0)?3:\n                 (frameNo%mediumEvery==0?2:(frameNo%lightEvery==0?1:0));\n        bool bodyScan=(frameNo==1||frameNo%bodyEvery==0);'''
rep(old,new,'adaptive cadence')

old='''                if(visual>=.14f&&quality>=.020f&&room){\n                    float p=gender.maleProbability(aligned);\n                    t.lastGender=ts;\n                    if(p>=0.f){addGenderSample(t,{p,quality});lockVote(t);}\n                }'''
new='''                if(visual>=.14f&&quality>=.020f&&room){\n                    float flipAgreement=1.f;\n                    float p=gender.maleProbabilityStable(aligned,&flipAgreement);\n                    t.lastGender=ts;\n                    if(p>=0.f&&flipAgreement>=.22f){\n                        quality*=.72f+.28f*flipAgreement;\n                        addGenderSample(t,{p,quality});lockVote(t);\n                    }\n                } else {\n                    t.lastGender=ts;\n                }'''
rep(old,new,'stable gender inference')

s=s.replace('''        bool ready=(n>=5&&confidence>=.90f&&agreement>=.90f)||\n                   (n>=8&&confidence>=.82f&&agreement>=.8125f)||\n                   (n>=11&&confidence>=.76f&&agreement>=.80f);''',
'''        bool ready=(n>=5&&confidence>=.91f&&agreement>=.92f)||\n                   (n>=8&&confidence>=.84f&&agreement>=.86f)||\n                   (n>=11&&confidence>=.79f&&agreement>=.84f);''')
s=s.replace('if(n>=11&&confidence>=.92f&&agreement>=.92f){',
            'if(n>=11&&confidence>=.94f&&agreement>=.94f){')

old='''        out[5]=(float)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count();\n        return out;'''
new='''        out[5]=(float)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count();\n        emaProcessMs=.88f*emaProcessMs+.12f*out[5];\n        return out;'''
rep(old,new,'EMA update')

if 'int frameNo=0,nextId=1;bool ready=false;' in s:
    s=s.replace('int frameNo=0,nextId=1;bool ready=false;',
                'int frameNo=0,nextId=1;bool ready=false;\n    float emaProcessMs=38.f;',1)
elif 'int frameNo = 0;' in s:
    s=s.replace('int frameNo = 0;','int frameNo = 0;\n    float emaProcessMs=38.f;',1)
else:
    raise SystemExit('missing frameNo anchor')

p.write_text(s)
print('patched',p,'bytes',len(s))
