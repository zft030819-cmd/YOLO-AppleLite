"""CPU NCNN inference and class-aware NMS for the bundled V5+NWD models."""
import hashlib
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ZH_NAMES={'Brown_Spot':'褐斑病','Frog_Eye_Leaf_Spot':'蛙眼叶斑病','Grey_Spot':'灰斑病',
          'Healthy':'健康叶片','Mosaic':'花叶病','Powdery_Mildew':'白粉病','Rust':'锈病','Scab':'黑星病'}


def preprocess(image,size):
    h,w=image.shape[:2]
    gain=min(size/h,size/w)
    resized=cv2.resize(image,(max(1,round(w*gain)),max(1,round(h*gain))))
    left=(size-resized.shape[1])//2;top=(size-resized.shape[0])//2
    padded=np.full((size,size,3),114,np.uint8)
    padded[top:top+resized.shape[0],left:left+resized.shape[1]]=resized
    return np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1),dtype=np.float32)/255,gain,left,top


def postprocess(raw,size,shape,gain,left,top,names,confidence,iou):
    anchors=sum((size//s)**2 for s in (8,16,32))
    if raw.shape!=(4+len(names),anchors) or not np.isfinite(raw).all():
        raise RuntimeError('模型输出格式异常：'+str(raw.shape))
    candidates=raw.T
    cls=candidates[:,4:].argmax(1);score=candidates[:,4:].max(1)
    mask=score>=confidence
    candidates,cls,score=candidates[mask],cls[mask],score[mask]
    boxes=candidates[:,:4].copy();boxes[:,:2]-=boxes[:,2:]/2
    # Bound pathological input/model output cost before NMS.
    if len(score)>30000:
        pick=np.argsort(score)[-30000:];boxes,cls,score=boxes[pick],cls[pick],score[pick]
    keep=[]
    for c in np.unique(cls):
        indices=np.flatnonzero(cls==c)
        chosen=cv2.dnn.NMSBoxes(boxes[indices].tolist(),score[indices].tolist(),confidence,iou)
        keep.extend(indices[np.asarray(chosen,dtype=int).reshape(-1)].tolist())
    h,w=shape[:2];detections=[]
    for k in sorted(keep,key=lambda i:-float(score[i]))[:300]:
        x,y,bw,bh=boxes[k]
        box=[float(np.clip((x-left)/gain,0,w)),float(np.clip((y-top)/gain,0,h)),
             float(np.clip((x+bw-left)/gain,0,w)),float(np.clip((y+bh-top)/gain,0,h))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        label=names[int(cls[k])]
        detections.append(dict(class_id=int(cls[k]),label=label,label_zh=ZH_NAMES.get(label,label),
                               confidence=float(score[k]),box=[round(v,2) for v in box]))
    return detections


def annotate(image,detections):
    out=image.copy();scale=max(.45,min(.8,out.shape[1]/1400))
    for d in detections:
        box=[int(round(v)) for v in d['box']]
        color=(78,150,49) if d['label']=='Healthy' else (49,132,240)
        p1=(min(out.shape[1]-1,box[0]),min(out.shape[0]-1,box[1]))
        p2=(min(out.shape[1]-1,box[2]),min(out.shape[0]-1,box[3]))
        cv2.rectangle(out,p1,p2,color,2)
        text=f'{d["label"]} {d["confidence"]:.2f}'
        (tw,th),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
        y=max(th+6,p1[1]);x=min(p1[0],max(0,out.shape[1]-tw-8))
        cv2.rectangle(out,(x,y-th-6),(min(out.shape[1]-1,x+tw+6),y+baseline),color,-1)
        cv2.putText(out,text,(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)
    return out


class Detector:
    def __init__(self,root,threads=4):
        self.root=Path(root);self.threads=threads
        self.manifest=json.loads((self.root/'models.json').read_text(encoding='utf-8'))
        self.profiles={p['id']:p for p in self.manifest['profiles']}
        self.names=self.manifest['class_names'];self.lock=threading.Lock();self.models={}
        if len(self.names)!=8 or not self.profiles:raise ValueError('模型清单不完整')
        for p in self.profiles.values():
            directory=self.root/p['path']
            for filename,key in [('model.ncnn.param','param_sha256'),('model.ncnn.bin','bin_sha256')]:
                path=directory/filename
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=p[key]:
                    raise RuntimeError('模型文件缺失或校验失败：'+str(path))

    def profile_info(self):
        return [dict(id=p['id'],size=p['size'],label=p['label'],precision=p['precision']) for p in self.profiles.values()]

    def _load(self,profile):
        if profile in self.models:return self.models[profile],False
        import ncnn
        p=self.profiles[profile];net=ncnn.Net()
        net.opt.num_threads=self.threads;net.opt.use_vulkan_compute=False
        net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False;net.opt.use_fp16_packed=False
        directory=self.root/p['path']
        if net.load_param(str(directory/'model.ncnn.param'))!=0 or net.load_model(str(directory/'model.ncnn.bin'))!=0:
            raise RuntimeError('模型加载失败，请检查部署文件')
        dummy=np.zeros((3,p['size'],p['size']),np.float32)
        for _ in range(2):self._extract(net,dummy,p)
        self.models[profile]=net
        return net,True

    @staticmethod
    def _extract(net,tensor,p):
        import ncnn
        with net.create_extractor() as ex:
            if ex.input(p.get('input_name','in0'),ncnn.Mat(tensor).clone())!=0:
                raise RuntimeError('NCNN输入失败')
            code,value=ex.extract(p.get('output_name','out0'))
            if code:raise RuntimeError('NCNN推理失败：'+str(code))
            return np.array(value,copy=True)

    def detect(self,image,profile,confidence=.25,nms_iou=.45):
        iou=nms_iou
        if profile not in self.profiles:raise ValueError('模型模式不可用')
        if not (np.isfinite(confidence) and .05<=confidence<=.95 and np.isfinite(iou) and .1<=iou<=.9):
            raise ValueError('检测阈值超出范围')
        if image is None or image.ndim!=3 or image.shape[2]!=3:raise ValueError('图像格式异常')
        waiting=time.perf_counter()
        with self.lock:
            lock_ms=(time.perf_counter()-waiting)*1000
            load_start=time.perf_counter();net,loaded=self._load(profile)
            load_ms=(time.perf_counter()-load_start)*1000 if loaded else 0.
            start=time.perf_counter();p=self.profiles[profile]
            tensor,gain,left,top=preprocess(image,p['size'])
            infer_start=time.perf_counter();raw=self._extract(net,tensor,p)
            infer_ms=(time.perf_counter()-infer_start)*1000
            detections=postprocess(raw,p['size'],image.shape,gain,left,top,self.names,confidence,iou)
            drawn=annotate(image,detections)
            ok,jpeg=cv2.imencode('.jpg',drawn,[cv2.IMWRITE_JPEG_QUALITY,88])
            if not ok:raise RuntimeError('结果图片编码失败')
            return dict(profile=profile,model=self.manifest['model_name'],input_size=p['size'],
                confidence=confidence,nms_iou=iou,width=image.shape[1],height=image.shape[0],
                detections=detections,inference_ms=round(infer_ms,2),
                processing_ms=round((time.perf_counter()-start)*1000,2),
                queue_ms=round(lock_ms,2),model_load_ms=round(load_ms,2),warmed_up=True),jpeg.tobytes()
