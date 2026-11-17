"""LeafWatch / AppleLite: local image detection and shared CSI/USB camera monitoring."""
import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import unquote,urlsplit
import uuid

import cv2
import numpy as np
from PIL import Image
from inference import Detector

ROOT=Path(__file__).resolve().parent
MAX_UPLOAD=12*1024*1024
MAX_PIXELS=16_000_000
ID_RE=re.compile(r'^[0-9a-f]{32}$')


def now():return datetime.now().astimezone().isoformat(timespec='seconds')


def settings(data,profiles,current):
    if not isinstance(data,dict) or set(data)-{'profile','confidence','nms_iou'}:
        raise ValueError('未知的检测设置')
    result=dict(current)
    for key,value in data.items():
        if key=='profile':
            if not isinstance(value,str) or value not in profiles:raise ValueError('模型模式不可用')
        else:
            low,high=(.05,.95) if key=='confidence' else (.1,.9)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not low<=value<=high:
                raise ValueError('阈值超出范围')
            value=float(value)
        result[key]=value
    return result


class Store:
    def __init__(self,path):self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock()

    def save(self,result,jpeg,source,filename=''):
        identity=uuid.uuid4().hex
        record=dict(result,id=identity,created=now(),source=source,filename=filename,
                    image_url=f'/api/results/{identity}/image',json_url=f'/api/results/{identity}/json')
        with self.lock:
            (self.path/f'{identity}.jpg').write_bytes(jpeg)
            temp=self.path/f'{identity}.tmp'
            temp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(self.path/f'{identity}.json')
        return record

    def get(self,identity):
        if not ID_RE.fullmatch(identity):raise FileNotFoundError('记录不存在')
        return json.loads((self.path/f'{identity}.json').read_text(encoding='utf-8'))

    def history(self):
        records=[]
        with self.lock:
            files=sorted(self.path.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:50]
            for p in files:
                try:
                    r=self.get(p.stem)
                    records.append({k:r[k] for k in ('id','created','source','filename','profile','model','input_size',
                        'width','height','detections','inference_ms','processing_ms','image_url','json_url')})
                except (OSError,ValueError,KeyError):continue
        return records


class Camera:
    def __init__(self,app):
        self.app=app;self.condition=threading.Condition();self.control=threading.Lock()
        self.halt=threading.Event();self.threads=[];self.frame=None;self.sequence=0;self.jpeg=None
        self.latest=None;self.published=0;self.captured=0.;self.mode='csi';self.index=0
        self.state=dict(status='idle',error='',capture_fps=0.,processing_fps=0.,frame_age_ms=0.,
                        width=0,height=0,inference_ms=0.,detections=[],published_at='',frame_id=0,demo=False)

    def start(self,mode,index):
        if mode not in ('csi','usb'):raise ValueError('摄像头类型必须为CSI或USB')
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<=9:raise ValueError('摄像头编号应为0至9')
        with self.control:
            if any(t.is_alive() for t in self.threads):raise RuntimeError('摄像头正在运行或停止中，请稍后重试')
            self.halt=threading.Event();self.mode=mode;self.index=index
            with self.condition:
                self.frame=None;self.sequence=0;self.published=0;self.jpeg=None;self.latest=None
                self.state.update(status='starting',error='',processing_fps=0.,capture_fps=0.,
                                  detections=[],frame_id=0,published_at='',demo=bool(self.app.args.demo_image))
            self.threads=[threading.Thread(target=self.capture,daemon=True,name='camera-capture'),
                          threading.Thread(target=self.process,daemon=True,name='camera-inference')]
            for t in self.threads:t.start()

    def stop(self):
        with self.control:
            self.halt.set()
            with self.condition:self.condition.notify_all()
            for t in self.threads:t.join(timeout=3)
            with self.condition:
                self.state.update(status='stopping' if any(t.is_alive() for t in self.threads) else 'stopped',
                                  processing_fps=0.,capture_fps=0.)

    def fail(self,text):
        self.halt.set()
        with self.condition:
            self.state.update(status='error',error=text,processing_fps=0.)
            self.condition.notify_all()

    def capture(self):
        camera=None;kind=None
        try:
            args=self.app.args
            if args.demo_image:
                still=cv2.imread(str(args.demo_image))
                if still is None:raise RuntimeError('无法读取测试图片')
                kind='demo'
            elif self.mode=='csi':
                from picamera2 import Picamera2
                if not Picamera2.global_camera_info():raise RuntimeError('未发现CSI摄像头，请先运行 rpicam-hello --list-cameras')
                camera=Picamera2(self.index);kind='csi'
                # Picamera2 RGB888 arrays are BGR byte ordered on little-endian Pi.
                camera.configure(camera.create_video_configuration(main=dict(size=(args.width,args.height),format='RGB888'),
                    controls={'FrameRate':args.camera_fps},buffer_count=4,queue=False))
                camera.start()
            else:
                camera=cv2.VideoCapture(self.index);kind='usb'
                camera.set(cv2.CAP_PROP_FRAME_WIDTH,args.width);camera.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
                camera.set(cv2.CAP_PROP_FPS,args.camera_fps);camera.set(cv2.CAP_PROP_BUFFERSIZE,1)
                if not camera.isOpened():raise RuntimeError('无法打开USB摄像头，请检查设备编号及占用情况')
            clock=deque(maxlen=60)
            while not self.halt.is_set():
                if kind=='demo':image=still.copy()
                elif kind=='csi':image=camera.capture_array('main')
                else:
                    ok,image=camera.read()
                    if not ok:raise RuntimeError('USB摄像头取帧失败')
                tick=time.perf_counter();clock.append(tick)
                fps=(len(clock)-1)/(clock[-1]-clock[0]) if len(clock)>1 else 0.
                with self.condition:
                    self.frame=image;self.sequence+=1;self.captured=tick
                    self.state.update(capture_fps=round(fps,2),width=image.shape[1],height=image.shape[0])
                    self.condition.notify_all()
                if kind=='demo':self.halt.wait(1/self.app.args.camera_fps)
        except Exception as e:self.fail('摄像头：'+str(e))
        finally:
            if camera is not None:
                if kind=='csi':camera.close()
                else:camera.release()

    def process(self):
        seen=0;clock=deque(maxlen=30);active=None
        try:
            while not self.halt.is_set():
                if self.app.single_pending.is_set():
                    self.halt.wait(.03);continue
                with self.condition:
                    self.condition.wait_for(lambda:self.halt.is_set() or self.sequence!=seen,timeout=1)
                    if self.halt.is_set():return
                    if self.frame is None or self.sequence==seen:continue
                    image,seen,captured=self.frame.copy(),self.sequence,self.captured
                config=self.app.config_snapshot()
                with self.condition:self.state['status']='processing'
                result,jpeg=self.app.detector.detect(image,**config)
                if self.halt.is_set():return
                tick=time.perf_counter()
                identity=(config['profile'],config['confidence'],config['nms_iou'])
                if identity!=active or result['model_load_ms']:
                    clock.clear();active=identity
                clock.append(tick)
                fps=(len(clock)-1)/(clock[-1]-clock[0]) if len(clock)>1 else 0.
                with self.condition:
                    self.published+=1;self.latest=result;self.jpeg=jpeg
                    self.state.update(status='running',error='',processing_fps=round(fps,2),
                        frame_age_ms=round((tick-captured)*1000,1),frame_id=self.published,published_at=now(),
                        inference_ms=result['inference_ms'],detections=result['detections'],profile=result['profile'],
                        processing_ms=result['processing_ms'])
                    self.condition.notify_all()
        except Exception as e:self.fail('检测：'+str(e))

    def snapshot(self):
        with self.condition:
            if self.state['status']=='stopping' and not any(t.is_alive() for t in self.threads):
                self.state['status']='stopped'
            return dict(self.state,source=self.mode,index=self.index)

    def save_frame(self):
        with self.condition:
            if not self.jpeg or self.latest is None:raise ValueError('暂无检测画面')
            result=dict(self.latest,camera_source=self.mode,demo=bool(self.app.args.demo_image))
            jpeg=self.jpeg
        return self.app.store.save(result,jpeg,'camera')


class Application:
    def __init__(self,args):
        self.args=args;self.detector=Detector(ROOT,args.threads)
        self.lock=threading.Lock();self.single_slot=threading.BoundedSemaphore(1);self.single_pending=threading.Event()
        self.config=dict(profile=self.detector.manifest['default_profile'],confidence=.25,nms_iou=.45)
        self.store=Store(args.results);self.camera=Camera(self)

    def config_snapshot(self):
        with self.lock:return dict(self.config)

    def status(self):
        temperature=None
        try:temperature=round(float(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000,1)
        except (OSError,ValueError):pass
        return dict(config=self.config_snapshot(),profiles=self.detector.profile_info(),camera=self.camera.snapshot(),
                    model=self.detector.manifest['model_name'],threads=self.args.threads,temperature_c=temperature,
                    upload_limit_mb=MAX_UPLOAD//1024//1024)

    def detect_upload(self,body,filename):
        if not self.single_slot.acquire(blocking=False):raise BlockingIOError('已有一张图片正在检测，请稍后再试')
        self.single_pending.set()
        try:
            try:
                with Image.open(BytesIO(body)) as img:
                    w,h=img.size
                    if img.format not in ('JPEG','PNG','WEBP'):raise ValueError('仅支持JPG、PNG、WebP图片')
                    if w*h>MAX_PIXELS or min(w,h)<8 or max(w,h)>8000:raise ValueError('图片尺寸需在8000像素边长、1600万像素以内')
            except (OSError,Image.DecompressionBombError) as e:raise ValueError('无法读取图片，请选择JPG、PNG或WebP文件') from e
            image=cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('图片内容损坏')
            original_size=[image.shape[1],image.shape[0]]
            if max(image.shape[:2])>1600:
                scale=1600/max(image.shape[:2]);image=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
            result,jpeg=self.detector.detect(image,**self.config_snapshot())
            result['original_size']=original_size
            return self.store.save(result,jpeg,'upload',filename)
        finally:self.single_pending.clear();self.single_slot.release()


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):super().setup();self.connection.settimeout(20)
        def log_message(self,*args):pass

        def send(self,code,body,kind='application/json; charset=utf-8',attachment=None):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode('utf-8')
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if attachment:self.send_header('Content-Disposition',f'attachment; filename="{attachment}"')
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path=='/':self.send(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
                elif path in ('/api/state','/api/health'):self.send(200,app.status())
                elif path=='/api/history':self.send(200,dict(records=app.store.history()))
                elif path=='/api/camera/frame.jpg':
                    with app.camera.condition:data=app.camera.jpeg
                    self.send(200,data,'image/jpeg') if data else self.send(503,dict(error='摄像头尚无画面'))
                elif path=='/stream.mjpg':self.stream()
                elif path.startswith('/api/results/'):
                    parts=path.split('/')
                    if len(parts)!=5:raise FileNotFoundError()
                    identity,kind=parts[3:];record=app.store.get(identity)
                    if kind=='image':
                        self.send(200,(app.store.path/f'{identity}.jpg').read_bytes(),'image/jpeg',f'applelite-{identity[:8]}.jpg')
                    elif kind=='json':self.send(200,record,attachment=f'applelite-{identity[:8]}.json')
                    else:raise FileNotFoundError()
                else:self.send(404,dict(error='页面不存在'))
            except (FileNotFoundError,KeyError):self.send(404,dict(error='记录不存在'))
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass

        def stream(self):
            self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control','no-store');self.end_headers();seen=-1
            while True:
                with app.camera.condition:
                    app.camera.condition.wait_for(lambda:app.camera.published!=seen,timeout=2)
                    jpeg,seen=app.camera.jpeg,app.camera.published
                if jpeg:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n')
                    self.wfile.flush()
                    if app.camera.halt.is_set():return
                elif app.camera.halt.is_set():return

        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                self.send(403,dict(error='请求来源不匹配'));return
            path=urlsplit(self.path).path
            try:
                length=int(self.headers.get('Content-Length','0'))
                limit=MAX_UPLOAD if path=='/api/detect' else 2048
                if length<0 or length>limit:self.send(413,dict(error='上传内容过大'));return
                body=self.rfile.read(length)
                if len(body)!=length:raise ValueError('请求数据不完整')
                if path=='/api/detect':
                    if not body:raise ValueError('请先选择图片')
                    filename=unquote(self.headers.get('X-Filename','image')).replace('\\','/').split('/')[-1][:120]
                    self.send(200,app.detect_upload(body,filename))
                elif path=='/api/config':
                    data=json.loads(body)
                    with app.lock:app.config=settings(data,app.detector.profiles,app.config)
                    self.send(200,app.status())
                elif path=='/api/camera/start':
                    data=json.loads(body or b'{}')
                    if not isinstance(data,dict) or set(data)-{'source','index'}:raise ValueError('摄像头参数无效')
                    app.camera.start(data.get('source','csi'),data.get('index',0));self.send(200,app.status())
                elif path=='/api/camera/stop':app.camera.stop();self.send(200,app.status())
                elif path=='/api/camera/save':self.send(200,app.camera.save_frame())
                else:self.send(404,dict(error='接口不存在'))
            except BlockingIOError as e:self.send(429,dict(error=str(e)))
            except (ValueError,TypeError,KeyError) as e:self.send(400,dict(error=str(e)))
            except RuntimeError as e:self.send(409,dict(error=str(e)))
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
            except Exception as e:
                print('Request failed: '+repr(e),flush=True);self.send(500,dict(error='处理失败，请查看服务日志：'+type(e).__name__))
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='0.0.0.0');parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=480);parser.add_argument('--camera-fps',type=float,default=20)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--demo-image',type=Path,help='Explicit still-frame testing only; not camera FPS')
    args=parser.parse_args()
    if min(args.threads,args.width,args.height,args.camera_fps)<=0 or not 1<=args.port<=65535:
        parser.error('线程、图像尺寸、帧率和端口必须有效')
    cv2.setNumThreads(1)
    app=Application(args);server=ThreadingHTTPServer((args.host,args.port),handler(app));server.daemon_threads=True
    print(f'AppleLite Studio http://{args.host}:{args.port}  Camera starts only when requested.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:app.camera.stop();server.server_close()


if __name__=='__main__':main()
