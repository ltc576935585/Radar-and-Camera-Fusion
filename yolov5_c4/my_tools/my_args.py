from easydict import  EasyDict as edict

def train_args():
    args={
        'adam': False,
        'batch_size': 16,## 改一下看GPU大小 我8g是16
        'bucket': '',#gsutil bucket
        'cache_images': False,
        'cfg': './models/yolov5s_c4.yaml',
        'data': './data/v1.0-trainval.yaml',
        'device': '0',
        'epochs': 600,#300
        'evolve': False,#evolve hyperparameters
        'img_size': [640,640],#[640,640]
        'multi_scale': False,
        'name': '',
        'noautoanchor': False,
        'nosave': False,
        'notest': False,
        'rect': False,
        'resume': False,
        'single_cls': False,
        'weights': '',
        'hyp': None
    }
    args=edict(args)
    return  args

def test_args():
    args={
            'augment':False,
            'batch_size':128,
            'conf_thres':0.001,#'object confidence threshold'
            'data':'./data/v1.0-trainval.yaml',
            'device':'0',
            'img_size':640,#640
            'iou_thres':0.65,#for nms
            'merge':False,
            'save_json':False,
            'single_cls':False,#treat as single-class dataset'
            'task':'test',
            'verbose':True,##report mAP by class
            'weights':'/home/ltc/PycharmProjects/A_to_robin/yolov5_c4/runs/c4_600_trainval/weights/best.pt'

    }
    args = edict(args)
    return args

def my_detect_args():
    args={
        'agnostic_nms':False,
        'augment':False,
        'classes':None,
        'conf_thres':0.6,
        'device':'0',
        'img_size':640,
        'iou_thres':0.7,
        'output':'/home/ltc/PycharmProjects/yolov5_c4/inference/output_crf_rain/',
        'save_txt':False,
        'source':'/media/ltc/F/Adatasets/nuscense/data/test_rain/images/imgs/',
        'update':False,
        'view_img':False,
        'weights':'/home/ltc/PycharmProjects/A_to_robin/yolov5_c4/runs/c4_600_trainval/weights/best.pt'

    }
    args = edict(args)
    return args

def detect_args():
    args={
        'agnostic_nms':False,
        'augment':False,
        'classes':None,
        'conf_thres':0.4,
        'device':'0',
        'img_size':512,
        'iou_thres':0.5,
        'output':'/home/ltc/PycharmProjects/yolov5/inference/output_crf/',
        'save_txt':False,
        'source':'/home/ltc/PycharmProjects/yolov5/inference/images/',
        'update':False,
        'view_img':False,
        'weights':'/home/ltc/PycharmProjects/yolov5/runs/exp64/weights/best.pt'

    }
    args = edict(args)
    return args