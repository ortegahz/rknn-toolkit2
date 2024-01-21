import cv2
from rknn.api import RKNN
import numpy as np

from test_mh import post_process_face, IMG_SIZE

# ONNX_MODEL = '/media/manu/data/sdks/sigmastar/Tiramisu_DLS00V010-20220107/ipu/SGS_IPU_SDK_vQ_0.1.0/demos/onnx_yolov5/acfree_320.onnx'
ONNX_MODEL = '/home/manu/tmp/face.onnx'
RKNN_MODEL = '/home/manu/tmp/acfree.rknn'
# IMG_PATH = '/media/manu/data/pics/students_lt_320.bmp'
IMG_PATH = '/media/manu/samsung/pics/students_lt.bmp'
DATASET = './dataset.txt'

QUANTIZE_ON = False
ACC_ANALYSIS_ON = False

OBJ_THRESH = 0.4
NMS_THRESH = 0.45
# IMG_SIZE = 320

CLASSES = ("head",)

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL,
                         outputs=['onnx::Concat_249',
                                  'onnx::Concat_257',
                                  'cls_output',
                                  'onnx::Concat_250',
                                  'onnx::Concat_258',
                                  'reg_output'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    if ACC_ANALYSIS_ON:
        # Accuracy analysis
        print('--> Accuracy analysis')
        ret = rknn.accuracy_analysis(inputs=[IMG_PATH], output_dir='./snapshot')
        if ret != 0:
            print('Accuracy analysis failed!')
            exit(ret)
        print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    post_process_face(outputs, img_1, base_idx=0)

    # show output
    cv2.imshow("post process result", img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('/home/manu/tmp/result.jpg', img_1)

    rknn.release()
