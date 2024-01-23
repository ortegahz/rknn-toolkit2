import cv2
import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '/media/manu/data/sdks/sigmastar/Tiramisu_DLS00V010-20220107/ipu/SGS_IPU_SDK_vQ_0.1.0/demos/onnx_yolov5/model.onnx'
RKNN_MODEL = '/home/manu/tmp/model.rknn'

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[127.5, 127.5, 127.5], std_values=[127.5, 127.5, 127.5],
                target_platform='rk3588')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(
        '/media/manu/data/sdks/sigmastar/Tiramisu_DLS00V010-20220107/ipu/SGS_IPU_SDK_vQ_0.1.0/demos/onnx_yolov5/1540490031567-0.504512.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    # save outputs
    for save_i in range(len(outputs)):
        save_output = outputs[save_i].flatten()
        np.savetxt('/home/manu/tmp/rknn_output_%s.txt' % save_i, save_output,
                   fmt="%f", delimiter="\n")
    print('done')

    rknn.release()
