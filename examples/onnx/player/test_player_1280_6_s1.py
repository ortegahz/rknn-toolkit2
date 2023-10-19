import os
import pickle
import shutil

import cv2
import numpy as np
from rknn.api import RKNN

IMG_PATH = '/media/manu/samsung/pics/player_1280.bmp'
ONNX_MODEL = '/home/manu/tmp/player.onnx'
RKNN_MODEL = '/home/manu/nfs/rk3588/install/rknn_yolov5_demo_Linux/model/RK3588/player.rknn'
# DATASET = '/home/manu/tmp/dataset.txt'
DATASET = './dataset.txt'
ACC_ANALYSIS_DIR_OUT = './snapshot'
ACC_ANALYSIS_DATASET = './dataset.txt'

QUANTIZE_ON = False
ACC_ANALYSIS_ON = False

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                optimization_level=3,
                target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL,
                         outputs=['outputs',
                                  '582',
                                  '583',
                                  '584',
                                  '801'])
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

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export model failed!')
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

    # Accuracy analysis
    if ACC_ANALYSIS_ON:
        dir_out = ACC_ANALYSIS_DIR_OUT
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
        print('--> Accuracy analysis')
        rknn.accuracy_analysis(inputs=ACC_ANALYSIS_DATASET)
        print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    # save outputs
    if QUANTIZE_ON:
        for save_i in range(len(outputs)):
            save_output = outputs[save_i].flatten()
            np.savetxt('/home/manu/tmp/rknn_output_%s.txt' % save_i, save_output,
                       fmt="%f", delimiter="\n")

        with open('/home/manu/tmp/rknn_sim_outputs.pickle', 'wb') as f:
            pickle.dump(outputs, f)
    else:
        for save_i in range(len(outputs)):
            save_output = outputs[save_i].flatten()
            np.savetxt('/home/manu/tmp/rknn_output_%s_nq.txt' % save_i, save_output,
                       fmt="%f", delimiter="\n")

        with open('/home/manu/tmp/rknn_sim_outputs_nq.pickle', 'wb') as f:
            pickle.dump(outputs, f)

    rknn.release()
