import os
import pickle
import shutil

import cv2
import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '/home/manu/tmp/iter-96000.onnx'
RKNN_MODEL = '/home/manu/nfs/rk3588/install/rknn_yolov5_demo_Linux/model/RK3588/iter-96000.rknn'
DATASET = './dataset_rsn.txt'
ACC_ANALYSIS_DIR_OUT = './snapshot'
ACC_ANALYSIS_DATASET = './dataset_rsn.txt'

QUANTIZE_ON = False
ACC_ANALYSIS_ON = False

IMG_PATH = '/media/manu/samsung/pics/kps.bmp'
DET = np.array([153.53, 231.12, 270.17, 403.95, 0.3091])  # [x, y, w, h, score]
# IMG_PATH = '/home/manu/nfs/rv1126/install/rknn_yolov5_demo/model/player_1280.bmp'
# DET = np.array([825., 679., 111.1, 244.2, 0.92078])  # [x, y, w, h, score]

X_EXTENTION = 0.01 * 9.0
Y_EXTENTION = 0.015 * 9.0
KEYPOINT_NUM = 17
PIXEL_STD = 200
INPUT_SHAPE = (256, 192)  # (height, width)
OUTPUT_SHAPE = (64, 48)
WIDTH_HEIGHT_RATIO = INPUT_SHAPE[1] / INPUT_SHAPE[0]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center, scale, rot, output_size):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    print(src)
    print(dst)
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def bbox_to_center_and_scale(bbox):
    x, y, w, h = bbox

    center = np.zeros(2, dtype=np.float32)
    center[0] = x + w / 2.0
    center[1] = y + h / 2.0

    scale = np.array([w * 1.0 / PIXEL_STD, h * 1.0 / PIXEL_STD],
                     dtype=np.float32)

    return center, scale


def image_alignment(img, det):
    *bbox, score = det
    center, scale = bbox_to_center_and_scale(bbox)
    rotation = 0
    scale[0] *= (1. + X_EXTENTION)
    scale[1] *= (1. + Y_EXTENTION)
    print(scale)
    # fit the ratio
    if scale[0] > WIDTH_HEIGHT_RATIO * scale[1]:
        scale[1] = scale[0] * 1.0 / WIDTH_HEIGHT_RATIO
    else:
        scale[0] = scale[1] * 1.0 * WIDTH_HEIGHT_RATIO
    print(scale)
    trans = get_affine_transform(center, scale, rotation, INPUT_SHAPE)
    print(trans)

    img_wa = cv2.warpAffine(img, trans, (int(INPUT_SHAPE[1]), int(INPUT_SHAPE[0])),
                            flags=cv2.INTER_LINEAR)
    cv2.imwrite('/home/manu/tmp/img_wa.bmp', img_wa)
    np.savetxt('/home/manu/tmp/rknn_output_img_wa.txt', img_wa.flatten(), fmt="%f", delimiter="\n")
    # sys.exit(0)
    return img_wa, center, scale


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)

    # pre-process config
    print('--> Config model')
    rknn.config(  # reorder_channel='2 1 0',  # for acc analysis
        mean_values=[[103.5300, 116.2800, 123.6750]],
        std_values=[[57.3750, 57.1200, 58.3950]],
        # mean_values=[[0., 0., 0.]],
        # std_values=[[1., 1., 1.]],
        optimization_level=3,
        target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['res'])
    # ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['input.4'])
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

    # Accuracy analysis
    if ACC_ANALYSIS_ON:
        dir_out = ACC_ANALYSIS_DIR_OUT
        if os.path.exists(dir_out):
            shutil.rmtree(dir_out)
        print('--> Accuracy analysis')
        rknn.accuracy_analysis(inputs=ACC_ANALYSIS_DATASET)
        print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk1808', device_id='1808')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img, _, _ = image_alignment(img, DET)

    # img = cv2.imread('/media/manu/samsung/pics/kps_align.bmp')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print(f'shape of outputs[0] -- > {outputs[0].shape}')

    # save outputs
    if QUANTIZE_ON:
        for save_i in range(len(outputs)):
            save_output = outputs[save_i].flatten()
            np.savetxt('/home/manu/tmp/rknn_output_rsn_%s.txt' % save_i, save_output,
                       fmt="%f", delimiter="\n")

        with open('/home/manu/tmp/rknn_sim_outputs_rsn.pickle', 'wb') as f:
            pickle.dump(outputs, f)
    else:
        for save_i in range(len(outputs)):
            save_output = outputs[save_i].flatten()
            np.savetxt('/home/manu/tmp/rknn_output_rsn_%s_nq.txt' % save_i, save_output,
                       fmt="%f", delimiter="\n")

        with open('/home/manu/tmp/rknn_sim_outputs_rsn_nq.pickle', 'wb') as f:
            pickle.dump(outputs, f)

    rknn.release()
