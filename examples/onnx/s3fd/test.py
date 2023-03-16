import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = '/home/manu/tmp/s3fd.onnx'
RKNN_MODEL = '/home/manu/nfs/tmp/install/rknn_yolov5_demo_Linux/model/RK3588/s3fd.rknn'
IMG_PATH = '/media/manu/samsung/pics/lishi.bmp'
DATASET = '/home/manu/tmp/dataset.txt'

QUANTIZE_ON = True

OBJ_THRESH = 0.4
NMS_THRESH = 0.4
IMG_SIZE = 640

CLASSES = ("face",)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input):
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 0])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = np.ones_like(box_confidence)

    stride = IMG_SIZE/grid_h

    # debug_idx = 0
    # bbox_preds_debug = np.reshape(input[..., 1:5], (-1, 4))[debug_idx] * stride

    kps_preds = input[..., 5:15] * stride

    box_x1y1 = input[..., 1:3] * stride
    box_x2y2 = input[..., 3:5] * stride

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(2, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(2, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_x1y1 = grid * stride - box_x1y1
    box_x2y2 = grid * stride + box_x2y2
    box_xy = (box_x2y2 + box_x1y1) / 2
    box_wh = box_x2y2 - box_x1y1

    box = np.concatenate((box_xy, box_wh), axis=-1)

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(2, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(2, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    grid = np.tile(grid, (1, 1, 1, 5))

    kps = grid * stride + kps_preds

    # box_confidence_debug = np.reshape(box_x1y1, (-1, 2))[debug_idx]
    # box_x1y1_debug = np.reshape(box_x1y1, (-1, 2))[debug_idx]
    # box_x2y2_debug = np.reshape(box_x2y2, (-1, 2))[debug_idx]

    # box_confidence_debug = np.reshape(box_confidence, (-1, 1))
    # kps_debug = np.reshape(kps, (-1, 10))
    # box_x1y1_debug = np.reshape(box_x1y1, (-1, 2))
    # box_x2y2_debug = np.reshape(box_x2y2, (-1, 2))

    # for i in range(len(box_confidence_debug)):
    #     score = box_confidence_debug[i]
    #     if score > 0.4:
    #         print(f'{score} {np.concatenate((box_x1y1_debug[i], box_x2y2_debug[i]), axis=-1)}')

    # with open('/home/manu/tmp/s3fd_sbox.txt', 'a') as f:
    #     for i in range(len(box_confidence_debug)):
    #         score = box_confidence_debug[i]
    #         if score > 0.4:
    #             f.writelines('%f\n' % score)
    #             tmp = np.concatenate((box_x1y1_debug[i], box_x2y2_debug[i]), axis=-1)
    #             for bbox in tmp:
    #                 f.writelines('%f\n' % bbox)

    # with open('/home/manu/tmp/s3fd_skp.txt', 'a') as f:
    #     for i in range(len(box_confidence_debug)):
    #         score = box_confidence_debug[i]
    #         if score > 0.4:
    #             f.writelines('%f\n' % score)
    #             for kp in kps_debug[i]:
    #                 f.writelines('%f\n' % kp)

    return box, box_confidence, box_class_probs, kps


def filter_boxes(boxes, box_confidences, box_class_probs, kps):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    kps = kps.reshape(-1, 10)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    kps = kps[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    kps = kps[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores, kps


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def s3fd_post_process(input_data):

    boxes, classes, scores, kps = [], [], [], []
    for input in input_data:
        b, c, s, k = process(input)
        b, c, s, k = filter_boxes(b, c, s, k)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
        kps.append(k)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)
    kps = np.concatenate(kps)

    nboxes, nclasses, nscores, nkps = [], [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        k = kps[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
        nkps.append(k[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    kps = np.concatenate(nkps)

    return boxes, classes, scores, kps


def draw(image, boxes, scores, classes, kps):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl, kp in zip(boxes, scores, classes, kps):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        kp = kp.reshape((-1, 2))
        for k in kp:
            k = k.astype(np.int)
            cv2.circle(image, tuple(k), 1, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[128, 128, 128]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL,
                         outputs=['437',
                                  '438',
                                  '439',
                                  '480',
                                  '481',
                                  '482',
                                  '523',
                                  '524',
                                  '525'])
    # compare with official outputs
    # ret = rknn.load_onnx(model=ONNX_MODEL,
    #                      outputs=['score_8',
    #                               'score_16',
    #                               'score_32',
    #                               'bbox_8',
    #                               'bbox_16',
    #                               'bbox_32',
    #                               'kps_8',
    #                               'kps_16',
    #                               'kps_32'])
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

    # # Accuracy analysis
    # print('--> Accuracy analysis')
    # ret = rknn.accuracy_analysis(inputs=[IMG_PATH], output_dir='./snapshot')
    # if ret != 0:
    #     print('Accuracy analysis failed!')
    #     exit(ret)
    # print('done')

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
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    # # save outputs
    # for save_i in range(len(outputs)):
    #     save_output = outputs[save_i].flatten()
    #     np.savetxt('/home/manu/tmp/rknn_output_%s.txt' % save_i, save_output,
    #                fmt="%f", delimiter="\n")

    # # post process
    input0_data = outputs[0]  # 1 x 2 x 80 x 80
    input1_data = outputs[1]  # 1 x 8 x 80 x 80
    input2_data = outputs[2]  # 1 x 20 x 80 x 80
    input3_data = outputs[3]  # 1 x 2 x 40 x 40
    input4_data = outputs[4]  # 1 x 8 x 40 x 40
    input5_data = outputs[5]  # 1 x 20 x 40 x 40
    input6_data = outputs[6]  # 1 x 2 x 20 x 20
    input7_data = outputs[7]  # 1 x 8 x 20 x 20
    input8_data = outputs[8]  # 1 x 20 x 20 x 20

    input0_data_t = np.transpose(input0_data, (2, 3, 0, 1))  # 80 x 80 x 1 x 2
    input1_data_t = np.transpose(input1_data, (2, 3, 0, 1))  # 80 x 80 x 1 x 8
    input2_data_t = np.transpose(input2_data, (2, 3, 0, 1))  # 80 x 80 x 1 x 20
    input3_data_t = np.transpose(input3_data, (2, 3, 0, 1))  # 40 x 40 x 1 x 2
    input4_data_t = np.transpose(input4_data, (2, 3, 0, 1))  # 40 x 40 x 1 x 8
    input5_data_t = np.transpose(input5_data, (2, 3, 0, 1))  # 40 x 40 x 1 x 20
    input6_data_t = np.transpose(input6_data, (2, 3, 0, 1))  # 20 x 20 x 1 x 2
    input7_data_t = np.transpose(input7_data, (2, 3, 0, 1))  # 20 x 20 x 1 x 8
    input8_data_t = np.transpose(input8_data, (2, 3, 0, 1))  # 20 x 20 x 1 x 20

    input0_data_r = np.reshape(input0_data_t, (input0_data_t.shape[0], input0_data_t.shape[1], 2, 1))
    input1_data_r = np.reshape(input1_data_t, (input1_data_t.shape[0], input1_data_t.shape[1], 2, 4))
    input2_data_r = np.reshape(input2_data_t, (input2_data_t.shape[0], input2_data_t.shape[1], 2, 10))
    input3_data_r = np.reshape(input3_data_t, (input3_data_t.shape[0], input3_data_t.shape[1], 2, 1))
    input4_data_r = np.reshape(input4_data_t, (input4_data_t.shape[0], input4_data_t.shape[1], 2, 4))
    input5_data_r = np.reshape(input5_data_t, (input5_data_t.shape[0], input5_data_t.shape[1], 2, 10))
    input6_data_r = np.reshape(input6_data_t, (input6_data_t.shape[0], input6_data_t.shape[1], 2, 1))
    input7_data_r = np.reshape(input7_data_t, (input7_data_t.shape[0], input7_data_t.shape[1], 2, 4))
    input8_data_r = np.reshape(input8_data_t, (input8_data_t.shape[0], input8_data_t.shape[1], 2, 10))

    input_data = list()
    input_data.append(np.concatenate((input0_data_r, input1_data_r, input2_data_r), axis=-1))
    input_data.append(np.concatenate((input3_data_r, input4_data_r, input5_data_r), axis=-1))
    input_data.append(np.concatenate((input6_data_r, input7_data_r, input8_data_r), axis=-1))

    boxes, classes, scores, kps = s3fd_post_process(input_data)

    # show results
    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img_1, boxes, scores, classes, kps)
    cv2.imshow("post process result", img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rknn.release()
