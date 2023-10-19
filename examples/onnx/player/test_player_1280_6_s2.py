import copy
import os
import pickle

import cv2
import numpy as np

from test_player_1280_6_s1 import QUANTIZE_ON
from test_player_1280_6_s1 import IMG_PATH

BOX_THRESH = 0.399
NMS_THRESH = 0.3
IMG_SIZE = (1280, 1280)  # (width, height), such as (1280, 736)

CLASSES = ("head",)


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


def process(input, aux):
    grid_h, grid_w = map(int, input.shape[0:2])

    assert grid_w == grid_h
    stride = IMG_SIZE[0] / grid_w

    box_confidence = np.expand_dims(input[..., 4], axis=-1)

    box_class_probs = sigmoid(input[..., 5:6])

    aux_sigmoid = sigmoid(aux.flatten())  # 34000 x 1 x 1

    strides = [8, 16, 32, 64]
    aux_sigmoid_lst = list()
    idx_s = 0
    for s in strides:
        _h, _w = int(IMG_SIZE[0] / s), int(IMG_SIZE[1] / s)
        idx_e = idx_s + int(_h * _w)
        aux_sigmoid_lst.append(aux_sigmoid[idx_s: idx_e].reshape(_h, _w))
        idx_s = idx_e

    # for i, aux_sigmoid_lst_s in enumerate(aux_sigmoid_lst):
    #     np.savetxt('/home/manu/tmp/rknn_outputs_aux_sigmoid_lst_s_%s.txt' % i,
    #                aux_sigmoid_lst_s.flatten(),
    #                fmt="%f", delimiter="\n")

    # indices = np.where(box_class_probs == 0.9171544313430786)
    # print(f'{grid_h} {grid_w} -- > {indices} <{input[..., 5:][indices[0], indices[1], indices[2], indices[3]]}>')
    # print(f'{grid_h} {grid_w} --> {len(box_class_probs[box_class_probs > BOX_THRESH].flatten())}')

    box_lt = input[..., :2]
    box_rb = input[..., 2:4]

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1)
    row = row.reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((col, row), axis=-1)
    box_x1y1 = grid - box_lt + 0.5
    box_x2y2 = grid + box_rb + 0.5
    box_xy = (box_x2y2 + box_x1y1) / 2
    box_wh = box_x2y2 - box_x1y1
    box_xy *= stride
    box_wh *= stride

    box = np.concatenate((box_xy, box_wh), axis=-1)

    pois = np.zeros((grid_h, grid_w, 1, 3))
    for i in range(grid_h):
        for j in range(grid_w):
            box_pick_us = box[i, j, 0, :]
            xyxy_unrs = [box_pick_us[0] - box_pick_us[2] / 2,
                         box_pick_us[1] - box_pick_us[3] / 2,
                         box_pick_us[0] + box_pick_us[2] / 2,
                         box_pick_us[1] + box_pick_us[3] / 2]
            max_phone_ac = [-1, -1, -1, 0.]  # xp, yp, s, conf
            for idx_s, s in enumerate(strides):
                x1, y1, x2, y2 = \
                    int(xyxy_unrs[0] / s), int(xyxy_unrs[1] / s), int(xyxy_unrs[2] / s), int(xyxy_unrs[3] / s),
                feat_psc = copy.deepcopy(aux_sigmoid_lst[idx_s])
                mask = np.zeros_like(feat_psc)
                mask[y1:y2, x1:x2] = 1.
                feat_psc *= mask
                value = np.max(feat_psc.flatten())
                pos = np.argmax(feat_psc.flatten())
                xp, yp = pos % feat_psc.shape[1], pos // feat_psc.shape[1]
                max_phone_ac = [xp, yp, s, value] if value > max_phone_ac[-1] else max_phone_ac
            pois[i, j, 0, 0], pois[i, j, 0, 1], pois[i, j, 0, 2] = \
                max_phone_ac[0] * max_phone_ac[2], max_phone_ac[1] * max_phone_ac[2], max_phone_ac[-1]
            # if i == 46 and j == 54:
            #     print(f'{pois[i, j, 0, :]}')
            #     print(f'{max_phone_ac}')
            #     print(f'{sigmoid(aux.flatten()[int(max_phone_ac[1] * IMG_SIZE[0] / max_phone_ac[2] + max_phone_ac[0])])}')

    return box, box_confidence, box_class_probs, pois


def filter_boxes(boxes, box_confidences, box_class_probs, pois):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.
        pois: ndarray, points of interesting of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
        pois: ndarray, points of interesting for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    pois = pois.reshape(-1, 3)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= BOX_THRESH)
    boxes = boxes[_box_pos]
    pois = pois[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= BOX_THRESH)

    boxes = boxes[_class_pos]
    pois = pois[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    return boxes, classes, scores, pois


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


def pd_post_process(input_data):
    boxes, classes, scores, pois = [], [], [], []
    for input in input_data[:-1]:
        b, c, s, p = process(input, input_data[-1])
        b, c, s, p = filter_boxes(b, c, s, p)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
        pois.append(p)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    pois = np.concatenate(pois)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores, npois = [], [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        p = pois[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
        npois.append(p[keep])
        # nboxes.append(b)
        # nclasses.append(c)
        # nscores.append(s)
        # npois.append(p)

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    pois = np.concatenate(npois)

    return boxes, classes, scores, pois


def draw(image, boxes, scores, classes, pois):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        pois: ndarray, pois of objects.
        all_classes: all classes name.
    """
    for box, score, cl, poi in zip(boxes, scores, classes, pois):
        left, top, right, bottom = box
        poi_x, poi_y, poi_conf = poi[0], poi[1], poi[2]
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (left, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        cv2.circle(image, (int(poi_x), int(poi_y)), 2, (0, 255, 0), 2)
        cv2.putText(image, f'{poi_conf:.2f}', (int(poi_x), int(poi_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        ppc_x, ppc_y = (left + right) / 2, (top + bottom) / 2
        cv2.line(image, (int(ppc_x), int(ppc_y)), (int(poi_x), int(poi_y)), (0, 255, 0), 2)


def main():
    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if QUANTIZE_ON:
        with open('/home/manu/tmp/rknn_sim_outputs.pickle', 'rb') as f:
            outputs = pickle.load(f)
    else:
        with open('/home/manu/tmp/rknn_sim_outputs_nq.pickle', 'rb') as f:
            outputs = pickle.load(f)

    # post process
    input0_data = outputs[0][:, :1, :, :]  # 1 x c x 160 x 160
    input1_data = outputs[1][:, :1, :, :]  # 1 x c x 80 x 80
    input2_data = outputs[2][:, :1, :, :]  # 1 x c x 40 x 40
    input3_data = outputs[3][:, :1, :, :]  # 1 x c x 20 x 20
    input4_data = outputs[0][:, 1:5, :, :]  # 1 x 4 x 160 x 160
    input5_data = outputs[1][:, 1:5, :, :]  # 1 x 4 x 80 x 80
    input6_data = outputs[2][:, 1:5, :, :]  # 1 x 4 x 40 x 40
    input7_data = outputs[3][:, 1:5, :, :]  # 1 x 4 x 20 x 20
    input8_data = outputs[4]  # 1 x 1 x 34000

    input0_data_t = np.transpose(input0_data, (2, 3, 0, 1))  # 160 x 160 x 1 x c
    input1_data_t = np.transpose(input1_data, (2, 3, 0, 1))  # 80 x 80 x 1 x c
    input2_data_t = np.transpose(input2_data, (2, 3, 0, 1))  # 40 x 40 x 1 x c
    input3_data_t = np.transpose(input3_data, (2, 3, 0, 1))  # 20 x 20 x 1 x c
    input4_data_t = np.transpose(input4_data, (2, 3, 0, 1))  # 160 x 160 x 1 x 4
    input5_data_t = np.transpose(input5_data, (2, 3, 0, 1))  # 80 x 80 x 1 x 4
    input6_data_t = np.transpose(input6_data, (2, 3, 0, 1))  # 40 x 40 x 1 x 4
    input7_data_t = np.transpose(input7_data, (2, 3, 0, 1))  # 20 x 20 x 1 x 4
    input8_data_t = np.transpose(input8_data, (2, 0, 1))  # 34000 x 1 x 1

    inputC0_data_t = np.ones((input0_data_t.shape[0], input0_data_t.shape[1], 1, 1), dtype=np.float32)
    inputC1_data_t = np.ones((input1_data_t.shape[0], input1_data_t.shape[1], 1, 1), dtype=np.float32)
    inputC2_data_t = np.ones((input2_data_t.shape[0], input2_data_t.shape[1], 1, 1), dtype=np.float32)
    inputC3_data_t = np.ones((input3_data_t.shape[0], input3_data_t.shape[1], 1, 1), dtype=np.float32)

    input_data = list()
    input_data.append(
        np.concatenate((input4_data_t, inputC0_data_t, input0_data_t), axis=-1))  # 160 x 160 x 1 x (4+1+c)
    input_data.append(np.concatenate((input5_data_t, inputC1_data_t, input1_data_t), axis=-1))  # 80 x 80 x 1 x (4+1+c)
    input_data.append(np.concatenate((input6_data_t, inputC2_data_t, input2_data_t), axis=-1))  # 40 x 40 x 1 x (4+1+c)
    input_data.append(np.concatenate((input7_data_t, inputC3_data_t, input3_data_t), axis=-1))  # 20 x 20 x 1 x (4+1+c)
    input_data.append(input8_data_t)  # 34000 x 1 x 1

    boxes, classes, scores, pois = pd_post_process(input_data)

    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img_1, boxes, scores, classes, pois)
    # show output
    cv2.imshow("post process result", img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('/home/manu/tmp/rknn_sim_img.bmp', img_1)

    print(f'number of boxes after nms --> {len(boxes)}')

    # save results for comparison
    fn_txt = '/home/manu/tmp/results_rknn.txt'
    if os.path.exists(fn_txt):
        os.remove(fn_txt)
    for box, score, cl, poi in zip(boxes, scores, classes, pois):
        IMG_W, IMG_H = IMG_SIZE[0], IMG_SIZE[1]
        left, top, right, bottom = box
        poi_x, poi_y, poi_conf = poi[0], poi[1], poi[2]
        xc = (left + right) / 2 / IMG_W
        yc = (top + bottom) / 2 / IMG_H
        w = (right - left) / IMG_W
        h = (bottom - top) / IMG_H

        with open(fn_txt, 'a') as f:
            f.write(f'{cl} {xc} {yc} {w} {h} {poi_x} {poi_y} {poi_conf}, {score} \n')


if __name__ == '__main__':
    main()
