import cv2
import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '/home/manu/tmp/best_ckpt.onnx'
RKNN_MODEL = '/home/manu/nfs/tmp/install/rknn_yolov5_demo_Linux/model/RK3588/acfree_mh.rknn'
IMG_PATH = '/home/manu/nfs/tmp/install/rknn_yolov5_demo_Linux/model/students_lt.bmp'
DATASET = './dataset.txt'

QUANTIZE_ON = True

OBJ_THRESH = 0.4
NMS_THRESH = 0.45
IMG_SIZE = 320

CLASSES = ('head', 'face')


def post_process_face(outputs, img, color=(0, 0, 255), base_idx=6):
    input0_data = outputs[base_idx + 0]  # 1 x c x 80 x 80
    input1_data = outputs[base_idx + 1]  # 1 x c x 40 x 40
    input2_data = outputs[base_idx + 2]  # 1 x c x 20 x 20
    input3_data = outputs[base_idx + 3]  # 1 x 14 x 80 x 80
    input4_data = outputs[base_idx + 4]  # 1 x 14 x 40 x 40
    input5_data = outputs[base_idx + 5]  # 1 x 14 x 20 x 20

    input0_data_t = np.transpose(input0_data, (2, 3, 0, 1))  # 80 x 80 x 1 x c
    input1_data_t = np.transpose(input1_data, (2, 3, 0, 1))  # 40 x 40 x 1 x c
    input2_data_t = np.transpose(input2_data, (2, 3, 0, 1))  # 20 x 20 x 1 x c
    input3_data_t = np.transpose(input3_data, (2, 3, 0, 1))  # 80 x 80 x 1 x 14
    input4_data_t = np.transpose(input4_data, (2, 3, 0, 1))  # 40 x 40 x 1 x 14
    input5_data_t = np.transpose(input5_data, (2, 3, 0, 1))  # 20 x 20 x 1 x 14

    input3_data_t_b = input3_data_t[:, :, :, :4]  # 80 x 80 x 1 x 4
    input4_data_t_b = input4_data_t[:, :, :, :4]  # 40 x 40 x 1 x 4
    input5_data_t_b = input5_data_t[:, :, :, :4]  # 20 x 20 x 1 x 4
    input3_data_t_kps = input3_data_t[:, :, :, -10:]  # 80 x 80 x 1 x 10
    input4_data_t_kps = input4_data_t[:, :, :, -10:]  # 40 x 40 x 1 x 10
    input5_data_t_kps = input5_data_t[:, :, :, -10:]  # 20 x 20 x 1 x 10

    input6_data_t = np.ones((input0_data_t.shape[0], input0_data_t.shape[1], 1, 1), dtype=np.float32)
    input7_data_t = np.ones((input1_data_t.shape[0], input1_data_t.shape[1], 1, 1), dtype=np.float32)
    input8_data_t = np.ones((input2_data_t.shape[0], input2_data_t.shape[1], 1, 1), dtype=np.float32)

    input_data = list()
    input_data.append(np.concatenate((input3_data_t_kps, input3_data_t_b, input6_data_t, input0_data_t), axis=-1))
    input_data.append(np.concatenate((input4_data_t_kps, input4_data_t_b, input7_data_t, input1_data_t), axis=-1))
    input_data.append(np.concatenate((input5_data_t_kps, input5_data_t_b, input8_data_t, input2_data_t), axis=-1))

    kpss, boxes, classes, scores = acfree_post_process_face(input_data)

    if boxes is not None:
        draw_face(img, kpss, boxes, scores, classes + 1, color=color)


def post_process_head(outputs, img, color=(0, 255, 0)):
    input0_data = outputs[0]  # 1 x c x 80 x 80
    input1_data = outputs[1]  # 1 x c x 40 x 40
    input2_data = outputs[2]  # 1 x c x 20 x 20
    input3_data = outputs[3]  # 1 x 14 x 80 x 80
    input4_data = outputs[4]  # 1 x 14 x 40 x 40
    input5_data = outputs[5]  # 1 x 14 x 20 x 20

    input0_data_t = np.transpose(input0_data, (2, 3, 0, 1))  # 80 x 80 x 1 x c
    input1_data_t = np.transpose(input1_data, (2, 3, 0, 1))  # 40 x 40 x 1 x c
    input2_data_t = np.transpose(input2_data, (2, 3, 0, 1))  # 20 x 20 x 1 x c
    input3_data_t = np.transpose(input3_data, (2, 3, 0, 1))  # 80 x 80 x 1 x 14
    input4_data_t = np.transpose(input4_data, (2, 3, 0, 1))  # 40 x 40 x 1 x 14
    input5_data_t = np.transpose(input5_data, (2, 3, 0, 1))  # 20 x 20 x 1 x 14

    input3_data_t_b = input3_data_t[:, :, :, :4]
    input4_data_t_b = input4_data_t[:, :, :, :4]
    input5_data_t_b = input5_data_t[:, :, :, :4]

    input6_data_t = np.ones((input0_data_t.shape[0], input0_data_t.shape[1], 1, 1), dtype=np.float32)
    input7_data_t = np.ones((input1_data_t.shape[0], input1_data_t.shape[1], 1, 1), dtype=np.float32)
    input8_data_t = np.ones((input2_data_t.shape[0], input2_data_t.shape[1], 1, 1), dtype=np.float32)

    input_data = list()
    input_data.append(np.concatenate((input3_data_t_b, input6_data_t, input0_data_t), axis=-1))
    input_data.append(np.concatenate((input4_data_t_b, input7_data_t, input1_data_t), axis=-1))
    input_data.append(np.concatenate((input5_data_t_b, input8_data_t, input2_data_t), axis=-1))

    boxes, classes, scores = acfree_post_process(input_data)

    if boxes is not None:
        draw(img, boxes, scores, classes, color=color)


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


def process_face(input):
    """

    :param input: shape(h, w, bs, 10 + 4 + 1 + c)
    :return:
    """
    grid_h, grid_w = map(int, input.shape[0:2])

    assert grid_h == grid_w  # TODO

    box_confidence = np.expand_dims(input[..., 10 + 4], axis=-1)

    box_class_probs = sigmoid(input[..., 10 + 5:])

    kpss = input[..., :10]

    box_lt = input[..., 10:12]
    box_rb = input[..., 12:14]

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1)
    row = row.reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((col, row), axis=-1)
    box_x1y1 = grid - box_lt + 0.5
    box_x2y2 = grid + box_rb + 0.5
    box_xy = (box_x2y2 + box_x1y1) / 2
    box_wh = box_x2y2 - box_x1y1
    box_xy *= int(IMG_SIZE / grid_w)
    box_wh *= int(IMG_SIZE / grid_h)

    box = np.concatenate((box_xy, box_wh), axis=-1)

    kpss = (kpss + np.tile(grid + 0.5, 5)) * (IMG_SIZE / grid_w)

    return kpss, box, box_confidence, box_class_probs


def process(input):
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = np.expand_dims(input[..., 4], axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

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
    box_xy *= int(IMG_SIZE / grid_w)
    box_wh *= int(IMG_SIZE / grid_h)

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes_face(kpss, boxes, box_confidences, box_class_probs):
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
    kpss = kpss.reshape(-1, 10)
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    kpss = kpss[_box_pos]
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    kpss = kpss[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    return kpss, boxes, classes, scores


def filter_boxes(boxes, box_confidences, box_class_probs):
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
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    return boxes, classes, scores


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


def acfree_post_process_face(input_data):
    """

    :param input_data: [l80, l40, l20] with [fs, fs, bs, 10 + 4 + 1 + c]
    :return:
    """
    kpss, boxes, classes, scores = [], [], [], []
    for input in input_data:
        k, b, c, s = process_face(input)
        k, b, c, s = filter_boxes_face(k, b, c, s)
        kpss.append(k)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    kpss = np.concatenate(kpss)
    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nkpss, nboxes, nclasses, nscores = [], [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        k = kpss[inds]
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nkpss.append(k[keep])
        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    kpss = np.concatenate(nkpss)
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return kpss, boxes, classes, scores


def acfree_post_process(input_data):
    boxes, classes, scores = [], [], []
    for input in input_data:
        b, c, s = process(input)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw_face(image, kpss, boxes, scores, classes, color=(255, 0, 0)):
    for kps, box, score, cl in zip(kpss, boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), color, 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for i in range(5):
            cv2.circle(image, (int(kps[2 * i]), int(kps[2 * i + 1])), 1, colors[i], -1)


def draw(image, boxes, scores, classes, color=(255, 0, 0)):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), color, 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)


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
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL,
                         outputs=['/detect/cls_preds.0/Conv_output_0',
                                  '/detect/cls_preds.1/Conv_output_0',
                                  '/detect/cls_preds.2/Conv_output_0',
                                  '/detect/reg_preds.0/Conv_output_0',
                                  '/detect/reg_preds.1/Conv_output_0',
                                  '/detect/reg_preds.2/Conv_output_0',
                                  '/detect_face/cls_preds.0/Conv_output_0',
                                  '/detect_face/cls_preds.1/Conv_output_0',
                                  '/detect_face/cls_preds.2/Conv_output_0',
                                  '/detect_face/reg_preds.0/Conv_output_0',
                                  '/detect_face/reg_preds.1/Conv_output_0',
                                  '/detect_face/reg_preds.2/Conv_output_0'])
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

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    post_process_head(outputs, img_1)
    post_process_face(outputs, img_1)

    # show output
    cv2.imshow("post process result", img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('/home/manu/tmp/result.jpg', img_1)

    rknn.release()
