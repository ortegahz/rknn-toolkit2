import cv2
import numpy as np

from test import yolov5_post_process, draw, IMG_PATH

if __name__ == '__main__':
    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    outputs = list()
    outputs.append(np.load('./onnx_yolov5_0.npy'))
    outputs.append(np.load('./onnx_yolov5_1.npy'))
    outputs.append(np.load('./onnx_yolov5_2.npy'))

    # post process
    input0_data = outputs[0]
    input1_data = outputs[1]
    input2_data = outputs[2]

    input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
    input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    boxes, classes, scores = yolov5_post_process(input_data)

    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img_1, boxes, scores, classes)
    # show output
    cv2.imshow("post process result", img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
