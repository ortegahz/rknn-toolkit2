import cv2
import numpy as np

from test import yolov5_post_process, draw, IMG_PATH

if __name__ == '__main__':
    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    outputs = list()
    # outputs.append(np.loadtxt('/home/manu/tmp/rknn_output_0.txt'))
    # outputs.append(np.loadtxt('/home/manu/tmp/rknn_output_1.txt'))
    # outputs.append(np.loadtxt('/home/manu/tmp/rknn_output_2.txt'))
    outputs.append(np.loadtxt('/home/manu/tmp/output.txt'))
    outputs.append(np.loadtxt('/home/manu/tmp/327.txt'))
    outputs.append(np.loadtxt('/home/manu/tmp/328.txt'))

    # post process
    # input0_data = outputs[0]
    # input1_data = outputs[1]
    # input2_data = outputs[2]
    # input0_data = np.transpose(outputs[0].reshape((1, 80, 80, 255)), (0, 3, 1, 2))
    # input1_data = np.transpose(outputs[1].reshape((1, 40, 40, 255)), (0, 3, 1, 2))
    # input2_data = np.transpose(outputs[2].reshape((1, 20, 20, 255)), (0, 3, 1, 2))
    input0_data = outputs[0].reshape((1, 255, 80, 80))
    input1_data = outputs[1].reshape((1, 255, 40, 40))
    input2_data = outputs[2].reshape((1, 255, 20, 20))

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
