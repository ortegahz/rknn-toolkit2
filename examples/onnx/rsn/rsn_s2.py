import logging
import os
import pickle

import cv2
import numpy as np

from rsn_s1 import QUANTIZE_ON
from rsn_s1 import PIXEL_STD, IMG_PATH, KEYPOINT_NUM, OUTPUT_SHAPE, INPUT_SHAPE, DET, image_alignment

SHIFTS = (0.25,)


def save_results(kps, scores, fn_txt='/home/manu/tmp/results_rknn_sim.txt'):
    if os.path.exists(fn_txt):
        os.remove(fn_txt)
    for kp, score in zip(kps, scores):
        with open(fn_txt, 'a') as f:
            f.write(f'{kp[0]} {kp[1]} {score[0]} \n')
            # f.write(f'{kp[0]} {kp[1]} \n')


def post_process(input_data, center, scale, kernel=5):
    scale *= PIXEL_STD
    score_map = input_data[0].copy()
    score_map = score_map / 255 + 0.5
    kps = np.zeros((KEYPOINT_NUM, 2))
    scores = np.zeros((KEYPOINT_NUM, 1))
    border = 10
    dr = np.zeros((KEYPOINT_NUM, OUTPUT_SHAPE[0] + 2 * border, OUTPUT_SHAPE[1] + 2 * border))
    dr[:, border: -border, border: -border] = input_data[0].copy()
    for w in range(KEYPOINT_NUM):
        # np.savetxt('/home/manu/tmp/rknn_output_dr_%s.txt' % w, dr[w].flatten(), fmt="%f", delimiter="\n")
        dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
        np.savetxt('/home/manu/tmp/rknn_output_dr_%s.txt' % w, dr[w].flatten(), fmt="%f", delimiter="\n")
    for w in range(KEYPOINT_NUM):
        for j in range(len(SHIFTS)):
            if j == 0:
                max_top1 = dr[w].max()
                lb = dr[w].argmax()
                y, x = np.unravel_index(lb, dr[w].shape)
                dr[w, y, x] = 0
                x -= border
                y -= border
            max_top2 = dr[w].max()
            lb = dr[w].argmax()
            py, px = np.unravel_index(lb, dr[w].shape)
            dr[w, py, px] = 0
            print(f'[w] x, y, max_top1, px, py, max_top2 --> {w} {x}, {y}, {max_top1} {px}, {py} {max_top2}')
            px -= border + x
            py -= border + y
            ln = (px ** 2 + py ** 2) ** 0.5
            if ln > 1e-3:
                x += SHIFTS[j] * px / ln
                y += SHIFTS[j] * py / ln
        x = max(0, min(x, OUTPUT_SHAPE[1] - 1))
        y = max(0, min(y, OUTPUT_SHAPE[0] - 1))
        kps[w] = np.array([x * 4 + 2, y * 4 + 2])
        scores[w, 0] = score_map[w, int(round(y) + 1e-9), int(round(x) + 1e-9)]
    save_results(kps, scores, fn_txt='/home/manu/tmp/results_rknn_sim_mid.txt')
    kps[:, 0] = kps[:, 0] / INPUT_SHAPE[1] * scale[0] + center[0] - scale[0] * 0.5
    kps[:, 1] = kps[:, 1] / INPUT_SHAPE[0] * scale[1] + center[1] - scale[1] * 0.5
    return kps, scores


def draw_line(img, p1, p2):
    c = (0, 0, 255)
    if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
        cv2.line(img, tuple(p1), tuple(p2), c, 2)


def draw_results(img, kps, scores):
    joints = kps
    pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
             [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
             [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    colors = np.random.randint(0, 256, (KEYPOINT_NUM, 3)).tolist()

    for i in range(KEYPOINT_NUM):
        cv2.circle(img, tuple(joints[i, :2].astype(int)), 2, tuple(colors[i]), 2)

    for pair in pairs:
        draw_line(img, joints[pair[0] - 1, :2].astype(int), joints[pair[1] - 1, :2].astype(int))


def main():
    # Set inputs
    img = cv2.imread(IMG_PATH)
    _, center, scale = image_alignment(img, DET)

    if QUANTIZE_ON:
        with open('/home/manu/tmp/rknn_sim_outputs_rsn.pickle', 'rb') as f:
            outputs = pickle.load(f)
    else:
        with open('/home/manu/tmp/rknn_sim_outputs_rsn_nq.pickle', 'rb') as f:
            outputs = pickle.load(f)

    # post process
    kps, scores = post_process(outputs[0], center, scale)

    # save results for comparison
    save_results(kps, scores)

    # show output
    draw_results(img, kps, scores)
    cv2.imshow("post process result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('/home/manu/tmp/rknn_sim_img.bmp', img)


if __name__ == '__main__':
    main()
