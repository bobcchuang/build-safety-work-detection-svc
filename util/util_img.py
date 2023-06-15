#!/usr/bin/env python
# coding: utf-8

import cv2, json
import numpy as np
from shapely.geometry import Polygon, mapping


def draw_label(img, label_info, color, draw_label=True):
    # draw label contour
    pred_class = label_info[0]
    pt = label_info[1]
    score = label_info[2]

    if score == 0:
        label = pred_class
    else:
        label = '{}_{}'.format(pred_class, score)

    pt = [[int(i[1]), int(i[0])] for i in pt]
    point = np.asarray(pt, dtype=np.int32).reshape((-1, 1, 2))
    img = cv2.polylines(img,
                        pts=[point],
                        isClosed=True,
                        color=color,
                        thickness=2)

    if draw_label:
        # draw label box
        left, top, w, h = cv2.boundingRect(point)
        if len(label) > 15:
            if left > (img.shape[1] - 150):
                left -= 50

        try:
            cv2.rectangle(img, (left + 5, top - 5),
                          (left + 8 + len(label) * 7, top + 7), color, -1)
            cv2.putText(img, label, (left + 8, top + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 1, cv2.LINE_AA)
        except:
            cv2.rectangle(img, (left, top), (right, top), color, -1)
            cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def cal_overlap_ratio(p_main, p_sub, overlap_type, ratio_th, img):
    draw_pt = []

    polygon_main = Polygon(p_main)
    polygon_sub = Polygon(p_sub)
    intersect = polygon_main.intersection(polygon_sub)
    ratio = 0
    if overlap_type == "main":
        ratio = round(intersect.area / polygon_main.area, 3)
        draw_pt.append(p_main)
    elif overlap_type == "sub":
        ratio = round(intersect.area / polygon_sub.area, 3)
        draw_pt.append(p_sub)
    # elif (overlap_type == "both"):
    else:
        ratio = round(intersect.area / (polygon_main.union(polygon_sub).area), 3)
        draw_pt.append(p_main)
        draw_pt.append(p_sub)

    judge = (ratio > ratio_th)

    if judge:

        # 畫父母區
        for pt in draw_pt:
            pt = [[int(i[1]), int(i[0])] for i in pt]
            point = np.asarray(pt, dtype=np.int32).reshape((-1, 1, 2))
            img = cv2.polylines(img, pts=[point], isClosed=True, color=(0, 0, 255), thickness=2)

        # 畫交集區
        inter_pt = mapping(intersect)['coordinates'][0]
        pt = [[int(i[1]), int(i[0])] for i in inter_pt]
        point = np.asarray(pt, dtype=np.int32).reshape((-1, 1, 2))
        img = cv2.polylines(img,
                            pts=[point],
                            isClosed=True,
                            color=(0, 0, 255),
                            thickness=2)
        # 畫透明mask
        zeros = img.copy()
        mask = cv2.fillPoly(zeros, [point], color=(0, 0, 255))
        cv2.addWeighted(mask, 0.5, img, 1 - 0.5, 0, img)

        # 填字
        c_y, c_x = intersect.centroid.coords[0]
        c_y, c_x = int(c_y), int(c_x)
        label = '{}:{}'.format(overlap_type, ratio)

        left, top = c_x - 80, c_y + 40
        img = cv2.line(img, (c_x, c_y), (c_x - 40, c_y + 40), (0, 0, 255), 2)

        try:
            cv2.rectangle(img, (left + 5, top - 5),
                          (left + 15 + len(label) * 7, top + 7), (0, 0, 255), -1)
            cv2.putText(img, label, (left + 8, top + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 1, cv2.LINE_AA)
        except:
            cv2.rectangle(img, (c_x, c_y), (c_x + 15 + len(label) * 7, c_y + 12), (0, 0, 255), -1)
            cv2.putText(img, label, (c_y + 8, c_x + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return judge, img


def cal_offset_judge(p_main, p_sub, offset_type, img):
    # offset_x 負:左邊 正:右邊
    # offset_y 負:上邊 正:下邊
    # 上下左右->UDLR

    main_y, main_x = Polygon(p_main).centroid.coords[0]
    sub_y, sub_x = Polygon(p_sub).centroid.coords[0]

    # 計算物件之間的方向性
    d_y = ""
    d_x = ""
    diff_y = sub_y - main_y
    if diff_y > 0:
        d_y = "D"
    else:
        d_y = "U"

    diff_x = sub_x - main_x
    if diff_x > 0:
        d_x = "R"
    else:
        d_x = "L"

    if (d_y not in offset_type) or (d_x not in offset_type):
        return img

    draw_pt = [p_main, p_sub]
    # 畫父母區
    for pt in draw_pt:
        pt = [[int(i[1]), int(i[0])] for i in pt]
        point = np.asarray(pt, dtype=np.int32).reshape((-1, 1, 2))
        # 畫透明mask
        zeros = img.copy()
        mask = cv2.fillPoly(zeros, [point], color=(255, 0, 255))
        cv2.addWeighted(mask, 0.2, img, 1 - 0.2, 0, img)
        img = cv2.polylines(img, pts=[point], isClosed=True, color=(255, 0, 255), thickness=1)

    img = cv2.line(img, (int(main_x), int(main_y)), (int(sub_x), int(main_y)), (0, 0, 255), 2)
    img = cv2.line(img, (int(sub_x), int(main_y)), (int(sub_x), int(sub_y)), (0, 0, 255), 2)

    text = 'type: ' + d_y + d_x + '#' + 'X: ' + str(int(abs(diff_x))) + '#' + 'Y: ' + str(int(abs(diff_y)))
    cy, cx = int((main_y + sub_y) * 0.5), int((main_x + sub_x) * 0.5)
    cx = cx + 30
    cy = cy + 10

    if cx > (img.shape[1] - 100):
        cx -= 80

    for i, txt in enumerate(text.split('#')):
        cy = cy + 15
        cv2.rectangle(img, (cx, cy - 10), (cx + len(txt) * 7, cy + 4), (0, 255, 0), -1)
        img = cv2.putText(img, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)

    # cv2.rectangle(img, (left + 5, top - 5), (left + 15 + len(label) * 7, top + 7), (0,0,255), -1)

    # cv2.putText(img, label, (left + 8, top + 6), cv2.FONT_HERSHEY_SIMPLEX,
    #            0.4, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(img, label, (left + 8, top + 6), cv2.FONT_HERSHEY_SIMPLEX,
    #            0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # abs(diff_y) abs(diff_x)

    return img
