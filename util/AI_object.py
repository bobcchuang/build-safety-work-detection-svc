# !/usr/bin/env python
# coding: utf-8


import json
import time
import datetime
from datetime import datetime, timedelta
import cv2
# import numpy as np
import ast
from shapely.geometry import Polygon
import copy


class AI_object():
    def __init__(self):
        pass

    def resize_obj(self, pt, resize_type, resize_x, resize_y):

        if resize_type == "none":
            return pt

        new_pt = []
        c_y, c_x = Polygon(pt).centroid.coords[0]
        for p in pt:
            new_y = p[0]
            new_x = p[1]

            dis_y = resize_y
            dis_x = resize_x
            if resize_type == "relative":
                dis_y = abs(int((p[0] - c_y) * (resize_y - 1)))
                dis_x = abs(int((p[1] - c_x) * (resize_x - 1)))

            if p[0] >= c_y:
                new_y = p[0] + dis_y
            else:
                new_y = p[0] - dis_y

            if p[1] >= c_x:
                new_x = p[1] + dis_x
            else:
                new_x = p[1] - dis_x

            new_pt.append([new_y, new_x])

        return new_pt

    def check_obj_pox(self, pt, img_size):
        h, w = img_size
        for p in pt:
            if p[0] < 0:
                p[0] = 2
            elif p[0] > h:
                p[0] = h - 2

            if p[1] < 0:
                p[1] = 2
            elif p[1] > w:
                p[1] = w - 2
        return pt

    def cal_overlap_ratio(self, p_main, p_sub, overlap_type, ratio_th):
        polygon_main = Polygon(p_main)
        polygon_sub = Polygon(p_sub)
        intersect = polygon_main.intersection(polygon_sub).area
        ratio = 0
        if overlap_type == "main":
            ratio = intersect / polygon_main.area
        elif overlap_type == "sub":
            ratio = intersect / polygon_sub.area
        elif overlap_type == "both":
            ratio = intersect / polygon_main.union(polygon_sub).area

        return ratio > ratio_th

    def cal_offset_judge(self, p_main, p_sub, offset_type, offset_x, offset_y):

        # offset_x 負:左邊 正:右邊
        # offset_y 負:上邊 正:下邊
        # 上下左右->UDLR
        if (offset_type == "UDLR") and (offset_x == 0) and (offset_y == 0):
            return True

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
            return False

        if (abs(diff_y) > offset_y) and (offset_y != 0):
            return False

        if (abs(diff_x) > offset_x) and (offset_x != 0):
            return False

        return True

    def merge_obj(self, label_info, main_config, sub_config_list, obj_box, img_size):

        # [point,score]
        result = []
        result_ng = []
        if obj_box == "main":
            # find all relative main obj
            main_obj_list = [
                i for i in label_info if (i[0] == main_config['object_name']) and (
                        float(i[2]) > float(main_config['score_spec']))
            ]

            # resize main object
            for r in main_obj_list:
                r[1] = self.resize_obj(r[1], main_config['resize_type'],
                                       main_config['size_x'], main_config['size_y'])
                r[1] = self.check_obj_pox(r[1], img_size)

            for main_obj in main_obj_list:

                check_trans = True

                for sub_config in sub_config_list:

                    obj_count = int(sub_config['obj_count'])
                    exist = sub_config['exist']

                    sub_obj_list = [
                        i for i in label_info
                        if (i[0] == sub_config['object_name']) and (
                                float(i[2]) > float(sub_config['score_spec']))
                    ]
                    for sub_obj in sub_obj_list:

                        # judge main obj & sub obj overlap
                        check_overlap = self.cal_overlap_ratio(main_obj[1], sub_obj[1],
                                                               sub_config['overlap_type'],
                                                               float(sub_config['overlap_ratio']))
                        check_offset = self.cal_offset_judge(main_obj[1], sub_obj[1],
                                                             sub_config['offset_type'],
                                                             int(sub_config['offset_x']),
                                                             int(sub_config['offset_y']))

                        if check_overlap and check_offset:
                            obj_count -= 1
                            if obj_count <= 0:
                                break
                    # 不存在 但卻出現過
                    if (not exist) and (obj_count <= 0):
                        check_trans = False
                        break
                    # 存在 未達指定數量
                    if exist and (obj_count > 0):
                        check_trans = False
                        break

                if check_trans:
                    result.append([main_obj[1], main_obj[2]])
                else:
                    result_ng.append([main_obj[1], main_obj[2]])
        else:
            main_obj_list = [
                i for i in label_info if (i[0] == main_config['object_name']) and (
                        float(i[2]) > float(main_config['score_spec']))
            ]
            # resize main object
            for r in main_obj_list:
                r[1] = self.resize_obj(r[1], main_config['resize_type'],
                                       main_config['size_x'], main_config['size_y'])
                r[1] = self.check_obj_pox(r[1], img_size)

            # find all relative sub obj
            sub_obj_list = [
                i for i in label_info if (i[0] == sub_config_list[0]['object_name']) and (
                        float(i[2]) > float(sub_config_list[0]['score_spec']))
            ]

            for sub_obj in sub_obj_list:
                check_trans = True

                obj_count = 1
                exist = True
                for main_obj in main_obj_list:

                    # judge main obj & sub obj overlap
                    check_overlap = self.cal_overlap_ratio(main_obj[1], sub_obj[1],
                                                           sub_config_list[0]['overlap_type'],
                                                           float(sub_config_list[0]['overlap_ratio']))
                    check_offset = self.cal_offset_judge(main_obj[1], sub_obj[1],
                                                         sub_config_list[0]['offset_type'],
                                                         int(sub_config_list[0]['offset_x']),
                                                         int(sub_config_list[0]['offset_y']))

                    if check_overlap and check_offset:
                        obj_count -= 1
                        if obj_count <= 0:
                            break
                # 不存在 但卻出現過
                if (not exist) and (obj_count <= 0):
                    check_trans = False
                    # break
                # 存在 未達指定數量
                if exist and (obj_count > 0):
                    check_trans = False
                    # break

                if check_trans:
                    result.append([sub_obj[1], sub_obj[2]])
                else:
                    result_ng.append([sub_obj[1], sub_obj[2]])

        return result, result_ng

    def main_obj_adjust(self, label_info, main_config, img_size):

        result = []
        # check config whether
        if main_config['resize_type'] == "none":
            return result
        if main_config['size_x'] == 0 and main_config['size_y'] == 0:
            return result

        # get main object need adjust
        main_obj_list = [
            i for i in label_info if (i[0] == main_config['object_name']) and (
                    float(i[2]) > float(main_config['score_spec']))
        ]

        # resize main object
        for r in main_obj_list:
            r[0] += "*"
            r[1] = self.resize_obj(r[1], main_config['resize_type'],
                                   main_config['size_x'], main_config['size_y'])
            # r[1] = self.check_obj_pox(r[1], img_size)
        return main_obj_list

    def create_obj(self, label_info, obj_config, img_size):
        raw_label_info = copy.deepcopy(label_info)
        result = []
        # new label name
        new_label_name = obj_config['label_name']
        main_object = obj_config['main_merge_object'][0]
        obj_box = obj_config['obj_box']

        if main_object['customize']:
            for p_list in main_object['pox_point']:
                result.append([new_label_name, p_list, 1])
        else:
            result_temp, result_temp_ng = self.merge_obj(label_info, main_object,
                                                         obj_config['sub_merge_object'],
                                                         obj_box, img_size)
            result.extend([[new_label_name] + r for r in result_temp])
            if obj_config['negative']:
                result.extend([[obj_config['negative_label_name']] + r
                               for r in result_temp_ng])
            result_main = self.main_obj_adjust(raw_label_info, main_object, img_size)
            result.extend(result_main)
        for r in result:
            r[1] = self.check_obj_pox(r[1], img_size)
        #     r[1] = self.resize_obj(r[1], obj_config['resize_type'],
        #                            obj_config['size_x'], obj_config['size_y'])
        return result


def check_obj_config(label_info, config, img):
    img_size = img.shape[:-1]
    ao = AI_object()

    for obj_info_list in config['object_info']:
        t = 0
        while 1:
            try:
                obj_info_temp = obj_info_list['lv' + str(t)]
                if t == 0:
                    new_label_info = []
                    for lb in label_info:
                        add = True
                        for j in obj_info_temp:
                            if lb[0] == j['label_name'] and float(lb[2]) < float(j['score_spec']):
                                add = False
                        if add:
                            new_label_info.append(lb)
                    label_info = new_label_info
                else:
                    for obj_info in obj_info_temp:
                        result = ao.create_obj(label_info, obj_info, img_size)
                        label_info.extend(result)
                t += 1
            except:
                break
    return label_info


def check_judge_info(judge_info, label_info, judge_config, fps=1):
    obj_name = judge_config["object_name"]
    exist = judge_config["exist"]

    # 目前檢出的物件清單
    detect_label_info = [i for i in label_info if obj_name == i[0]]
    # 已judge的物件資訊
    judge_label_info = [i for i in judge_info if obj_name in i]

    # 檢查資訊
    # 超過count閥值
    add = False
    label_temp = []
    if len(detect_label_info) >= int(judge_config["object_count"]):
        # 檢查offset值
        for detect_label in detect_label_info:
            # 檢查之前判斷
            if ao.cal_offset_judge(judge_info[judge_config["object_name"]][2], detect_label[1],
                                   judge_config["offset_type"], int(judge_config["offset_x"]),
                                   int(judge_config["offset_y"])):
                add = True
                label_temp = detect_label
                break
            else:
                add = False

    judge_result = True
    if (add and exist) or (not add and not exist):
        judge_info[obj_name][0] += 1
        judge_info[obj_name][1] = 0
        try:
            judge_info[obj_name][2] = label_temp[1]
        except:
            judge_info[obj_name][2] = []
        judge_info[obj_name][3] = True
    else:
        judge_info[obj_name][0] = 0
        judge_info[obj_name][1] += 1
        judge_info[obj_name][2] = []
        judge_info[obj_name][3] = False

    if judge_info[obj_name][0] > int(judge_config["occur_times"]) * fps:
        judge_result = True
    else:
        judge_result = False

    return judge_result