#!/usr/bin/env python
# coding: utf-8

import cv2, json
import numpy as np
from .AI_object import *
from .util_img import *
import requests
import base64, io
import os
from random import randint


proxies = {'http': "http://TCPXY001:8080",
           'https': "https://TCPXY001:8080"}
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
ao = AI_object()


def SplitPattern(string, count):
    result = ''
    for x in range(len(string)):
        if x % count == 0:
            result += '\n' + string[x]
        else:
            result += string[x]
    return result[1::].split('\n')


def merge_img(img1, img2, tp='R'):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    w1_new = int(h2 / h1 * w1)
    h1_new = h2

    result = np.zeros((h1_new, w1_new + w2, 3), dtype=np.uint8)

    img1 = cv2.resize(img1, (w1_new, h1_new), interpolation=cv2.INTER_AREA)

    if tp == 'R':
        result[::, 0:w1_new, ::] = img1
        result[::, w1_new::, ::] = img2
    else:
        result[::, 0:w2, ::] = img2
        result[::, w2::, ::] = img1

    return result


def get_status_color(label):
    label = label.upper()
    color = (255, 255, 255)

    if 'NG' == label:
        color = (0, 0, 255)
    elif 'OFF' == label:
        color = (100, 100, 100)
    elif 'ON' == label:
        color = (0, 255, 0)
        
    elif 'WARN' == label:
        color = (0, 255, 255)
        
    elif 'NGTEXT' == label:
        color = (255, 255, 255)
    elif 'OFFTEXT' == label:
        color = (0, 0, 0)
    elif 'ONTEXT' == label:
        color = (0, 0, 0)

    return color


def base64tocv2(b64_string):
    img_bytes = base64.b64decode(b64_string.encode('utf-8'))
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def cv2tobase64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = base64.b64encode(base64_str).decode("utf8")
    return base64_str


# # SERVER ALARM Response
def alarm_code(json, code):
    
    return_str = ''
    
    if code == '0000':
        return_str = 'success!'
    elif code == '0001':
        return_str = 'Model name not exist!'
    elif code == '0002':
        return_str = 'Get model list fail!'
    elif code == '0003':
        return_str = 'Get label list fail!'
        
    json["return_code"] = code
    json['return_text'] = return_str
    return json


# # 物件確認與Judge Info整理
# object check info
def check_obj_config(label_info, config, img):
    img_size = img.shape[:-1]
    ao = AI_object()
    
    for obj_info_list in config['object_info']:
        t = 0
        while(1):
            try:
                obj_info_temp = obj_info_list['lv'+str(t)]
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


# draw tool function
# draw alarm info
def draw_alarm_info_(img_alarm, draw_alarm_list, trigger_label_list, fps=1):
    y = 180
    for i in range(len(draw_alarm_list)):
        trigger_label = [l for l, v in trigger_label_list[i].items()]
        img_alarm = cv2.putText(img_alarm, "Trigger" + str(i+1), (50, y),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        if draw_alarm_list[i]['trigger'][3]:
            img_alarm = cv2.rectangle(img_alarm, (205, y-20), (298, y+5), get_status_color('ON'), -1)
            img_alarm = cv2.putText(img_alarm, str(timedelta(seconds=int(draw_alarm_list[i]['trigger'][1] / fps))),
                                    (210,y), cv2.FONT_HERSHEY_COMPLEX,0.7, get_status_color('ONTEXT'), 1, cv2.LINE_AA)

        else:
            img_alarm = cv2.rectangle(img_alarm,(205,y-20),(285,y+5),get_status_color('OFF'),-1)

        y += 30
        for t in trigger_label:
            img_alarm = cv2.putText(img_alarm, t, (50,y), cv2.FONT_HERSHEY_COMPLEX,0.5, (0,0,0), 1, cv2.LINE_AA)

            if trigger_label_list[i][t][3]:
                img_alarm = cv2.rectangle(img_alarm, (205, y-10), (275, y+5), get_status_color('ON'), -1)
                img_alarm = cv2.putText(img_alarm, str(timedelta(seconds=int(trigger_label_list[i][t][0]/fps))),
                                        (210, y+2), cv2.FONT_HERSHEY_COMPLEX, 0.5, get_status_color('ONTEXT'),
                                        1, cv2.LINE_AA)
            else:
                img_alarm = cv2.rectangle(img_alarm,(205, y-20),(285, y+5),get_status_color('OFF'),-1)
            y += 25
        y += 5

        for t in draw_alarm_list[i]['alarm_text']:
            text_list = SplitPattern(t[0], 27)
            # text_list = SplitPattern('someone didnt wear the helmet and ng occur !!',27)
            for text in text_list:
                img_alarm = cv2.putText(img_alarm, text, (50, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1,
                                        cv2.LINE_AA)
                y += 25

            if t[3]:
                if t[1] >= t[4]:
                    img_alarm = cv2.rectangle(img_alarm,(205, y-20),(298, y+5),get_status_color('NG'),-1)
                    img_alarm = cv2.putText(img_alarm, str(timedelta(seconds = int(t[1]/fps))), (210,y),
                                            cv2.FONT_HERSHEY_COMPLEX,0.7, get_status_color('NGTEXT'), 1, cv2.LINE_AA)
                else:
                    
                    # img_alarm = cv2.rectangle(img_alarm,(205, y-20),(298, y+5),get_status_color('WARN'),-1)
                    c = (int(100 - t[1] * (100 / t[4])) , int(100 - t[1] * (100 / t[4])), int(100 + t[1] * (150 / t[4])))
                    img_alarm = cv2.rectangle(img_alarm,(205, y-20),(298, y+5),c,-1)                    
                    img_alarm = cv2.putText(img_alarm, str(timedelta(seconds = int((t[4] - t[1])/fps))), (210,y),
                                            cv2.FONT_HERSHEY_COMPLEX,0.7, get_status_color('ONTEXT'), 1, cv2.LINE_AA)
            else:
                img_alarm = cv2.rectangle(img_alarm,(205, y-20),(298, y+5),get_status_color('OFF'),-1)

            y+=25
        y += 30
    return img_alarm


def draw_obj_info_(img_obj, draw_label_list, config, fps=1):
    y = 180
    for lb in [l for l, v in draw_label_list.items()]:
        # if lb
        img_obj = cv2.putText(img_obj, lb, (40, y), cv2.FONT_HERSHEY_COMPLEX,
                              0.6, config['label_color'][lb][0], 1, cv2.LINE_AA)
        if len(lb) > 12:
            y += 25
        img_obj = cv2.putText(img_obj, str(draw_label_list[lb][0]), (170, y),
                              cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        img_obj = cv2.putText(img_obj, str(timedelta(seconds=int(draw_label_list[lb][1]/fps))),
                              (240, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # draw_label_list[lb]
        y += 40
    return img_obj


def draw_label_list_(label_info, img, color_list={}, overlap=True, score_spec=0, focus_obj=''):
    
    if overlap:
        color_list = {}
        for l in label_info:
            if focus_obj == '':
                try:
                    color = color_list[l[0]]
                except:
                    color = (randint(60, 200), randint(60, 200), randint(60, 200))
                    color_list[l[0]] = color
                if float(l[2]) > score_spec:
                    draw_label(img, l, color)
            else:
                if (l[0] == focus_obj) and (float(l[2]) > score_spec):
                    draw_label(img, l, (0, 0, 255))
                    
    else:
        label_info_draw = []
        for i in range(len(label_info)):
            if len([j for j in label_info[i::] if label_info[i][1] == j[1]]) == 1:
                label_info_draw.append(label_info[i])
        for l in label_info_draw:
            try:
                color = color_list[l[0]][0]
            except:
                color = (36, 146, 255)
            
            draw_label(img, l, color)
    
    return img


def init_result_predict(config, json_judge):
    fps = float(config['basic_setting']['inference_fps'])
    # print(len(json_judge))
    # print(json_judge)
    if len(json_judge) > 0:
        trigger_label_list = json_judge['trigger_label_list']
        alarm_label_list = json_judge['alarm_label_list']
        draw_label_list = json_judge['draw_label_list']
        draw_alarm_list = json_judge['draw_alarm_list']
        box_label = json_judge['box_label']
        
    else:
        trigger_label_list = []
        alarm_label_list = []
        draw_label_list = {}
        draw_alarm_list = []

        for judge_info_config in config['judge_info']:

            # trigger set
            trigger_label = {}
            for trigger_set_conifg in judge_info_config['trigger_set']:
                check_label = [l for l in trigger_label if trigger_set_conifg['object_name'] in l]
                if len(check_label) == 0:
                    trigger_label[trigger_set_conifg["object_name"]] = [0, 0, [], False]

            # alarm set
            alarm_label = {}
            draw_alarm = []
            for alarm_set_conifg in judge_info_config['alarm_set']:
                for alarm_object in alarm_set_conifg['alarm_object_list']:
                    check_label = [l for l in alarm_label if alarm_object['object_name'] in l]
                    if len(check_label) == 0:
                        alarm_label[alarm_object["object_name"]] = [0, 0, [], False]
                draw_alarm.append([alarm_set_conifg['alarm_text'], 0, 0, False,
                                   alarm_set_conifg['alarm_occur_times'] * fps])

            # if len(trigger_label) > 0:
            trigger_label_list.append(trigger_label)

            if len(alarm_label) > 0:
                alarm_label_list.append(alarm_label)
            draw_alarm_list.append({"alarm_text": draw_alarm, "trigger": ["", 0, 0, False]})

        # draw label

        for i in [k for k, v in config['label_color'].items() if v[1]]:
            draw_label_list[i] = [0, 0, 0, False]

        box_label = [k for k, v in config['label_color'].items()]

    return trigger_label_list, alarm_label_list, draw_label_list, draw_alarm_list, box_label


def result_predict(Frame, label_info, config, json_judge, bypass=1):
    trigger_label_list, alarm_label_list, draw_label_list, draw_alarm_list, box_label = init_result_predict(config,
                                                                                                            json_judge)
    fps = float(config['basic_setting']['inference_fps'])
    
    if bypass == 0:
        # alarm 檢查
        for i in range(len(config['judge_info'])):
            alarm_text = []

            trigger_label = trigger_label_list[i]
            try:
                alarm_label = alarm_label_list[i]
            except:
                alarm_label = []
            judge_info_config = config['judge_info'][i]
            draw_alarm_temp = draw_alarm_list[i]

            judge_trigger = True

            # 檢查Trigger物件
            for trigger_config in judge_info_config['trigger_set']:
                trigger_temp = check_judge_info(trigger_label, label_info, trigger_config, fps)
                if not trigger_temp:
                    judge_trigger = False

            # 檢查Alarm物件
            for alarm_config in judge_info_config['alarm_set']:
                alarm = True
                for alarm_object in alarm_config['alarm_object_list']:
                    alarm_temp = check_judge_info(alarm_label, label_info, alarm_object, fps)
                    if not alarm_temp:
                        alarm = False

                if judge_trigger and alarm:
                    alarm_text.append(alarm_config['alarm_text'])

            # 檢查繪圖用的物件(Alarm/Trigger)
            for alarm in draw_alarm_temp['alarm_text']:
                if alarm[0] in alarm_text:
                    alarm[1] += 1
                    alarm[2] = 0
                    alarm[3] = True
                else:
                    alarm[1] = 0
                    alarm[2] += 1
                    alarm[3] = False

            trigger = draw_alarm_temp['trigger']
            if judge_trigger:
                trigger[1] += 1
                trigger[2] = 0
                trigger[3] = True
            else:
                trigger[1] = 0
                trigger[2] += 1
                trigger[3] = False

        # object info 繪圖用物件統計
        for lb_name in [k for k, v in draw_label_list.items()]:
            count = len([k for k in label_info if k[0] == lb_name])
            if count > 0:
                draw_label_list[lb_name][0] = count
                draw_label_list[lb_name][1] += 1
                draw_label_list[lb_name][2] = 0
                draw_label_list[lb_name][3] = True
            else:
                draw_label_list[lb_name][0] = 0
                draw_label_list[lb_name][1] = 0
                draw_label_list[lb_name][2] += 1
                draw_label_list[lb_name][3] = False

    img_ori = Frame
    img_obj = cv2.imread("backimg/object_info.jpg")
    img_alarm = cv2.imread("backimg/alarm_info.jpg")
    img_alarm = draw_alarm_info_(img_alarm.copy(), draw_alarm_list, trigger_label_list, fps)

    img_obj = draw_obj_info_(img_obj.copy(), draw_label_list, config, fps)
    img_draw = draw_label_list_([i for i in label_info if i[0] in box_label], img_ori.copy(),
                                config['label_color'], overlap=False)
    img_result = merge_img(merge_img(img_draw, img_obj, 'R'), img_alarm, 'L')
    
    judge_json = {'trigger_label_list': trigger_label_list, 'alarm_label_list': alarm_label_list,
                  'draw_label_list': draw_label_list, 'draw_alarm_list': draw_alarm_list, 'box_label': box_label}

    return img_result, judge_json


def predict_(model_name, gpu_api, frame, json_config, ai_judge, by_pass, judge_json):
    
    height, width = frame.shape[:2]
    size = (int(width), int(height))
    
    if width > 1000:
        height, width = height/2, width/2
        size = (int(width), int(height))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    if ai_judge == 1:
        try:
            data = {'image': cv2tobase64(frame), 'model_name': model_name}
            result = requests.post(gpu_api, data=json.dumps(data), headers=headers, proxies=proxies)
            print("AI RESULT", result.text)
            label_info = json.loads(result.text)['label_info']
        except:
            label_info = []
        
        try:
            label_info = check_obj_config(label_info, json_config, frame)
            img_result, judge_json = result_predict(frame.copy(), label_info, json_config, judge_json, by_pass)
        except:
            img_result = frame.copy()
        
    else:
        img_result = frame.copy()

    return img_result, judge_json
    

'''test
judge_json = []
img = cv2.imread('test2.jpg')


with open('scene_config/scene.json') as f:
    judge_config =  json.load(f)
    judge_config['basic_setting']['inference_fps'] = 0.2
    
r , judge_json = predict('PM', "http://10.97.220.170:8085/predict", img, judge_config, 1, 0, judge_json)


r , judge_json = predict('PM', "http://10.97.220.170:8085/predict", img, [], 1, 0, judge_json)

cv2.imshow('r', r)
cv2.waitKey(0)
cv2.destroyAllWindows()

alarm_obj = []
for alarm in judge_json['draw_alarm_list']:
    for tmp in alarm['alarm_text']:
        if tmp[3]:
            alarm_obj.append(tmp[0])

print(alarm_obj)

'''