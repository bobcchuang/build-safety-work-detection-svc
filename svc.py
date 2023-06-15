#!/usr/bin/env python
# coding: utf-8

import cv2, json
import numpy as np
import base64, io
import os
from util.AI_object import *
from util.detect import *
from PIL import Image
from io import BytesIO

# fastapi package
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import requests
import gc
import copy


def base64tocv2(b64_string):
    img_bytes = base64.b64decode(b64_string.encode('utf-8'))
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def cv2tobase64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = base64.b64encode(base64_str).decode("utf8")
    return base64_str


def bytes_to_cv2image(imgdata):
    cv2img = cv2.cvtColor(np.array(Image.open(BytesIO(imgdata))), cv2.COLOR_RGB2BGR)
    return cv2img


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs2", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    For local js, css swagger in AUO
    :return:
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


class objBase(BaseModel):
    label_info: list
    config: dict

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/check_obj_config/")
def check_obj_config(parameter: objBase = Form(...),
                     file: UploadFile = File(...)):
    t0 = time.time()
    label_info = parameter.label_info
    config = parameter.config
    cv2_img = bytes_to_cv2image(file.file.read())

    img_size = cv2_img.shape[:-1]
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
                    l_result = copy.deepcopy(label_info)
                else:
                    for obj_info in obj_info_temp:
                        label_info = l_result
                        l_result = copy.deepcopy(label_info)
                        result = ao.create_obj(label_info, obj_info, img_size)
                        l_result.extend(result)
                t += 1
            except:
                break
    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    result = {"label_info": l_result, "fps": f_fps}
    return result


class alarmBase(BaseModel):
    label_info: list
    config: dict
    json_judge: dict
    bypass: int
    return_type: int

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/alarm_predict/")
def alarm_predict(parameter: alarmBase = Form(...),
                  file: UploadFile = File(...)):
    t0 = time.time()
    label_info = parameter.label_info
    config = parameter.config
    json_judge = parameter.json_judge
    bypass = parameter.bypass
    return_type = parameter.return_type
    # 0: no image result
    # 1: image result with label
    # 2: image result with label, alarm_info and object_info
    if return_type!=0:
        img_ori = bytes_to_cv2image(file.file.read())
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

    judge_json = {'trigger_label_list': trigger_label_list, 'alarm_label_list': alarm_label_list,
                              'draw_label_list': draw_label_list, 'draw_alarm_list': draw_alarm_list,
                              'box_label': box_label}
    if return_type==0:
        t1 = time.time()
        f_fps = 1.0 / (t1 - t0)
        result = {"img_result": "", "judge_json": judge_json, "fps": f_fps}
        return result

    img_draw = draw_label_list_([i for i in label_info if i[0] in box_label], img_ori.copy(),
                                config['label_color'], overlap=False)
    if return_type==1:
        t1 = time.time()
        f_fps = 1.0 / (t1 - t0)
        result = {"img_result": cv2tobase64(img_draw), "judge_json": judge_json, "fps": f_fps}
        return result

    img_alarm = cv2.imread("backimg/alarm_info.jpg")
    img_alarm = draw_alarm_info_(img_alarm.copy(), draw_alarm_list, trigger_label_list, fps)

    img_obj = cv2.imread("backimg/object_info.jpg")
    img_obj = draw_obj_info_(img_obj.copy(), draw_label_list, config, fps)

    img_result = merge_img(img_alarm, merge_img(img_obj, img_draw, 'L'), 'R')

    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    result = {"img_result": cv2tobase64(img_result), "judge_json": judge_json, "fps": f_fps}
    return result


@app.get("/")
def HelloWorld():
    return {"Hello": "World"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5124))
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)
