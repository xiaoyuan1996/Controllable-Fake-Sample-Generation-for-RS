import numpy as np
from PIL import Image
import model as Model
import argparse
import core.logger as Logger
import core.metrics as Metrics
import os
import data.util as Util
import base64
import json
import cv2
import torch.nn as nn
from flask import Flask, redirect, send_file, request, jsonify,make_response
app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/predict.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-infer', '-i', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_infer', action='store_true')
# parse configs
args = parser.parse_args()
def predict(img_path,new_path,name = "test",args = None):
    result_path = new_path
    sr_path = os.path.join(result_path, 'sr_save')
    hr_path = os.path.join(result_path, 'hr_save')
    print(result_path)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(sr_path, exist_ok=True)
    os.makedirs(hr_path, exist_ok=True)
    img_HR = Image.open(img_path).convert("RGB")
    img_SR = Image.open(img_path).convert("RGB")
    [img_HR, img_SR] = Util.transform_augment(
        [img_HR, img_SR], split='val', min_max=(-1, 1))
    data = {'HR': img_HR, 'SR': img_SR}
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    diffusion.feed_data(data)
    diffusion.test(continous=True)
    visuals = diffusion.get_current_visuals(need_LR=False)
    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
    hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
    return_path = os.path.join(result_path, name + '_sr_process.png')
    Metrics.save_img(
        sr_img, '{}/{}_sr_process.png'.format(result_path, name))
    Metrics.save_img(
        Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_sr.png'.format(sr_path, name))
    Metrics.save_img(
        hr_img, '{}/{}_hr.png'.format(hr_path, name))
    return return_path
@app.route('/convert', methods=['POST'])
def convert():
    stream = bytes("", 'utf-8')
    dst_dir = r"/data/diffusion_data/predict/image"
    img_path = r"/data/diffusion_data/predict/label"
    ff = request.files['file']  
    if ff is None:
        return {
            'code': 500,
            'success': 'false',
            'message': 'File upload failture!'
        } 
    #print(ff)
    os.makedirs(img_path,exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    local_path = os.path.join(img_path, ff.filename)
    print(img_path)
    ff.save(local_path)
    name = ff.filename.split(".")[0]
    file_path = predict(img_path=local_path,new_path=dst_dir,name=name,args =args)
    img_predict = Image.open(file_path).convert("RGB")
    predict_list = np.array(img_predict)
    #print(file_path)
    try:
        send_name = os.path.basename(file_path)
        #return send_file(upload_path)
        f = open(file_path, 'rb')
        file_stream = f.readline()
        # stream = base64.b64encode(stream)
        while (file_stream):
            file_stream = f.readline()
            #print(file_stream)
            stream = file_stream + stream
        f.close()

    except:
        return {
            'code': 500,
            'success': 'false',
            'message': 'Path is invalid!'
        }
    file_str = base64.b64encode(stream)
    #
    file_str = str(file_str, 'utf-8')
    result = {
        'code': 200,
        'success': 'true',
        'file': file_str,
        'file_list': predict_list,
        'file_name': send_name
    }
    image_data = open(file_path, 'rb').read()
    result = json.dumps(result)
    response = make_response(image_data)
    print("finish!")
    return result

@app.route('/hello',methods = ['GET'])
def hello(): 
   print("hello")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='18900')
