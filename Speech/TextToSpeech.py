#!/usr/bin/env python3
# coding: utf8
# author: Xin Jin (xin.jin12@pactera.com)
import requests
import time
import hashlib
import base64
import json

URL = 'http://api.xfyun.cn/v1/service/v1/tts'  # API URL
AUE = 'raw'  # Audio coding format

def _get_header(app_id, api_key):
  cur_time = str(int(time.time()))
  param = {
    'aue': AUE,
    'auf': 'audio/L16;rate=16000',
    'voice_name': 'xiaoyan',
    'engine_type': 'intp65'
  }
  param_str = json.dumps(param)
  param_utf8 = param_str.encode('utf8')
  param_base64 = base64.b64encode(param_utf8).decode('utf8')
  check_sum = (api_key + cur_time + param_base64).encode('utf8')
  check_sum_md5 = hashlib.md5(check_sum).hexdigest()
  header = {'X-CurTime': cur_time,
            'X-Param': param_base64,
            'X-Appid': app_id,
            'X-CheckSum': check_sum_md5,
            'X-Real-Ip': '127.0.0.1',
            'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
            }
  return header

def _get_body(text):
  data = {'text': text}
  return data

def _write_file(file, content):
  with open(file, 'wb') as f:
    f.write(content)
  f.close()

def text_to_speech(content, prefix, app_id='5b7f49bd',
                   api_key='95855982aa95346883233f1059d18855'):
  r = requests.post(URL, headers=_get_header(app_id, api_key),
                    data=_get_body(content))
  contentType = r.headers['Content-Type']
  if contentType == 'audio/mpeg':
    sid = r.headers['sid']
    if AUE == "raw":
      _write_file(prefix + '.' + sid + '.wav', r.content)
    else:
      _write_file(prefix + '.' + sid + '.mp3', r.content)
    print('success, sid = ' + sid)
  else:
    print(r.text)

if __name__ == '__main__':
  text_to_speech('苟利国家生死以，岂因祸福避趋之', 'audio')