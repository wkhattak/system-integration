#!/usr/bin/env python

import eventlet
eventlet.monkey_patch(socket=True, select=True, time=True)

import eventlet.wsgi
import socketio
import time
from flask import Flask, render_template

from bridge import Bridge
from conf import conf

IMAGE_FREQUENCY = 10

sio = socketio.Server()
app = Flask(__name__)
#msgs = [] ####################################################
# Source: https://github.com/jdleesmiller/CarND-Capstone/commit/33dae9248a73feab9b577dd135116b6575e85788
#Changed to only send the latest message for each topic, rather than queuing out-of-date messages
msgs = {} #########################################################

dbw_enable = False
prev_secs = 0
image_counter = 0

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

def send(topic, data):
    #s = 1
    #msgs.append((topic, data))
    ##sio.emit(topic, data=json.dumps(data), skip_sid=True)
    msgs[topic] = data ###################################################

bridge = Bridge(conf, send)

@sio.on('telemetry')
def telemetry(sid, data):
    global dbw_enable
    if data["dbw_enable"] != dbw_enable:
        dbw_enable = data["dbw_enable"]
        bridge.publish_dbw_status(dbw_enable)
    bridge.publish_odometry(data)
    for i in range(len(msgs)):
        #topic, data = msgs.pop(0)
	topic, data = msgs.popitem() ################################################
        sio.emit(topic, data=data, skip_sid=True)

@sio.on('control')
def control(sid, data):
    bridge.publish_controls(data)

@sio.on('obstacle')
def obstacle(sid, data):
    bridge.publish_obstacles(data)

@sio.on('lidar')
def obstacle(sid, data):
    bridge.publish_lidar(data)

@sio.on('trafficlights')
def trafficlights(sid, data):
    bridge.publish_traffic(data)

@sio.on('image')
def image(sid, data):
    current_secs = int(time.time())
    global prev_secs
    global image_counter
    global IMAGE_FREQUENCY
    if current_secs == prev_secs:
        if image_counter < IMAGE_FREQUENCY:
	    bridge.publish_camera(data)
	    image_counter += 1
    else:
	image_counter = 0
    prev_secs = current_secs

if __name__ == '__main__':

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
