"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    """
    Connect to the MQQT server with the supplied connection information.
    """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Variables for tracking desired statistics
    num_counted_people = 0
    num_people_in_frame = 0
    num_frames_since_detection = 0
    num_frames_with_detection = 0
    durations = []
    
    # We don't need to see a person for at least 5 frames (0.5 secs) to consider them a new person
    relax_time_no_detection = 5
    
    # Indicates if this is an image or a video file
    is_image = False
    is_image = (args.input.split('.')[-1] in ['jpg', 'bpm'])
    
    # Handle web cam
    if args.input == 'CAM':
        args.input = 0
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    # Load Model
    infer_network.load_model(args.model, device='CPU', cpu_extension=args.cpu_extension)
    
    # Handle Input Stream
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Grab the shape of the source 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Get desired network input shape
    input_shape = infer_network.get_input_shape()

    # While there is data
    while cap.isOpened():
            
        # Grab frame
        state, frame = cap.read()
        
        # When there is no more data to read, this will execute (end of stream)
        if state == 0:
            break

        # Order color channels, resize and add extra dimension
        proc_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        proc_frame = np.transpose(proc_frame, (2,0,1))
        proc_frame = proc_frame[np.newaxis, :]

        
        
        # Run NN and wait for result
        infer_network.exec_net(proc_frame)
        infer_network.wait()

        # Get results from the nn (boxes, probabilities, etc...)
        out_nn = infer_network.get_output()

        # Stay only with boxes with high confidence
        detection = False
        num_people_in_frame = 0
        for box in out_nn[0,0]: 
            if box[2]> args.prob_threshold:
                num_people_in_frame += 1
                detection = True
                
                # Add bounding box to the frame
                x_min = int(box[3]* width)
                y_min = int(box[4]* height)
                x_max = int(box[5]* width)
                y_max = int(box[6]* height)
                frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                
                
        # Depending on our previous history we track the desired statistics.
        # We use a 'relaxing time' for disposing of situations where there are people
        # on the image but we momentaryly lost track of them (so that we  don't count them double)
        
        if detection:
            num_frames_with_detection += 1
            
            if num_frames_since_detection > relax_time_no_detection:
                # 'Relax time' has passed without getting a detection (i.e we detected a new people)
                num_counted_people += 1
                client.publish("person", json.dumps({"total": num_counted_people}))
                
            num_frames_since_detection = 0
        else:
            num_frames_since_detection += 1
            
            # After 'relaxation time' without detections we finish counting the duration of the current person
            # (And any further detection will be considered as a new person)
            if num_frames_since_detection == relax_time_no_detection and num_frames_with_detection > relax_time_no_detection:
                durations.append(num_frames_with_detection - relax_time_no_detection)
                avg_duration = np.round((np.mean(durations)/10))
                client.publish("person/duration", json.dumps({"duration": avg_duration}))
                num_frames_with_detection = 0
                
            # We have no detection currently but we are during the relaxation time ( so that the app thinks that we are still seeing the person)
            elif num_frames_since_detection < relax_time_no_detection:
                num_frames_with_detection += 1

        # In each frame we send the current number of people
        client.publish("person", json.dumps({"count": num_people_in_frame}))
    
        # Send annotated frame to the FFMPEG server 
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        
    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

    # Write an output image if `single_image_mode` 
    if is_image:
        cv2.imwrite("output.png", frame)

        

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
