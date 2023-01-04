import sys
import os

from flask import Flask, request, json, jsonify, url_for, send_file
from worker import celery

import pandas as pd
import json as BaseJson
import ast

import base64
from io import BytesIO

app = Flask(__name__)

last_ten_tasks = []
temp_api_key="lksalsdfjliadfds"

# Make sure response JSON matches local JSON schema
def validate_request(request, path:str):
    with open(path, "r") as arg_file:
        schema_args = BaseJson.loads(arg_file.read())
        arg_file.close()
    for key in schema_args:
        if key not in request:
            return (False, key)
    request["ckpt"] = schema_args["ckpt"].get(request["ckpt"], "model.ckpt")
    acting_model = request["ckpt"]
    if "model2" in request["ckpt"]:
        if "v2" not in request["config"]:
            return (False, "incorrect config file for {} model".format(request["ckpt"]))
    else:
        if "v1" not in request["config"]:
            return (False, "incorrect config file for {} model".format(request["ckpt"]))
    print(f"Using model: {acting_model}")
    return (True, None)

def update_ids(task_id:str):
    global last_ten_tasks
    last_ten_tasks.append(task_id)
    # If last_ten_tasks > 10 remove oldest
    if len(last_ten_tasks) > 10:
        last_ten_tasks.pop(0)
        

# Handle responding to client
def send_response(taskid, request_args):
    if request_args["synchronous_request"]:
        result = celery.AsyncResult(taskid)
        result.get()
        if result.state == "SUCCESS":
            if result.result.get("return_base64", True):
                return jsonify({
                        'State': result.state, 
                        'Size': [result.result["width"], result.result["height"]], 
                        'Format': result.result["format"],
                        'Image': result.result["base64"]
                })
            else:
                image = base64.b64decode(result.result["base64"][0])
                return send_file(BytesIO(image), mimetype="image/"+result.result["format"])
        else:
            return jsonify(result.result)
    else:
        # Return URL to check status
        return jsonify({'active_model':request_args["ckpt"], 'Location': url_for('taskstatus',task_id=taskid)}), 202
       
       
# Check async task
@app.route('/status/<task_id>', methods=['GET'])
def taskstatus(task_id):
    task = celery.AsyncResult(task_id)
    if task.state == "PENDING":
        response = {
            'State': 'Unknown',
            'Status': 'This task ID is not recognized. The task is either waiting to be picked up or failed to be sent',
            'Progress': None
            
        } 
    elif task.state == 'PROGRESS':
        response = {
            'State': task.state,
            'Status': task.result.get('status', ''),
            'Progress': task.result.get('progress', 0)
            
        }
    elif task.state == 'FAILURE':
        if task.info == None:
            response = {
                'State': task.state,
                'Status': "Task failed",
                'Progress': 'N/A'
            }
        else:
            response = task.result
            
    elif task.state == 'SUCCESS': 
        # return base64 string or load image from base64 string and send file
        if task.result.get("return_base64", True):
            response = {
                    'State': task.state, 
                    'Size': [task.result["width"], task.result["height"]], 
                    'Format': task.result["format"],
                    'Image': task.result["base64"]
            }
        else:
            image = base64.b64decode(task.result["base64"][0])
            return send_file(BytesIO(image), mimetype="image/"+task.result["format"])
    else:
        if task.info == None:
            response = {
                'State': task.state,
                'Status': task.state,
                'Progress': 'N/A'
            }
        else:
            response = {
                'State': task.state,
                'Status': task.info.get('status', ''),
                'Progress': task.info.get('progress', ''),
                'Message': "Couldn't determine state of task."
        }
    return jsonify(response)


@app.route('/generate', methods=['POST'])
def Generate():
    global temp_api_key
    content_type = request.headers.get('Content-Type')
    request_args = request.get_json()
    request_template_loc = "./templates/template-default.json"
    
    if request_args is None:
        return jsonify({"error": "No JSON body found"}), 400
    
    if request_args.get("type", "default") == "optimized":
        request_template_loc = "./templates/template-optimized.json"
    
    validate_request_result = validate_request(request_args, request_template_loc)
    if (validate_request_result[0] == False):
        return jsonify({"error": "Check JSON schema. Missing or Bad Key/Value: " + validate_request_result[1]}), 400
    
    if (content_type != 'application/json'):
        return 'Content-Type not supported!'
    
    if (request_args.get("api_key", None) != temp_api_key):
        return jsonify({"error": "Unauthorized"}), 401
    
    task = celery.send_task("tasks.trigger", args=[request_args])
    
    update_ids(task.id)
    return send_response(task.id, request_args)


@app.route('/img2img', methods=['POST'])
def Img2img():
    global temp_api_key
    content_type = request.headers.get('Content-Type')
    request_args = request.get_json()
    request_template_loc = "./templates/template-img2img.json"
    
    if request_args is None:
        return jsonify({"error": "No JSON body found"}), 400
    
    if request_args.get("type", "default") == "optimized":
        request_template_loc = "./templates/template-img2img.json"
    
    validate_request_result = validate_request(request_args, request_template_loc)
    if (validate_request_result[0] == False):
        return jsonify({"error": "Check JSON schema. Found missing key: " + validate_request_result[1]}), 400
    
    if (content_type != 'application/json'):
        return 'Content-Type not supported!'
    
    if (request_args["denoising_strength"]) > 1.0:
        request_args["denoising_strength"] = 1.0
    elif (request_args["denoising_strength"]) <= 0.0:
        request_args["denoising_strength"] = 0.01
        
    # Make sure the image is a valid base64 string
    try:
        base64.b64decode(request_args["init_img"])
    except:
        return jsonify({"error": "init_img is not a valid base64 string"}), 400
    
    
    if (request_args.get("api_key", None) != temp_api_key):
        return jsonify({"error": "Unauthorized"}), 401
    
    task = celery.send_task("tasks.img2img", args=[request_args])
    
    update_ids(task.id)
    return send_response(task.id, request_args)
    

@app.route('/recent', methods=['GET'])
def GetImages():
    global temp_api_key
    request_args = request.get_json()
    if (request_args.get("api_key", None) != temp_api_key):
        return jsonify({"error": "Unauthorized"}), 401
    
    global last_ten_tasks
    finished_tasks = []
    # Get last 10 celery task IDs
    for task in last_ten_tasks:
        task_status = celery.AsyncResult(task)
        if task_status.state == 'SUCCESS':
            task_status_url = url_for('taskstatus',task_id=task)
            finished_tasks.append(task_status_url)
    if len(finished_tasks) == 0:
        return jsonify({
            "Recent_finished_images": "No recently finished images stored in memory!"
        })
    return jsonify({
        "Recent_finished_images": finished_tasks
    })
    


if __name__ == '__main__':
    app.run(debug=True)  # run our Flask app