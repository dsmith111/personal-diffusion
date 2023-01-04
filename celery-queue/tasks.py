import os
import time
import json as BaseJson
import traceback

import sys
sys.path.insert(0,'stable-diffusion-repo')
# sys.path.insert(0,'stable-diffusion-repo/facialEnhancer')
import scripts.txt2img as txt2img
import scripts.txt2imgv2 as txt2imgv2
import scripts.img2imgv2 as img2imgv2
import scripts.img2img as img2img
# import facialEnhancer.inferenceGfpgan as gfpganInference
from celery import Celery
import time


CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

# global LOADED_MODEL
loaded_model = None
loading_model = False
current_model = ""

celery = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


# Async Task
@celery.task(name="tasks.trigger", bind=True)
def TriggerModel(self, request_args):
    global loaded_model
    global loading_model
    global current_model
    
    self.update_state(state='PROGRESS', meta={'progress': 0, "status": "Starting with prompt: " + request_args["prompt"]})
    if ("stable-diffusion-repo" not in os.getcwd()):
        os.chdir("./stable-diffusion-repo")
    
    try:
        # Load model into worker memory
        while loading_model:
            time.sleep(5)
            print("Checking if model is loaded!")
            
        if loaded_model == None or request_args["ckpt"] != current_model:
            print("Pre-loading Model")
            loading_model = True
            if loaded_model:
                del loaded_model
                loaded_model = None
            if "v2" in request_args["config"]:
                loaded_model = txt2imgv2.preload_model(args=request_args, celery_task=self)
            else:
                loaded_model = txt2img.preload_model(args=request_args, celery_task=self)
            loading_model = False
        else:
            print("Model is already in memory")
        
        current_model = request_args["ckpt"]

        if "v2" in request_args["config"]:
            image_results = txt2imgv2.main(main_module=False, mod_args=request_args, celery_task=self, model_pkg=loaded_model)
        else:
            image_results = txt2img.main(main_module=False, mod_args=request_args, celery_task=self, model_pkg=loaded_model)
        
        # if request_args.get("gfpgan", False):
        #     print("Running GFPGAN")
        #     request_args.update({"input_path": "./"+image_loc, "output_path": "./"+image_loc})
        #     gfpganInference.main(request_args, self)
            
        # self.update_state(state="COMPLETED", meta={"progress": 100, "status": "Completed. Samples located at " + image_loc})
        image_results["return_base64"] = request_args.get("return_base64", False)
        return image_results
    except Exception as e:
        print("FAILED to generate")
        print(e)
        print(traceback.format_exc())
        loading_model = False
        if loaded_model:
            del loaded_model
        loaded_model = None
        raise
    
# Async Img2Img Task
@celery.task(name="tasks.img2img", bind=True)
def TriggerModel(self, request_args):
    global loaded_model
    global loading_model
    global current_model
    
    self.update_state(state='PROGRESS', meta={'progress': 0, "status": "Starting img2img with prompt: " + request_args["prompt"]})
    if ("stable-diffusion-repo" not in os.getcwd()):
        os.chdir("./stable-diffusion-repo")
    
    try:
        # Load model into worker memory
        while loading_model:
            time.sleep(5)
            print("Checking if model is loaded!")
            
        if loaded_model == None or request_args["ckpt"] != current_model:
            print("Pre-loading Model")
            loading_model = True
            if loaded_model:
                del loaded_model
                loaded_model = None
            if "v2" in request_args["config"]:
                loaded_model = txt2imgv2.preload_model(args=request_args, celery_task=self)
            else:
                loaded_model = txt2img.preload_model(args=request_args, celery_task=self)
            loading_model = False
        else:
            print("Model is already in memory")
        
        current_model = request_args["ckpt"]

        if "v2" in request_args["config"]:
            image_results = img2imgv2.main(main_module=False, mod_args=request_args, celery_task=self, model_pkg=loaded_model)
        else:
            image_results = img2img.main(main_module=False, mod_args=request_args, celery_task=self, model_pkg=loaded_model)
        
        image_results["return_base64"] = request_args.get("return_base64", False)
        return image_results
    except Exception as e:
        print("FAILED to generate")
        print(e)
        print(traceback.format_exc())
        loading_model = False
        del loaded_model
        loaded_model = None
        raise
