import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
import itertools
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from scripts.optimUtils import split_weighted_subprompts, logger
from transformers import logging

import base64
from io import BytesIO
# from scripts.samplers import CompVisDenoiser

import celery

# from samplers import CompVisDenoiser
logging.set_verbosity_error()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def im_2_b64(image):
    print("Encoding")
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    print("Completed encoding: "+img_str[:10])
    return img_str


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load("models/ldm/stable-diffusion/" + ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def preload_model(args, celery_task:celery.Task == None):
    opt = dotdict(args)

    config = opt.config

    print("Loading model from config")
    sd = load_model_from_config(f"{opt.ckpt}")
    li, lo = [], []
    print("Splitting model")
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    if (celery_task):
        celery_task.update_state(state="PROGRESS", meta={ "status": "Loading model..."})
        
    print("Loading OmegaConf")
    config = OmegaConf.load(f"{config}")

    print("Instantiate Unet")
    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = opt.unet_bs
    model.cdevice = opt.device
    model.turbo = opt.turbo

    print("Instantiating model CondStage")
    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = opt.device

    print("Instantiating model FirstStage")
    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    print("Half model")
    if opt.device != "cpu" and opt.precision == "autocast":
        model.half()
        modelCS.half()

    return (model, modelCS, modelFS)


def main(main_module=False, mod_args:dict={}, celery_task:celery.Task=None, model_pkg:tuple=()):
    DEFAULT_CKPT = "models/ldm/stable-diffusion-v1/model.ckpt"
    print("Parsing request")
    opt = dotdict(mod_args)
        
    tic = time.time()
    print("Parse request finished. Making dir")
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    if opt.seed == None:
        opt.seed = randint(0, 1000000)
    seed_everything(opt.seed)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=opt.device)

    model, modelCS, modelFS = model_pkg
    
    batch_size = opt.n_samples
    if not opt.from_file:
        assert opt.prompt is not None
        prompt = opt.prompt
        print(f"Using prompt: {prompt}")
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            text = f.read()
            print(f"Using prompt: {text.strip()}")
            data = text.splitlines()
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))

    if (celery_task):
        celery_task.update_state(state="PROGRESS", meta={"status": "Opening torch"})
        
    if opt.precision == "autocast" and opt.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    final_image_paths = []
    with torch.no_grad():

        all_samples = list()
        for n in trange(opt.n_iter, desc="Sampling"):
            print("SAMPLE START")
            if (celery_task):
                celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sampling step: {0}".format(n)})
                        
            for prompts in tqdm(data, desc="data"):

                sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                os.makedirs(sample_path, exist_ok=True)
                base_count = len(os.listdir(sample_path))

                with precision_scope("cuda"):
                    print("PROMPT START sub-sample")
                    if (celery_task):
                        celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Get learned conditions (1/5)".format(n)})
                    
                    modelCS.to(opt.device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [opt.negative_prompt])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    if (celery_task):
                        celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Split and normalize subprompt (2/5)".format(n)})
                    
                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]

                    if (celery_task):
                        celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Allocate virtual memory (3/5)".format(n)})
                            
                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    if (celery_task):
                        celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Begin sampling (4/5)".format(n)})
                        
                    samples_ddim = model.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        seed=opt.seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=start_code,
                        sampler = opt.sampler,
                    )

                    modelFS.to(opt.device)

                    if (celery_task):
                        celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Save samples (5/5)".format(n)})
                        
                    print(samples_ddim.shape)
                    print("saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        image_path=os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            image_path
                        )
                        final_image_paths.append(image_path)
                        seeds += str(opt.seed) + ","
                        opt.seed += 1
                        base_count += 1

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated(device=opt.device) / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0

    print(
        (
            "Samples finished in {0:.2f} minutes and exported to "
            + sample_path
            + "\n Seeds used = "
            + seeds[:-1]
        ).format(time_taken)
    )
    b64_images = []
    for final_path in final_image_paths:
        loaded_image = Image.open(final_path)
        b64img = im_2_b64(loaded_image)
        b64_images.append(b64img)

    return {
        "base64": b64_images,
        "image_path": final_image_paths,
        "width": opt.W,
        "height": opt.H,
        "format": opt.format
        }
    # return str(os.path.join(sample_path))

if __name__ == "__main__":
    main(True)
