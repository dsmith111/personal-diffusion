import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
import time
from io import BytesIO
import base64
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from random import randint
from scripts.optimUtils import split_weighted_subprompts
import re

import base64
from io import BytesIO
# from scripts.samplers import CompVisDenoiser

import celery

# from samplers import CompVisDenoiser

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

def get_resampling_mode():
    try:
        from PIL import __version__, Image
        major_ver = int(__version__.split('.')[0])
        if major_ver >= 9:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS
    except Exception as ex:
        return 1  # 'Lanczos' irrespective of version.
    
# Load image from base64 string
def load_img(b64, h0, w0):
    print("Decoding")
    image = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h})")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=get_resampling_mode())
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


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

    model,sampler = model_pkg
    model.to(opt.device)
    
    # Load image
    init_image = load_img(opt.init_img, opt.H, opt.W).to(opt.device)
    
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


    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
            
    opt.denoising_strength = min(opt.denoising_strength, 0.9999)
    
    t_enc = int(opt.denoising_strength * opt.ddim_steps)
    if t_enc < 5:
        t_enc = 5
        
    print(f"target t_enc is {t_enc} steps")
    
    
    if (celery_task):
        celery_task.update_state(state="PROGRESS", meta={"status": "Opening torch"})
        
    if opt.precision == "autocast" and opt.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    final_image_paths = []
    with torch.no_grad(), \
        precision_scope("cuda"), \
            model.ema_scope():
                for n in trange(opt.n_iter, desc="Sampling"):
                    print("SAMPLE START")
                    if (celery_task):
                        celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sampling step: {0}".format(n)})
                                
                    for prompts in tqdm(data, desc="data"):

                        sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                        os.makedirs(sample_path, exist_ok=True)
                        base_count = len(os.listdir(sample_path))

                        print("PROMPT START sub-sample")
                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Get learned conditions (1/5)".format(n)})
                        
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [opt.negative_prompt])
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
                                c = torch.add(c, model.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else:
                            c = model.get_learned_conditioning(prompts)


                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Allocate virtual memory (3/5)".format(n)})
                                

                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Begin sampling (4/5)".format(n)})
                            
                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(opt.device))
                        
                        # decode it
                        samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc, )


                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Save samples (5/5)".format(n)})
                            
                    print(samples_ddim.shape)
                    print("saving images")
                    samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples = torch.clamp((samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    for x_sample in x_samples:
                        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        image_path=os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}")
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            image_path
                        )
                        final_image_paths.append(image_path)
                        seeds += str(opt.seed) + ","
                        opt.seed += 1
                        base_count += 1

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
