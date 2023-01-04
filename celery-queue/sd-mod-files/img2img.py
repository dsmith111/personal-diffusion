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
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from scripts.optimUtils import split_weighted_subprompts, logger
from transformers import logging
import pandas as pd

import scripts.esrgan_model_arch as arch
# from basicsr.archs.rrdbnet_arch import RRDBNet
import scripts.esrgan_utils as esrgan_utils
import scripts.processingUtils as p_utils

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
def convert_pil_img(image, h, w):
    # w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=get_resampling_mode())
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def run_realesrgan(opt, b64):
    if type(b64) == str:
        image = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    else:
        # Assume it's a PIL image
        image = b64
    weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real2x.pth")
    state_dict = torch.load(weight_path, map_location='cpu')
    n_state_dict = esrgan_utils.resrgan2normal(state_dict["params_ema"])
    in_nc, out_nc, nf, nb, plus, mscale = esrgan_utils.infer_params(n_state_dict)
    model = arch.RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=mscale, plus=plus)
    model.load_state_dict(n_state_dict)
    model.to(opt.device)

    new_img = esrgan_utils.upscale_without_tiling(model, image, opt.device)
    # Quarter image size
    new_img = new_img.resize((int(new_img.size[0] / 4), int(new_img.size[1] / 4)), get_resampling_mode())
    return new_img
    

def main(main_module=False, mod_args:dict={}, celery_task:celery.Task=None, model_pkg:tuple=()):
    DEFAULT_CKPT = "models/ldm/stable-diffusion-v1/model.ckpt"
    print("Parsing request")
    opt = dotdict(mod_args)
    # Clear cuda memory before starting
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
        
    tic = time.time()
    print("Parse request finished. Making dir")
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    # Check if our steps are a factor of 8
    if (opt.ddim_steps % 8) != 0:
        # Round up to nearest multiple of 8
        opt.ddim_steps = opt.ddim_steps + (8 - (opt.ddim_steps % 8))
    if opt.seed == None:
        opt.seed = randint(0, 1000000)
    seed_everything(opt.seed)

    model, modelCS, modelFS = model_pkg
    
    # Load image
    if opt.upscale:
        init_image = run_realesrgan(opt, opt.init_img)
    else:
        init_image = Image.open(BytesIO(base64.b64decode(opt.init_img))).convert("RGB")
        
    init_image = convert_pil_img(init_image, opt.H, opt.W).to(opt.device)

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

    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    modelFS.to(opt.device)

    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    if (opt.sampler in ["ddim", "plms"]):
        noise = p_utils.create_random_tensors(shape, seeds=[opt.seed], opt=opt)
    else:
        noise = torch.randn_like(init_latent, device="cpu").to(opt.device)
        
    if opt.noise_multiplier != 1.0:
        noise *= opt.noise_multiplier
    shape = init_latent.shape 
    if opt.device != "cpu":
        mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
        modelFS.to("cpu")
        while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
            time.sleep(1)
          
    opt.denoising_strength = min(opt.denoising_strength, 0.9999)
    
    # model.to(opt.device)
    t_enc = int(opt.denoising_strength * opt.ddim_steps)
    if t_enc < 8:
        t_enc = 8
    if t_enc%8 != 0:
        t_enc = t_enc + 8 - (t_enc%8)
    

    print(f"target t_enc is {t_enc} steps")
    if (opt.sampler not in ["ddim", "plms"]):
        schedule = p_utils.NoiseSchedule(
            model_num_timesteps=model.num_timesteps,
            ddim_num_steps=opt.ddim_steps,
            model_alphas_cumprod=model.alphas_cumprod,
            ddim_discretize="uniform",
            opt=opt
        )
        
        init_latent_noised = p_utils.noise_an_image(
            init_latent,
            torch.tensor([t_enc - 1]).to(opt.device),
            schedule=schedule,
            noise=noise,
        )    
    # if opt.device != "cpu":
    #     mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
    #     model.to("cpu")
    #     while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
    #         time.sleep(1)
    
    if (celery_task):
        celery_task.update_state(state="PROGRESS", meta={"status": "Opening torch"})
        
    if opt.precision == "autocast" and opt.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    final_image_paths = []
    image_grid = []
    with torch.no_grad(), \
        precision_scope("cuda"):
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


                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Allocate virtual memory (3/5)".format(n)})
                                
                        if opt.device != "cpu":
                            mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                            modelCS.to("cpu")
                            while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                                time.sleep(1)

                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Begin sampling (4/5)".format(n)})
                          
                        model.to(opt.device)  
                        # encode (scaled latent)
                        
                        model.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
                        if (opt.sampler in ["ddim", "plms"]):
                            z_enc = model.stochastic_encode(
                                x0=init_latent,
                                t=torch.tensor([t_enc] * int(init_latent.shape[0])).to(opt.device),
                                seed=opt.seed,
                                ddim_eta=opt.ddim_eta,
                                ddim_steps=opt.ddim_steps,
                                noise=noise
                            )

                            # decode it
                            samples_ddim = model.img2img_decode(z_enc, c, t_enc, shape, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc, sampling_type=opt.sampler,)
                        else:
                            samples_ddim = model.img2img_decode(init_latent_noised, c, t_enc, shape, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc, sampling_type=opt.sampler, orig_latent=init_latent, orig_steps = opt.ddim_steps, )

                        if opt.device != "cpu":
                            mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                            model.to("cpu")
                            while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                                time.sleep(1)
                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n/opt.n_iter, "status": "Sample {0} substep: Save samples (5/5)".format(n)})
                            
                        print(samples_ddim.shape)
                        print("saving images")
                        modelFS.to(opt.device)

                        for i in range(batch_size):
                            x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            if not opt.skip_grid:
                                image_grid.append(x_sample)
                            x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                            image_path=os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}")
                            if not opt.skip_save:
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

    if not opt.skip_grid:
        # additionally, save as grid
        grid = torch.stack(image_grid, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_rows)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        img = Image.fromarray(grid.astype(np.uint8)
                              )
        img.save(os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}"))
        final_path=[os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}")]
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
        if opt.post_upscale:
            loaded_image = run_realesrgan(opt, loaded_image)
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
