import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from io import BytesIO
import base64
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.ldmv2.util import instantiate_from_config
from ldm.ldmv2.models.diffusion.ddim import DDIMSampler
from ldm.ldmv2.models.diffusion.plms import PLMSSampler
from scripts.optimUtils import split_weighted_subprompts, logger



import celery

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load("models/ldm/stable-diffusion/" + ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def im_2_b64(image):
    print("Encoding")
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    print("Completed encoding: "+img_str[:10])
    return img_str


def preload_model(args, celery_task:celery.Task == None):
    opt = dotdict(args)

    config = opt.config
    if (celery_task):
        celery_task.update_state(state="PROGRESS", meta={"status": "Loading model..."})
        
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    return (model, sampler)


def main(main_module=False, mod_args:dict={}, celery_task:celery.Task=None, model_pkg:tuple=()):
    opt = dotdict(mod_args)
        
        
    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    model, sampler = model_pkg

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=opt.device)

    if (celery_task):
        celery_task.update_state(state="PROGRESS", meta={"status": "Opening torch"})
        
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    seeds = ""
    final_image_paths = []
    with torch.no_grad(), \
        precision_scope("cuda"), \
            model.ema_scope():
                tic = time.time()

                for n in trange(opt.n_iter, desc="Sampling"):
                    print("SAMPLE START")
                    if (celery_task):
                        celery_task.update_state(state="PROGRESS", meta={"progress": n /opt.n_iter, "status": "Sampling step: {0}".format(n)})
                    
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        print("PROMPT START sub-sample")
                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n /opt.n_iter, "status": "Sample {0} substep: Get learned conditions (1/4)".format(n)})
                        
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [opt.negative_prompt])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        
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
                            
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        
                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n /opt.n_iter, "status": "Sample {0} substep: Begin sample (2/4)".format(n)})
                            
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n /opt.n_iter, "status": "Sample {0} substep: Decode sample (3/4)".format(n)})
                            
                        x_samples_ddim = model.decode_first_stage(samples_ddim)

                        if (celery_task):
                            celery_task.update_state(state="PROGRESS", meta={"progress": n /opt.n_iter, "status": "Sample {0} substep: Formatting (4/4)".format(n)})

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
