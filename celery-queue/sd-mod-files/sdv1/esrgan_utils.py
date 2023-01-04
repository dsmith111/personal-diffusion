import cv2
from PIL import Image, ImageDraw
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from realesrgan import RealESRGANer

def resrgan2normal(state_dict, nb=23):
    # this code is copied from https://github.com/victorca25/iNNfer
    if "conv_first.weight" in state_dict and "body.0.rdb1.conv1.weight" in state_dict:
        re8x = 0
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if "rdb" in k:
                ori_k = k.replace('body.', 'model.1.sub.')
                ori_k = ori_k.replace('.rdb', '.RDB')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net[f'model.1.sub.{nb}.weight'] = state_dict['conv_body.weight']
        crt_net[f'model.1.sub.{nb}.bias'] = state_dict['conv_body.bias']
        crt_net['model.3.weight'] = state_dict['conv_up1.weight']
        crt_net['model.3.bias'] = state_dict['conv_up1.bias']
        crt_net['model.6.weight'] = state_dict['conv_up2.weight']
        crt_net['model.6.bias'] = state_dict['conv_up2.bias']

        if 'conv_up3.weight' in state_dict:
            # modification supporting: https://github.com/ai-forever/Real-ESRGAN/blob/main/RealESRGAN/rrdbnet_arch.py
            re8x = 3
            crt_net['model.9.weight'] = state_dict['conv_up3.weight']
            crt_net['model.9.bias'] = state_dict['conv_up3.bias']

        crt_net[f'model.{8+re8x}.weight'] = state_dict['conv_hr.weight']
        crt_net[f'model.{8+re8x}.bias'] = state_dict['conv_hr.bias']
        crt_net[f'model.{10+re8x}.weight'] = state_dict['conv_last.weight']
        crt_net[f'model.{10+re8x}.bias'] = state_dict['conv_last.bias']

        state_dict = crt_net
    return state_dict
@torch.no_grad()
def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
    print("Checking shape")
    h_input, w_input = img.shape[0:2]
    # img: numpy
    print("Loading as float32")
    img = img.astype(np.float32)
    print("Checking image bit")
    if np.max(img) > 256:  # 16-bit image
        max_range = 65535
        print('\tInput is a 16-bit image')
    else:
        max_range = 255
        print('\tInput is an 8-bit image')
    img = img / max_range
    print("Checking color scheme")
    if len(img.shape) == 2:  # gray image
        print('\tInput is a gray image')
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        print('\tInput is an RGBA image')
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if alpha_upsampler == 'realesrgan':
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        print("\tInput is an RGB image")
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Finished checking image")

    # ------------------- process image (without the alpha channel) ------------------- #
    print("Preprocessing image")
    self.pre_process(img)
    if self.tile_size > 0:
        self.tile_process()
    else:
        self.process()
    output_img = self.post_process()
    output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    if img_mode == 'L':
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    print("Processing alpha channel if necessary")
    # ------------------- process the alpha channel if necessary ------------------- #
    if img_mode == 'RGBA':
        if alpha_upsampler == 'realesrgan':
            self.pre_process(alpha)
            if self.tile_size > 0:
                self.tile_process()
            else:
                self.process()
            output_alpha = self.post_process()
            output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
        else:  # use the cv2 resize for alpha channel
            h, w = alpha.shape[0:2]
            output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

        # merge the alpha channel
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    # ------------------------------ return ------------------------------ #
    print("Final processing")
    if max_range == 65535:  # 16-bit image
        output = (output_img * 65535.0).round().astype(np.uint16)
    else:
        output = (output_img * 255.0).round().astype(np.uint8)
    
    print("Outscaling if necessary")

    if outscale is not None and outscale != float(self.scale):
        output = cv2.resize(
            output, (
                int(w_input * outscale),
                int(h_input * outscale),
            ), interpolation=cv2.INTER_LANCZOS4)
    print("Returning output and image mode")

    return output, img_mode


def upscale_without_tiling(model, img, device):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255. * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, 'RGB')
    
def infer_params(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    scale2x = 0
    scalemin = 6
    n_uplayer = 0
    plus = False

    for block in list(state_dict):
        parts = block.split(".")
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if (part_num > scalemin
                and parts[0] == "model"
                and parts[2] == "weight"):
                scale2x += 1
            if part_num > n_uplayer:
                n_uplayer = part_num
                out_nc = state_dict[block].shape[0]
        if not plus and "conv1x1" in block:
            plus = True

    nf = state_dict["model.0.weight"].shape[0]
    in_nc = state_dict["model.0.weight"].shape[1]
    out_nc = out_nc
    scale = 2 ** scale2x

    return in_nc, out_nc, nf, nb, plus, scale

