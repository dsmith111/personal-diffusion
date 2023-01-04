import torch
from ldm.modules.diffusionmodules.util import extract_into_tensor
import numpy as np

def randn(seed, shape, opt):
    torch.manual_seed(seed)
    if opt.device == 'cpu':
        return torch.randn(shape, device="cpu").to(opt.device)
    return torch.randn(shape, device=opt.device)


def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def create_random_tensors(shape, seeds, opt, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    sampler_noises = None

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = randn(subseed, noise_shape, opt)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = randn(seed, noise_shape, opt)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = randn(seed, shape, opt)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        xs.append(noise)

    x = torch.stack(xs).to(opt.device)
    return x

def txt2img_image_conditioning(x):
    # Dummy zero conditioning if we're not using inpainting model.
    # Still takes up a bit of memory, but no encoder call.
    # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
    return x.new_zeros(x.shape[0], 5, 1, 1)

def img2img_image_conditioning(source_image, latent_image, image_mask=None):
    return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

def to_torch(x, opt):
    return x.clone().detach().to(torch.float32).to(opt.device)


def frange(start, stop, step):
    """Range but handles floats"""
    x = start
    while True:
        if x >= stop:
            return
        yield x
        x += step
        
        
def make_ddim_timesteps(
    ddim_discr_method,
    num_ddim_timesteps,
    num_ddpm_timesteps,
):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps / num_ddim_timesteps
        ddim_timesteps = [int(i) for i in frange(0, num_ddpm_timesteps - 1, c)]
        ddim_timesteps = np.asarray(ddim_timesteps)
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )
    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    # according to the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )

    return sigmas, alphas, alphas_prev


class NoiseSchedule:
    def __init__(
        self,
        model_num_timesteps,
        model_alphas_cumprod,
        ddim_num_steps,
        opt,
        ddim_discretize="uniform",
        ddim_eta=0.0,
    ):
        device = opt.device
        if model_alphas_cumprod.shape[0] != model_num_timesteps:
            raise ValueError("alphas have to be defined for each timestep")
        self.alphas_cumprod = to_torch(model_alphas_cumprod, opt)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(model_alphas_cumprod.cpu()), opt)
        self.sqrt_one_minus_alphas_cumprod = to_torch(
            np.sqrt(1.0 - model_alphas_cumprod.cpu()), opt
        )
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=model_num_timesteps,
        )
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=model_alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
        )
        self.ddim_sigmas = ddim_sigmas.to(torch.float32).to(device)
        self.ddim_alphas = ddim_alphas.to(torch.float32).to(device)
        self.ddim_alphas_prev = ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = (
            np.sqrt(1.0 - ddim_alphas).to(torch.float32).to(device)
        )
        
@torch.no_grad()
def noise_an_image(init_latent, t, schedule, noise=None):
    # fast, but does not allow for exact reconstruction
    # t serves as an index to gather the correct alphas
    t = t.clamp(0, 1000)
    sqrt_alphas_cumprod = torch.sqrt(schedule.ddim_alphas)
    sqrt_one_minus_alphas_cumprod = schedule.ddim_sqrt_one_minus_alphas

    if noise is None:
        noise = torch.randn_like(init_latent, device="cpu").to(init_latent.device)
    return (
        extract_into_tensor(sqrt_alphas_cumprod, t, init_latent.shape) * init_latent
        + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, init_latent.shape)
        * noise
    )