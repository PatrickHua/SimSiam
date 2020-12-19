"""
Only the recent torchvision package has gaussian blur.
So I copy the functions here

"""


import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.nn.functional import conv2d, pad as torch_pad
from typing import Any, List, Sequence, Optional
import numbers
import numpy as np
import torch
from PIL import Image
from typing import Tuple

class GaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return gaussian_blur(img, self.kernel_size, [sigma, sigma])

    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s

@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
    return isinstance(img, Image.Image)
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size
def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2
def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

def _cast_squeeze_in(img: Tensor, req_dtype: torch.dtype) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype != req_dtype:
        need_cast = True
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype
def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        # it is better to round before cast
        img = torch.round(img).to(out_dtype)

    return img
def _get_gaussian_kernel2d(
        kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d
def _gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    """PRIVATE METHOD. Performs Gaussian blurring on the img by given kernel.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be blurred
        kernel_size (sequence of int or int): Kernel size of the Gaussian kernel ``(kx, ky)``.
        sigma (sequence of float or float, optional): Standard deviation of the Gaussian kernel ``(sx, sy)``.

    Returns:
        Tensor: An image that is blurred using gaussian kernel of given parameters
    """
    if not (isinstance(img, torch.Tensor) or _is_tensor_a_torch_image(img)):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, kernel.dtype)

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img

def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Tensor:
    """Performs Gaussian blurring on the img by given kernel.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor): Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.
            In torchscript mode kernel_size as single int is not supported, use a tuple or
            list of length 1: ``[ksize, ]``.
        sigma (sequence of floats or float, optional): Gaussian kernel standard deviation. Can be a
            sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            Default, None. In torchscript mode sigma as single float is
            not supported, use a tuple or list of length 1: ``[sigma, ]``.

    Returns:
        PIL Image or Tensor: Gaussian Blurred version of the image.
    """
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError('kernel_size should be int or a sequence of integers. Got {}'.format(type(kernel_size)))
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError('If kernel_size is a sequence its length should be 2. Got {}'.format(len(kernel_size)))
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError('kernel_size should have odd and positive integers. Got {}'.format(kernel_size))

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError('sigma should be either float or sequence of floats. Got {}'.format(type(sigma)))
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError('If sigma is a sequence, its length should be 2. Got {}'.format(len(sigma)))
    for s in sigma:
        if s <= 0.:
            raise ValueError('sigma should have positive values. Got {}'.format(sigma))

    t_img = img
    if not isinstance(img, torch.Tensor):
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image or Tensor. Got {}'.format(type(img)))

        t_img = to_tensor(img)

    output = _gaussian_blur(t_img, kernel_size, sigma)

    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output)
    return output




# if __name__ == "__main__":
#     gaussian_blur = GaussianBlur(kernel_size=23)