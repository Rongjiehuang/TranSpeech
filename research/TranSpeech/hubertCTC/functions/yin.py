# adapted from https://github.com/patriceguyot/Yin
# https://github.com/NVIDIA/mellotron/blob/master/yin.py

import numpy as np
import torch


def differenceFunction(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]
    This solution is implemented directly with Numpy fft.


    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv


def cumulativeMeanNormalizedDifferenceFunction(df, N, eps=1e-8):
    """
    Compute cumulative mean normalized difference function (CMND).

    This corresponds to equation (8) in [1]

    :param df: Difference function
    :param N: length of data
    :return: cumulative mean normalized difference function
    :rtype: list
    """
    np.seterr(divide='ignore', invalid='ignore')
    # scipy method, assert df>0 for all element
    cmndf = df[1:] * np.asarray(list(range(1, N))) / (np.cumsum(df[1:]).astype(float) + eps)
    return np.insert(cmndf, 0, 1)


def differenceFunctionBatch(xs: np.ndarray, N, tau_max):
    """numpy backend batch-wise differenceFunction

    Args:
        xs: audio segments, np.ndarray of shape (B x t)
        N:
        tau_max:

    Returns:
        y: dF. np.ndarray of shape (B x tau_max)

    """
    xs = xs.astype(np.float64)
    w = xs.shape[-1]
    tau_max = min(tau_max, w)
    zeros = np.zeros((xs.shape[0], 1))
    x_cumsum = np.concatenate((np.zeros((xs.shape[0], 1)), (xs * xs).cumsum(axis=-1)), axis=-1)  # B x w
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)

    convs = []
    for i in range(xs.shape[0]):
        x = xs[i]
        fc = np.fft.rfft(x, size_pad)
        conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
        convs.append(conv)
    convs = np.asarray(convs)

    y = x_cumsum[:, w:w - tau_max:-1] + x_cumsum[:, w, np.newaxis] - x_cumsum[:, :tau_max] - 2 * convs
    return y


def cumulativeMeanNormalizedDifferenceFunctionBatch(dFs, N, eps=1e-8):
    """numpy backend batch-wise cumulative Mean Normalized Difference Functions

    Args:
        dFs: differenceFunctions. np.ndarray of shape (B x tau_max)
        N:
        eps:

    Returns:
        cMNDFs: np.ndarray of shape (B x tau_max)

    """
    arange = np.asarray(list(range(1, N)))[np.newaxis, ...]
    cumsum = np.cumsum(dFs[:, 1:], axis=-1).astype(float)
    cMNDFs = dFs[:, 1:] * arange / (cumsum + eps)
    cMNDFs = np.concatenate((np.zeros((cMNDFs.shape[0], 1)), cMNDFs), axis=1)
    return cMNDFs


def differenceFunctionTorch(xs: torch.Tensor, N, tau_max) -> torch.Tensor:
    """pytorch backend batch-wise differenceFunction
    has 1e-4 level error with input shape of (32, 22050*1.5)
    Args:
        xs:
        N:
        tau_max:

    Returns:

    """
    xs = xs.double()
    w = xs.shape[-1]
    tau_max = min(tau_max, w)
    zeros = torch.zeros((xs.shape[0], 1))
    x_cumsum = torch.cat(
        (torch.zeros((xs.shape[0], 1), device=xs.device), (xs * xs).cumsum(dim=-1, dtype=torch.double)),
        dim=-1)  # B x w
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)

    fcs = torch.fft.rfft(xs, n=size_pad, dim=-1)
    convs = torch.fft.irfft(fcs * fcs.conj())[:, :tau_max]
    y1 = torch.flip(x_cumsum[:, w - tau_max + 1:w + 1], dims=[-1])
    y = y1 + x_cumsum[:, w, np.newaxis] - x_cumsum[:, :tau_max] - 2 * convs
    return y


def cumulativeMeanNormalizedDifferenceFunctionTorch(dfs: torch.Tensor, N, eps=1e-8) -> torch.Tensor:
    arange = torch.arange(1, N, device=dfs.device, dtype=torch.float64)
    cumsum = torch.cumsum(dfs[:, 1:], dim=-1, dtype=torch.float64).to(dfs.device)

    cmndfs = dfs[:, 1:] * arange / (cumsum + eps)
    cmndfs = torch.cat(
        (torch.ones(cmndfs.shape[0], 1, device=dfs.device), cmndfs),
        dim=-1)
    return cmndfs


if __name__ == '__main__':
    wav = torch.randn(32, int(22050 * 1.5)).cuda()
    wav_numpy = wav.detach().cpu().numpy()
    x = wav_numpy[0]

    w_len = 2048
    w_step = 256
    tau_max = 2048
    W = 2048

    startFrames = list(range(0, x.shape[-1] - w_len, w_step))
    startFrames = np.asarray(startFrames)
    # times = startFrames / sr
    frames = [x[..., t:t + W] for t in startFrames]
    frames = np.asarray(frames)
    frames_torch = torch.from_numpy(frames).cuda()

    cmndfs0 = []
    for idx, frame in enumerate(frames):
        df = differenceFunction(frame, frame.shape[-1], tau_max)
        cmndf = cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
        cmndfs0.append(cmndf)
    cmndfs0 = np.asarray(cmndfs0)

    dfs = differenceFunctionTorch(frames_torch, frames_torch.shape[-1], tau_max)
    cmndfs1 = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, tau_max).detach().cpu().numpy()
    print(cmndfs0.shape, cmndfs1.shape)
    print(np.sum(np.abs(cmndfs0 - cmndfs1)))
