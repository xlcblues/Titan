import numpy as np
from scipy import ndimage


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _gauss_window(size: int, sigma: float) -> np.ndarray:
    coords = np.arange(size) - size // 2
    window = np.exp(-(coords**2) / (2.0 * sigma**2))
    return window / window.sum()


def ssim(
    a: np.ndarray,
    b: np.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
        b = b[None, ...]
    elif a.ndim == 3 and a.shape[-1] <= 4:
        a = np.moveaxis(a, -1, 0)
        b = np.moveaxis(b, -1, 0)
    window = np.outer(
        _gauss_window(window_size, sigma), _gauss_window(window_size, sigma)
    )
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(a.shape[0]):
        x = a[channel]
        y = b[channel]
        mu_x = ndimage.convolve(x, window, mode="constant", cval=0.0)
        mu_y = ndimage.convolve(y, window, mode="constant", cval=0.0)
        mu_xx = ndimage.convolve(x * x, window, mode="constant", cval=0.0)
        mu_yy = ndimage.convolve(y * y, window, mode="constant", cval=0.0)
        mu_xy = ndimage.convolve(x * y, window, mode="constant", cval=0.0)
        var_x = mu_xx - mu_x * mu_x
        var_y = mu_yy - mu_y * mu_y
        cov_xy = mu_xy - mu_x * mu_y
        value = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2) + 1e-12
        )
        scores.append(float(value.mean()))
    return float(np.mean(scores))


def lpips_available() -> bool:
    try:
        import lpips  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False
