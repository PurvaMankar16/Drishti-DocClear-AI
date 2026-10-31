import cv2
import numpy as np


def sharpen(image: np.ndarray, amount: float = 1.0, sigma: float = 1.5, adaptive: bool = True) -> np.ndarray:
    """Apply adaptive unsharp masking for document sharpening.

    Increased default parameters for document images:
    - amount: 1.0 (vs 0.45) for stronger sharpening
    - sigma: 1.5 (vs 1.1) for broader sharpening radius

    Args:
        image: BGR or grayscale image
        amount: Sharpening strength (0.5-2.0 typical range)
        sigma: Gaussian blur radius for unsharp mask
        adaptive: If True, adjust amount based on image characteristics

    Returns:
        Sharpened image
    """
    if image.size == 0:
        return image.copy()

    # Adaptive amount selection
    if adaptive:
        # Measure current sharpness (Laplacian variance)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Adjust amount based on current sharpness
        # Blurrier images need more sharpening
        if laplacian_var < 50:  # Very blurry
            amount = min(amount * 1.5, 2.0)
        elif laplacian_var < 100:  # Moderately blurry
            amount = min(amount * 1.2, 1.5)
        elif laplacian_var > 200:  # Already sharp
            amount = max(amount * 0.7, 0.5)

    # Apply unsharp masking
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    # Clamp to valid range
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def sharpen_highpass(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Alternative: high-pass filtering for severe blur.

    Can be more aggressive than unsharp mask for very blurry documents.

    Args:
        image: BGR or grayscale image
        kernel_size: Size of high-pass kernel (3 or 5 typical)

    Returns:
        Sharpened image
    """
    if image.size == 0:
        return image.copy()

    # High-pass kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)

    if kernel_size == 5:
        kernel = np.array([[-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, 25, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1]], dtype=np.float32)

    if image.ndim == 3:
        # Apply to each channel
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = cv2.filter2D(image, -1, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)
