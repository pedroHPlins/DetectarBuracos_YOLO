import cv2
import numpy as np

def convert_to_grayscale(image):
    """Converte uma imagem para escala de cinza."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Aplica um filtro Gaussiano na imagem."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges_canny(image, low_threshold=50, high_threshold=150):
    """Detecta bordas usando o algoritmo Canny."""
    # Canny funciona melhor em escala de cinza
    gray_image = convert_to_grayscale(image)
    blurred_image = apply_gaussian_blur(gray_image) # Reduz ruído antes de Canny
    return cv2.Canny(blurred_image, low_threshold, high_threshold)

def apply_thresholding(image, threshold_value=127, max_value=255, threshold_type=cv2.THRESH_BINARY):
    """Aplica limiarização simples na imagem."""
    gray_image = convert_to_grayscale(image)
    _, thresh_image = cv2.threshold(gray_image, threshold_value, max_value, threshold_type)
    return thresh_image

# Funções de morfologia podem ser adicionadas aqui (erosão, dilatação, etc.)
def apply_erosion(image, kernel_size=(5,5), iterations=1):
    """Aplica erosão morfológica."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def apply_dilation(image, kernel_size=(5,5), iterations=1):
    """Aplica dilatação morfológica."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

# Outras técnicas como segmentação por cor, textura, subtração de fundo seriam mais complexas
# e podem ser adicionadas conforme a necessidade específica da aplicação.



def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) na imagem."""
    # CLAHE funciona melhor em escala de cinza
    gray_image = convert_to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


