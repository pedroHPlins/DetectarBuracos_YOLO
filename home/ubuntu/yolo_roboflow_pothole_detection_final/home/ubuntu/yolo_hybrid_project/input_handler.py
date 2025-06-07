import cv2
import os

def load_image(image_path):
    """Carrega uma imagem de um arquivo."""
    if not os.path.exists(image_path):
        print(f"Erro: Arquivo de imagem não encontrado em {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem de {image_path}")
    return image

# Funções para carregar vídeo podem ser adicionadas aqui
# def load_video(video_path):
#     ...

