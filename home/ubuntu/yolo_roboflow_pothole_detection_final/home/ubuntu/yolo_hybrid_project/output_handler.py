import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def save_image(image, output_path):
    """Salva uma imagem em um arquivo."""
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, image)
        print(f"Imagem salva com sucesso em: {output_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar a imagem em {output_path}: {e}")
        return False

def save_image_plot(image, title, output_path):
    """Salva a visualização de uma imagem como um plot Matplotlib."""
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(10, 8)) # Ajuste o tamanho conforme necessário
        
        # Matplotlib espera RGB, OpenCV usa BGR (exceto para escala de cinza)
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
        else:
            plt.imshow(image, cmap='gray') # Imagem em escala de cinza
            
        plt.title(title, fontsize=14)
        plt.axis('off') # Remove os eixos
        plt.savefig(output_path, bbox_inches='tight', dpi=150) # Salva com boa resolução
        plt.close() # Fecha a figura para liberar memória
        print(f"Plot da imagem salvo com sucesso em: {output_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar o plot da imagem em {output_path}: {e}")
        return False

def save_detections_to_csv(detections, output_path):
    """Salva os dados de detecção em um arquivo CSV."""
    if not detections:
        print("Nenhuma detecção para salvar.")
        return False
    
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        header = detections[0].keys()
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(detections)
        print(f"Dados de detecção salvos com sucesso em: {output_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar os dados de detecção em {output_path}: {e}")
        return False

