import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Importar funções do yolo_processor.py
from yolo_processor import detect_objects_yolo, extract_detection_data

# --- Funções de input_handler.py ---
def load_image(image_path):
    """Carrega uma imagem de um arquivo."""
    if not os.path.exists(image_path):
        print(f"Erro: Arquivo de imagem não encontrado em {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem de {image_path}")
    return image

# --- Funções de classic_processing.py ---
def convert_to_grayscale(image):
    """Converte uma imagem para escala de cinza."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Aplica um filtro Gaussiano na imagem."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) na imagem."""
    gray_image = convert_to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)

# --- Funções de output_handler.py ---
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

        plt.figure(figsize=(10, 8))
        
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
        else:
            plt.imshow(image, cmap="gray")
            
        plt.title(title, fontsize=14)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
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
        
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(detections)
        print(f"Dados de detecção salvos com sucesso em: {output_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar os dados de detecção em {output_path}: {e}")
        return False

# --- Configuração e Execução Principal ---
image_paths = [
    "./imagens/rua_asfaltada2.jpg",
    "./imagens/rua_asfaltada1.png",
    "./imagens/buracos_1.png",
    "./imagens/buracos_2.jpg"
]

output_base_dir = "./output_processed"
confidence_threshold_yolo = 0.25

def process_single_image(image_path, output_base_dir, confidence_threshold_yolo):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nIniciando processamento para a imagem: {image_name}")

    output_dir_original = os.path.join(output_base_dir, image_name, "original")
    output_dir_clahe = os.path.join(output_base_dir, image_name, "clahe")
    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_clahe, exist_ok=True)

    original_image = load_image(image_path)
    if original_image is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None, None, None

    save_image(original_image, os.path.join(output_dir_original, f"{image_name}_original.jpg"))

    print("Aplicando CLAHE na imagem...")
    clahe_processed_image = convert_to_grayscale(original_image)
    clahe_processed_image = apply_clahe(clahe_processed_image)
    clahe_processed_image_bgr = cv2.cvtColor(clahe_processed_image, cv2.COLOR_GRAY2BGR)
    save_image(clahe_processed_image_bgr, os.path.join(output_dir_clahe, f"{image_name}_clahe_processed.jpg"))
    print(f"Imagem processada com CLAHE salva em: {os.path.join(output_dir_clahe, f'{image_name}_clahe_processed.jpg')}")

    print("Aplicando detecção YOLO na imagem original...")
    yolo_results_original, yolo_annotated_original_image = detect_objects_yolo(original_image, confidence_threshold_yolo)
    save_image(yolo_annotated_original_image, os.path.join(output_dir_original, f"{image_name}_yolo_annotated_original.jpg"))
    print("Detecção YOLO na imagem original concluída.")

    print("Aplicando detecção YOLO na imagem processada com CLAHE...")
    yolo_results_clahe, yolo_annotated_clahe_image = detect_objects_yolo(clahe_processed_image_bgr, confidence_threshold_yolo)
    save_image(yolo_annotated_clahe_image, os.path.join(output_dir_clahe, f"{image_name}_yolo_annotated_clahe.jpg"))
    print("Detecção YOLO na imagem CLAHE concluída.")

    df_original = pd.DataFrame(extract_detection_data(yolo_results_original)) if yolo_results_original else pd.DataFrame()
    df_clahe = pd.DataFrame(extract_detection_data(yolo_results_clahe)) if yolo_results_clahe else pd.DataFrame()

    csv_path_original = os.path.join(output_dir_original, f"{image_name}_detections_original.csv")
    csv_path_clahe = os.path.join(output_dir_clahe, f"{image_name}_detections_clahe.csv")

    if not df_original.empty:
        df_original.to_csv(csv_path_original, index=False)
        print(f"{len(df_original)} detecções YOLO (original) salvas em CSV.")
    else:
        print("Nenhuma detecção YOLO encontrada para a imagem original.")

    if not df_clahe.empty:
        df_clahe.to_csv(csv_path_clahe, index=False)
        print(f"{len(df_clahe)} detecções YOLO (CLAHE) salvas em CSV.")
    else:
        print("Nenhuma detecção YOLO encontrada para a imagem CLAHE.")

    return df_original, df_clahe, image_name

def main():
    all_results = []
    for img_path in image_paths:
        df_orig, df_clahe, img_name = process_single_image(img_path, output_base_dir, confidence_threshold_yolo)
        if df_orig is not None and df_clahe is not None:
            all_results.append({
                'image_name': img_name,
                'original_detections': df_orig,
                'clahe_detections': df_clahe
            })

    print("\n--- Comparação de Confiabilidade ---")
    for result in all_results:
        img_name = result['image_name']
        df_orig = result['original_detections']
        df_clahe = result['clahe_detections']

        print(f"\nImagem: {img_name}")
        if not df_orig.empty:
            avg_conf_orig = df_orig['confidence'].mean()
            print(f"  Confiança média (Original): {avg_conf_orig:.4f}")
        else:
            print("  Nenhuma detecção na imagem original.")

        if not df_clahe.empty:
            avg_conf_clahe = df_clahe['confidence'].mean()
            print(f"  Confiança média (CLAHE): {avg_conf_clahe:.4f}")
        else:
            print("  Nenhuma detecção na imagem CLAHE.")

    print("Processamento de todas as imagens concluído.")
    plt.close("all")

if __name__ == "__main__":
    main()


