# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
import os

# Caminho para os pesos YOLOv5 baixados localmente
LOCAL_WEIGHTS_PATH = "./pothole_yolov5s.pt"
# Repositório oficial para carregar a estrutura YOLOv5
YOLOV5_REPO_OR_DIR = 'ultralytics/yolov5'

# Variável global para o modelo
model = None

# Tentar carregar o modelo YOLOv5 local via PyTorch Hub
try:
    print(f"Tentando carregar o modelo YOLOv5 local via PyTorch Hub: {LOCAL_WEIGHTS_PATH}")
    # force_reload=True pode ser necessário se houver cache problemático
    # verbose=False para reduzir o output do torch.hub
    model = torch.hub.load(YOLOV5_REPO_OR_DIR, 'custom', path=LOCAL_WEIGHTS_PATH, force_reload=False, verbose=False)
    # Definir explicitamente o nome da classe, pois pode não vir com o modelo customizado
    model.names = {0: 'pothole'} # Baseado na estrutura comum de datasets de buracos
    print("Modelo YOLOv5 local carregado com sucesso via PyTorch Hub.")
    print(f"Nomes das classes definidos: {model.names}")

except Exception as e:
    print(f"Erro ao carregar o modelo local {LOCAL_WEIGHTS_PATH} via PyTorch Hub: {e}")
    print("Não foi possível carregar o modelo de buracos. Verifique o arquivo de pesos e a conexão com a internet (para PyTorch Hub).")
    model = None # Define como None para indicar falha

def detect_objects_yolo(image, confidence_threshold=0.25):
    """Detecta objetos (buracos) em uma imagem usando o modelo YOLOv5 carregado via PyTorch Hub.

    Args:
        image: A imagem de entrada (formato NumPy BGR).
        confidence_threshold: Limiar de confiança mínimo para considerar uma detecção.

    Returns:
        results_df: DataFrame do Pandas com as detecções (formato YOLOv5), ou None se falhar.
        annotated_image: Imagem com as detecções desenhadas.
    """
    if model is None:
        print("Modelo YOLOv5 não está carregado. Abortando detecção.")
        # Retorna None e a imagem original sem anotações
        return None, image.copy()

    print(f"Executando detecção com modelo YOLOv5 local e limiar de confiança: {confidence_threshold}")
    model.conf = confidence_threshold # Define o limiar de confiança no modelo

    # A inferência retorna um objeto Results
    results = model(image) # Passa a imagem BGR (YOLOv5 lida com a conversão)

    # O método render() desenha as caixas na imagem. Retorna uma lista de imagens anotadas.
    annotated_images = results.render()
    annotated_image = annotated_images[0] if annotated_images else image.copy()

    # results.pandas().xyxy[0] retorna um DataFrame com as detecções para a imagem 0
    results_df = results.pandas().xyxy[0] if results and hasattr(results, 'pandas') else None

    if results_df is not None and not results_df.empty:
         print(f"Detecções encontradas: {len(results_df)}")
    else:
         print("Nenhuma detecção encontrada com o limiar especificado.")
         results_df = None # Garante que None seja retornado se o DataFrame estiver vazio ou não existir

    return results_df, annotated_image # Retorna o DataFrame e a imagem anotada

def extract_detection_data(yolo_results_df):
    """Extrai dados relevantes das detecções do DataFrame YOLOv5.

    Args:
        yolo_results_df: DataFrame do Pandas retornado pela inferência YOLOv5 (results.pandas().xyxy[0]).

    Returns:
        detections: Uma lista de dicionários, cada um representando um objeto detectado.
                    Ex: [{"class_id": 0, "class_name": "pothole", "confidence": 0.85, "box": [x1, y1, x2, y2]}]
    """
    detections = []
    if yolo_results_df is None or yolo_results_df.empty:
        print("DataFrame de resultados YOLOv5 inválido ou vazio para extrair dados.")
        return detections

    # Garante que model.names existe
    class_names = getattr(model, 'names', {0: 'pothole'}) # Usa fallback se names não existir

    # Itera sobre as linhas do DataFrame
    for index, row in yolo_results_df.iterrows():
        try:
            class_id = int(row['class'])
            # Usa o nome da classe definido no modelo carregado
            class_name = class_names.get(class_id, f"class_{class_id}")
            confidence = float(row['confidence'])
            # Coordenadas já estão no formato [xmin, ymin, xmax, ymax]
            box = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": box
            })
        except Exception as e:
            print(f"Erro ao processar linha de detecção {index}: {e}")
            continue # Pula para a próxima detecção em caso de erro

    print(f"Dados extraídos para {len(detections)} detecções.")
    return detections

