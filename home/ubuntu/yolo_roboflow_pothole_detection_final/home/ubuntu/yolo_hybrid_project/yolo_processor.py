# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pandas as pd
# Importar a classe correta conforme a documentação
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# --- Configuração Roboflow ---
ROBOFLOW_API_KEY = "d3nDPMwWE1xqV74ma2MN"
ROBOFLOW_MODEL_ID = "pothole-detection-yolov8/1"
ROBOFLOW_API_URL = "https://detect.roboflow.com"

# Variável global para o cliente de inferência
client = None

# Tentar inicializar o cliente Roboflow
try:
    print(f"Inicializando cliente HTTP de inferência Roboflow para o modelo: {ROBOFLOW_MODEL_ID}")
    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY
    )
    client.select_api_v0()
    print("Cliente Roboflow inicializado com sucesso.")
except Exception as e:
    print(f"Erro ao inicializar o cliente Roboflow: {e}")
    client = None

def draw_predictions(image, predictions):
    """Desenha as caixas delimitadoras e labels na imagem usando OpenCV.
       MODIFICADO: Espera predições com a chave 'box' contendo [x1, y1, x2, y2].
    """
    annotated_image = image.copy()
    height, width, _ = annotated_image.shape
    
    for p in predictions:
        try:
            # Extrair dados do dicionário de predição
            box = p.get("box") # Espera [x1, y1, x2, y2]
            if box is None or len(box) != 4:
                print(f"Skipping prediction due to missing or invalid 'box': {p}")
                continue
                
            confidence = p.get("confidence", 0.0)
            class_name = p.get("class", "N/A") # Nome da classe

            # Coordenadas do canto superior esquerdo (x1, y1) e inferior direito (x2, y2)
            x1, y1, x2, y2 = map(int, box)

            # Garantir que as coordenadas estejam dentro dos limites da imagem
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            # Desenhar retângulo (cor BGR: Verde)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Preparar texto do label
            label = f"{class_name}: {confidence:.2f}"

            # Obter tamanho do texto para o fundo
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Desenhar fundo para o label (acima da caixa)
            label_y1 = max(y1, text_height + 10) # Garante que o fundo não saia do topo
            cv2.rectangle(annotated_image, (x1, label_y1 - text_height - baseline), (x1 + text_width, label_y1), (0, 255, 0), cv2.FILLED)

            # Desenhar texto do label (cor BGR: Preto)
            cv2.putText(annotated_image, label, (x1, label_y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        except Exception as e:
            print(f"Erro ao desenhar predição {p}: {e}")
            continue
            
    return annotated_image

def detect_objects_yolo(image, confidence_threshold=0.25):
    """Detecta objetos (buracos) em uma imagem usando a API Roboflow.
       MODIFICADO: Assume que a API retorna predições com 'box' [x1, y1, x2, y2].
                   Remove a chamada para extract_detection_data.
    Args:
        image: A imagem de entrada (formato NumPy BGR).
        confidence_threshold: Limiar de confiança mínimo para considerar uma detecção.

    Returns:
        results_list: Lista de dicionários com as detecções filtradas, ou None se falhar.
        annotated_image: Imagem com as detecções desenhadas manualmente.
    """
    if client is None:
        print("Cliente Roboflow não está inicializado. Abortando detecção.")
        return None, image.copy()

    print(f"Executando inferência via API Roboflow com limiar de confiança: {confidence_threshold}")

    try:
        result_dict = client.infer(image, model_id=ROBOFLOW_MODEL_ID)
        predictions_raw = result_dict.get("predictions", [])
        
        # Adiciona a chave 'box' [x1, y1, x2, y2] a cada predição raw
        # Isso assume que a API retorna 'x', 'y', 'width', 'height'
        # Se a API já retornar 'box', esta etapa pode ser simplificada/removida.
        # Vamos assumir por agora que precisamos calcular 'box'.
        predictions_processed = []
        for p in predictions_raw:
            try:
                x_center = p["x"]
                y_center = p["y"]
                w = p["width"]
                h = p["height"]
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                p["box"] = [float(x1), float(y1), float(x2), float(y2)]
                predictions_processed.append(p)
            except KeyError as ke:
                print(f"Predição raw não contém chaves esperadas ('x', 'y', 'width', 'height'): {p}. Erro: {ke}. Pulando predição.")
                continue
            except Exception as e:
                 print(f"Erro ao processar predição raw {p}: {e}. Pulando predição.")
                 continue
        
        # Filtrar predições com base no limiar de confiança
        predictions_filtered = [p for p in predictions_processed if p["confidence"] >= confidence_threshold]
        
        # A lista de resultados é a própria lista filtrada
        results_list = predictions_filtered

        # Chamar a função manual para desenhar as caixas (agora espera 'box')
        annotated_image = draw_predictions(image, predictions_filtered)

        if predictions_filtered:
             print(f"Detecções encontradas e filtradas via Roboflow API: {len(predictions_filtered)}")
        else:
             print("Nenhuma detecção encontrada com o limiar especificado via Roboflow API.")

        # Retorna a lista de dicionários filtrados (que agora inclui 'box')
        return results_list, annotated_image

    except Exception as e:
        print(f"Erro durante a inferência com a API Roboflow: {e}")
        return None, image.copy()

# REMOVIDO: extract_detection_data não é mais necessária se a formatação ocorrer em detect_objects_yolo
# def extract_detection_data(predictions):
#    ...

