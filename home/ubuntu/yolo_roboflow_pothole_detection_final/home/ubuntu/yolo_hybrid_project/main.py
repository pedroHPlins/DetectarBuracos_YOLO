import cv2
import os
import input_handler
import classic_processing
import yolo_processor
import output_handler
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuração ---
# Lista de caminhos para as novas imagens fornecidas pelo usuário
image_paths = [
    "/home/ubuntu/yolo_hybrid_project/home/ubuntu/yolo_roboflow_pothole_detection_final/home/ubuntu/yolo_hybrid_project/input_images/rua_asfaltada2.jpg",
    "/home/ubuntu/yolo_hybrid_project/home/ubuntu/yolo_roboflow_pothole_detection_final/home/ubuntu/yolo_hybrid_project/input_images/rua_asfaltada1.png",
    "/home/ubuntu/yolo_hybrid_project/home/ubuntu/yolo_roboflow_pothole_detection_final/home/ubuntu/yolo_hybrid_project/input_images/buracos_1.png",
    "/home/ubuntu/yolo_hybrid_project/home/ubuntu/yolo_roboflow_pothole_detection_final/home/ubuntu/yolo_hybrid_project/input_images/buracos_2.jpg"
]

output_base_dir = "/home/ubuntu/yolo_hybrid_project/output_processed"
confidence_threshold_yolo = 0.25 # Limiar de confiança para detecção YOLO

# --- Execução ---
def process_single_image(image_path, output_base_dir, confidence_threshold_yolo):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nIniciando processamento para a imagem: {image_name}")

    # Cria diretórios de saída se não existirem
    output_dir_original = os.path.join(output_base_dir, image_name, "original")
    output_dir_clahe = os.path.join(output_base_dir, image_name, "clahe")
    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_clahe, exist_ok=True)

    # 1. Carregar Imagem Original
    original_image = input_handler.load_image(image_path)
    if original_image is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None, None, None

    # Salva a imagem original
    output_handler.save_image(original_image, os.path.join(output_dir_original, f"{image_name}_original.jpg"))

    # 2. Aplicar CLAHE na imagem
    print("Aplicando CLAHE na imagem...")
    # CLAHE funciona melhor em escala de cinza, então convertemos antes de aplicar
    clahe_processed_image = classic_processing.convert_to_grayscale(original_image)
    clahe_processed_image = classic_processing.apply_clahe(clahe_processed_image)
    # Para visualização e detecção YOLO, convertemos de volta para 3 canais (BGR)
    clahe_processed_image_bgr = cv2.cvtColor(clahe_processed_image, cv2.COLOR_GRAY2BGR)
    output_handler.save_image(clahe_processed_image_bgr, os.path.join(output_dir_clahe, f"{image_name}_clahe_processed.jpg"))
    print(f"Imagem processada com CLAHE salva em: {os.path.join(output_dir_clahe, f'{image_name}_clahe_processed.jpg')}")

    # 3. Aplicar Detecção YOLO na Imagem Original
    print("Aplicando detecção YOLO na imagem original...")
    yolo_results_original, yolo_annotated_original_image = yolo_processor.detect_objects_yolo(original_image, confidence_threshold_yolo)
    output_handler.save_image(yolo_annotated_original_image, os.path.join(output_dir_original, f"{image_name}_yolo_annotated_original.jpg"))
    print("Detecção YOLO na imagem original concluída.")

    # 4. Aplicar Detecção YOLO na Imagem Processada com CLAHE
    print("Aplicando detecção YOLO na imagem processada com CLAHE...")
    yolo_results_clahe, yolo_annotated_clahe_image = yolo_processor.detect_objects_yolo(clahe_processed_image_bgr, confidence_threshold_yolo)
    output_handler.save_image(yolo_annotated_clahe_image, os.path.join(output_dir_clahe, f"{image_name}_yolo_annotated_clahe.jpg"))
    print("Detecção YOLO na imagem CLAHE concluída.")

    # 5. Salvar Dados da Detecção em CSV
    df_original = pd.DataFrame(yolo_results_original) if yolo_results_original else pd.DataFrame()
    df_clahe = pd.DataFrame(yolo_results_clahe) if yolo_results_clahe else pd.DataFrame()

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

    # Comparação de confiabilidade (exemplo simples)
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

        # Comparação mais detalhada pode ser feita aqui, como contagem de detecções, etc.

    print("Processamento de todas as imagens concluído.")
    plt.close("all") # Fecha todas as figuras Matplotlib abertas

if __name__ == "__main__":
    main()


