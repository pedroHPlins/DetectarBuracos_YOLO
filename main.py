import cv2
import os
import input_handler
import yolo_processor
import output_handler
import classic_processing

# --- Configurações ---
# Usar uma das imagens de buraco baixadas
IMAGE_PATH = "./imagens/buracos_2.jpg" # Imagem com buraco (índice 3 da busca)
OUTPUT_DIR = "./output_pothole"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots") # Diretório específico para plots
CONFIDENCE_THRESHOLD = 0.40 # Ajustar conforme necessário para o modelo de buraco

# Criar diretórios de saída se não existirem
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# --- Fluxo Principal ---
def main():
    print(f"Iniciando processamento para detecção de BURACOS em: {IMAGE_PATH}")

    # 1. Carregar Imagem
    original_image = input_handler.load_image(IMAGE_PATH)
    if original_image is None:
        return
    print("Imagem original carregada com sucesso.")
    # Salvar imagem e plot
    output_handler.save_image(original_image, os.path.join(OUTPUT_DIR, "00_pothole_original.jpg"))
    output_handler.save_image_plot(original_image, "Figura 1: Imagem Original (Buraco)", os.path.join(PLOT_DIR, "00_pothole_original_plot.png"))

    # --- Processamento Clássico (Opcional - pode ser útil para realçar buracos) ---
    print("\nAplicando técnicas clássicas de PDI (Opcional)...")
    # Técnica 1: Escala de Cinza
    gray_image = classic_processing.convert_to_grayscale(original_image)
    output_handler.save_image(gray_image, os.path.join(OUTPUT_DIR, "01_pothole_classic_grayscale.jpg"))
    output_handler.save_image_plot(gray_image, "Figura 2: Imagem em Escala de Cinza", os.path.join(PLOT_DIR, "01_pothole_classic_grayscale_plot.png"))

    # Técnica 2: Filtro Gaussiano
    blurred_image = classic_processing.apply_gaussian_blur(original_image, kernel_size=(5, 5)) # Kernel menor talvez?
    output_handler.save_image(blurred_image, os.path.join(OUTPUT_DIR, "02_pothole_classic_gaussian_blur.jpg"))
    output_handler.save_image_plot(blurred_image, "Figura 3: Imagem com Filtro Gaussiano (5x5)", os.path.join(PLOT_DIR, "02_pothole_classic_gaussian_blur_plot.png"))

    # Técnica 3: Detecção de Bordas Canny
    edges_canny = classic_processing.detect_edges_canny(original_image, low_threshold=50, high_threshold=150)
    output_handler.save_image(edges_canny, os.path.join(OUTPUT_DIR, "03_pothole_classic_canny_edges.jpg"))
    output_handler.save_image_plot(edges_canny, "Figura 4: Detecção de Bordas (Canny)", os.path.join(PLOT_DIR, "03_pothole_classic_canny_edges_plot.png"))
    print("Técnicas clássicas aplicadas e imagens/plots intermediários salvos.")

    # --- Aplicação YOLOv8 (Modelo Específico para Buracos) --- 
    print("\nAplicando YOLOv8 (modelo de buracos) na imagem ORIGINAL...")
    try:
        # Usar a imagem original para detecção
        yolo_results, annotated_image = yolo_processor.detect_objects_yolo(original_image, confidence_threshold=CONFIDENCE_THRESHOLD)
        print("Detecção YOLOv8 (modelo de buracos) concluída.")
        
        # Salvar imagem e plot anotados
        output_handler.save_image(annotated_image, os.path.join(OUTPUT_DIR, "04_yolo_pothole_annotated.jpg"))
        output_handler.save_image_plot(annotated_image, "Figura 5: Detecção de Buracos (YOLOv8 Específico)", os.path.join(PLOT_DIR, "04_yolo_pothole_annotated_plot.png"))
        
        # Salvar dados CSV
        detections_data = yolo_processor.extract_detection_data(yolo_results)
        if detections_data:
            output_handler.save_detections_to_csv(detections_data, os.path.join(OUTPUT_DIR, "yolo_pothole_detections.csv"))
            print(f"Dados de detecção ({len(detections_data)} buracos?) salvos.")
        else:
            print("Nenhum buraco detectado pelo modelo YOLO específico.")
            
    except Exception as e:
        print(f"Erro durante o processamento YOLOv8 para buracos: {e}")

    print(f"\nProcessamento concluído. Resultados salvos em: {OUTPUT_DIR} e plots em: {PLOT_DIR}")

if __name__ == "__main__":
    main()

