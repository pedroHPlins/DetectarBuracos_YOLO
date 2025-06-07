# Avaliação Prática - AV3

## Descrição do projeto

Este projeto aplica técnicas de Processamento Digital de Imagens (PDI) no desenvolvimento de um sistema híbrido para análise de cenas urbanas. O contexto é acadêmico, visando explorar a combinação de abordagens clássicas de PDI com redes neurais convolucionais modernas, especificamente o modelo YOLOv8. A relevância reside na crescente demanda por sistemas inteligentes de monitoramento urbano, capazes de extrair informações úteis de imagens e vídeos para aplicações como gerenciamento de tráfego, segurança pública e planejamento urbano. O objetivo principal do projeto é desenvolver e validar um protótipo em Python que integre pré-processamento clássico e detecção de objetos via YOLOv8 para identificar e classificar elementos como veículos, pedestres e outros objetos relevantes em imagens de ambientes urbanos. A meta é criar uma base funcional que demonstre a viabilidade da abordagem híbrida, capitalizando a robustez do YOLOv8 na detecção e a possibilidade de refinamento e análise complementar através das técnicas clássicas.

## Metodologia e Implementação

A metodologia adotada foi a criação de um sistema modular em Python, combinando funções de bibliotecas estabelecidas para PDI e aprendizado profundo. A arquitetura da solução foi projetada para ser extensível, compreendendo os seguintes módulos principais:

*   **Manipulação de Entrada/Saída (`input_handler.py`, `output_handler.py`)**: Funções para carregar imagens e salvar os resultados processados (imagens anotadas e dados tabulares).
*   **Processamento Clássico (`classic_processing.py`)**: Implementa algoritmos tradicionais de PDI, como conversão para escala de cinza, filtragem Gaussiana, detecção de bordas (Canny) e operações morfológicas. Embora a validação inicial tenha focado no YOLOv8, este módulo permite a futura integração dessas técnicas.
*   **Processador YOLO (`yolo_processor.py`)**: Encapsula a interação com o modelo YOLOv8 (`yolov8n.pt` utilizado nos testes), realizando a inferência para detecção de objetos e extraindo informações relevantes como classes, confianças e caixas delimitadoras (bounding boxes).
*   **Orquestração (`main.py`)**: Script principal que coordena o fluxo de processamento, desde o carregamento da imagem até a geração dos resultados.

A implementação utilizou Python 3.11 com as seguintes bibliotecas chave:
*   **OpenCV (`opencv-python`)**: Para leitura/escrita de imagens e implementação das funções de PDI clássico.
*   **Ultralytics (`ultralytics`)**: Para carregar o modelo YOLOv8 pré-treinado e realizar a detecção de objetos.
*   **NumPy**: Para manipulação eficiente de arrays de imagem.

O código-fonte completo, incluindo os módulos mencionados e o arquivo de dependências (`requirements.txt`), encontra-se na pasta do projeto.

## Execução e Resultados

Para executar o projeto, é necessário um ambiente Python 3.11 (ou compatível) com as dependências instaladas via `pip install -r requirements.txt`. O script principal (`main.py`) pode ser executado diretamente (`python main.py`). Por padrão, ele processa a imagem de teste `/home/ubuntu/upload/search_images/8ZDEbTivVnaz.jpg` e salva os resultados no diretório `output/`.

A validação foi realizada com a imagem mencionada, uma cena urbana complexa. O sistema executou o fluxo esperado:
1.  Carregamento da imagem.
2.  Aplicação do YOLOv8 (modelo `yolov8n.pt`, limiar de confiança 0.3).
3.  Geração de uma imagem anotada (`output/yolo_annotated_output.jpg`), mostrando as caixas delimitadoras e rótulos dos objetos detectados (carros, pessoas, ônibus, etc.).
4.  Criação de um arquivo CSV (`output/yolo_detections.csv`) com detalhes de cada detecção (classe, confiança, coordenadas).

Os resultados demonstram a capacidade do sistema em detectar múltiplos objetos corretamente na imagem de teste, validando a funcionalidade básica da integração com o YOLOv8. Os arquivos de saída estão disponíveis na pasta `output/` do projeto.

## Discussão e Conclusão

O projeto cumpriu o objetivo de desenvolver um framework inicial para um sistema híbrido de processamento de imagem urbana. A arquitetura modular implementada em Python, utilizando OpenCV e YOLOv8, provou ser funcional para a detecção de objetos em imagens estáticas. A validação confirmou que o sistema consegue carregar imagens, aplicar um modelo YOLO pré-treinado e gerar resultados visuais e tabulares das detecções.

A principal limitação da implementação atual é a ausência dos módulos de fusão e análise, que seriam responsáveis por combinar os resultados do YOLOv8 com técnicas clássicas e extrair informações de nível superior. A abordagem híbrida, portanto, foi apenas parcialmente explorada.

Como trabalhos futuros e possíveis extensões, sugere-se:
*   Implementar a lógica de fusão para refinar as detecções do YOLO ou usar PDI clássico em regiões de interesse.
*   Desenvolver o módulo de análise para tarefas como contagem, rastreamento ou detecção de eventos específicos.
*   Experimentar com treinamento customizado do YOLOv8 para aplicações específicas.
*   Adaptar o sistema para processamento de vídeo e, eventualmente, tempo real.

Em conclusão, o projeto estabelece uma base sólida e funcional para futuras explorações em sistemas híbridos de PDI para análise urbana, demonstrando a integração bem-sucedida do YOLOv8 em um fluxo de processamento em Python.

