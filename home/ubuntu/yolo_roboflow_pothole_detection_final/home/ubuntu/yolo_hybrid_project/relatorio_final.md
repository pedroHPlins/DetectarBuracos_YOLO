# Relatório Final: Sistema Híbrido de Processamento de Imagem com YOLOv8

## Introdução e Objetivo

Este relatório documenta o desenvolvimento de um sistema híbrido de processamento de imagem, concebido como parte de um projeto universitário. O objetivo central foi criar uma solução em Python que integrasse técnicas clássicas de processamento de imagem com a inteligência artificial do modelo YOLOv8. A finalidade do sistema é a detecção, análise e classificação de objetos urbanos presentes em imagens ou vídeos, com aplicações práticas que incluem monitoramento de tráfego, identificação de veículos estacionados irregularmente, contagem de pedestres e ciclistas, e mapeamento de anomalias na infraestrutura urbana, como falhas na pavimentação. A abordagem híbrida visa capitalizar as fortalezas de ambos os paradigmas: a interpretabilidade e o controle fino das técnicas clássicas, e a robustez na detecção de objetos complexos oferecida pelo YOLOv8.

## Arquitetura da Solução

Para garantir modularidade, clareza e facilidade de manutenção, o sistema foi estruturado em diversos módulos Python, cada um com responsabilidades bem definidas. A arquitetura detalhada foi documentada separadamente no arquivo `architecture.md`, mas seus componentes principais são:

*   **`main.py`**: Orquestrador central, gerenciando o fluxo de execução desde a entrada de dados até a geração de resultados.
*   **`input_handler.py`**: Responsável pelo carregamento e validação das imagens de entrada.
*   **`classic_processing.py`**: Implementa funções para técnicas clássicas como conversão para escala de cinza, aplicação de filtros (Gaussiano), detecção de bordas (Canny) e operações morfológicas. Embora não totalmente explorado na validação inicial focada no YOLOv8, este módulo estabelece a base para a integração futura dessas técnicas.
*   **`yolo_processor.py`**: Encapsula a interação com o modelo YOLOv8 (especificamente, `yolov8n.pt` foi utilizado nos testes), realizando a detecção de objetos e o pós-processamento inicial dos resultados, como a filtragem por limiar de confiança.
*   **`output_handler.py`**: Gerencia o salvamento dos resultados, incluindo imagens anotadas com as detecções e arquivos CSV contendo os dados detalhados de cada objeto detectado (classe, confiança, coordenadas da caixa delimitadora).
*   **`fusion.py`** e **`analysis.py`**: Módulos planejados, mas não implementados nesta fase inicial, destinados a combinar os resultados das abordagens clássica e YOLOv8 e a realizar análises mais aprofundadas sobre os dados extraídos, respectivamente.
*   **`config.py`** e **`utils.py`**: Previstos para configurações e funções utilitárias, respectivamente, embora não extensivamente utilizados nesta versão.
*   **`requirements.txt`**: Lista as dependências Python essenciais, como `opencv-python`, `numpy` e `ultralytics`.

O fluxo de processamento básico implementado segue a sequência: Carregamento da Imagem (`input_handler`) -> Detecção com YOLOv8 (`yolo_processor`) -> Salvamento dos Resultados (`output_handler`).

## Implementação e Tecnologias

O desenvolvimento foi realizado inteiramente em Python 3.11. As bibliotecas cruciais empregadas foram:

*   **OpenCV (`opencv-python`)**: Utilizada para operações fundamentais de processamento de imagem, como leitura, escrita, conversão de cores e implementação das técnicas clássicas.
*   **Ultralytics (`ultralytics`)**: Fornece a interface para o modelo YOLOv8, simplificando o carregamento do modelo pré-treinado, a execução da inferência e o acesso aos resultados da detecção.
*   **NumPy**: Essencial para manipulação eficiente de arrays multidimensionais, que são a representação padrão das imagens no OpenCV.

O código foi organizado conforme a arquitetura descrita, com cada funcionalidade encapsulada em funções dentro dos módulos correspondentes.

## Instruções de Uso e Execução

Para executar o projeto, siga os passos abaixo:

1.  **Ambiente**: Certifique-se de ter Python 3.11 ou compatível instalado.
2.  **Dependências**: Navegue até o diretório raiz do projeto (`yolo_hybrid_project`) via terminal e instale as dependências necessárias executando o comando: `pip install -r requirements.txt`. Isso instalará OpenCV, Ultralytics, NumPy e outras bibliotecas requeridas.
3.  **Configuração**: O script principal (`main.py`) está configurado para processar uma imagem de exemplo (`/home/ubuntu/upload/search_images/8ZDEbTivVnaz.jpg`) e salvar os resultados no diretório `output/` dentro da pasta do projeto. Para processar outra imagem, modifique a variável `IMAGE_PATH` no início do `main.py`. O limiar de confiança para as detecções do YOLOv8 pode ser ajustado na variável `CONFIDENCE_THRESHOLD`.
4.  **Execução**: Execute o script principal a partir do diretório raiz do projeto com o comando: `python main.py` (ou `python3.11 main.py` se houver múltiplas versões do Python).
5.  **Resultados**: Após a execução, os resultados serão salvos no diretório `output/`. Você encontrará:
    *   `yolo_annotated_output.jpg`: A imagem original com as caixas delimitadoras e rótulos dos objetos detectados pelo YOLOv8.
    *   `yolo_detections.csv`: Um arquivo CSV contendo informações detalhadas sobre cada detecção (ID da classe, nome da classe, confiança, coordenadas da caixa delimitadora).

## Resultados da Validação

O sistema foi validado utilizando uma imagem real de uma cena urbana contendo múltiplos veículos e pedestres (`/home/ubuntu/upload/search_images/8ZDEbTivVnaz.jpg`). A execução do `main.py` demonstrou o funcionamento correto do fluxo:

*   A imagem foi carregada com sucesso.
*   O modelo YOLOv8 (`yolov8n.pt`) foi carregado (baixado automaticamente na primeira execução) e aplicado à imagem.
*   Diversos objetos (carros, pessoas, ônibus, etc.) foram detectados com sucesso, conforme o limiar de confiança estabelecido (0.3).
*   A imagem anotada (`yolo_annotated_output.jpg`) foi gerada, visualizando claramente as detecções.
*   Os dados detalhados das detecções foram extraídos e salvos corretamente no arquivo `yolo_detections.csv`.

Os arquivos de saída gerados durante esta validação estão incluídos no diretório `output/` dentro do pacote do projeto.

## Possíveis Extensões

Conforme delineado no documento original do projeto, existem várias direções para expansão futura:

*   **Implementação da Fusão**: Desenvolver a lógica no módulo `fusion.py` para combinar efetivamente os resultados das técnicas clássicas (ex: segmentação, detecção de bordas) com as detecções do YOLOv8, visando melhorar a precisão e reduzir falsos positivos/negativos.
*   **Módulo de Análise**: Implementar funcionalidades no `analysis.py` para extrair informações de nível superior a partir das detecções, como contagem de objetos específicos, análise de fluxo ou detecção de eventos.
*   **Treinamento Customizado**: Treinar ou ajustar finamente um modelo YOLOv8 em um dataset específico para melhorar o desempenho em domínios particulares (ex: detecção de tipos específicos de falhas na pavimentação).
*   **Processamento de Vídeo e Tempo Real**: Adaptar os módulos `input_handler` e o fluxo principal para processar streams de vídeo, potencialmente em tempo real, incluindo técnicas como subtração de fundo e rastreamento de objetos.
*   **Integração com Dados Geográficos**: Combinar as detecções com informações de geolocalização para aplicações de mapeamento.

## Conclusão

O projeto atingiu seu objetivo inicial de desenvolver um framework básico para um sistema híbrido de processamento de imagem utilizando Python, OpenCV e YOLOv8. A arquitetura modular estabelecida fornece uma base sólida para futuras implementações das funcionalidades de fusão, análise e outras extensões. A validação demonstrou a capacidade do sistema de carregar imagens, aplicar o YOLOv8 para detecção de objetos e salvar os resultados de forma organizada.

