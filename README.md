# Person Detected — Edge AI Risk Zone Monitor

Projeto técnico para monitoramento de comportamento de risco em zonas de máquinas pesadas usando visão computacional embarcada (Edge AI).

## Objetivo

Detectar pessoas em imagens ou frames de vídeo e verificar se houve entrada em zonas virtuais de risco definidas por polígonos.

Fluxo principal:

`imagem/frame -> YOLO -> pessoa -> bounding box -> ponto dos pés -> zona -> risco`

## Escopo inicial

Nesta primeira etapa, o projeto contém a fundação e um pipeline mínimo para detectar a classe `person` com um modelo YOLO pré-treinado.

## Stack inicial

- Python 3.11+
- Ultralytics YOLO
- OpenCV
- FastAPI (próxima etapa)
- Docker / Docker Buildx (próxima etapa)

## Roadmap

1. Detectar `person` em uma imagem.
2. Desenhar bounding boxes.
3. Extrair o ponto inferior central da pessoa.
4. Classificar a pessoa em zona segura, amarela ou vermelha.
5. Expor endpoints JSON e PNG.
6. Containerizar a aplicação.
7. Gerar build `linux/amd64` e `linux/arm64`.
8. Documentar adaptações para Raspberry Pi 5.

## Execução inicial

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/detect_person.py samples/input.jpg
```

> Adicione uma imagem em `samples/input.jpg` para o primeiro teste.

## Executor

Fabiano Barros
18/08/2026
