# --------------------------------------------------
# Imagem base
# --------------------------------------------------
#
# Python 3.11 sobre Debian Slim.
#
# A imagem slim reduz componentes desnecessários
# e será adequada posteriormente para builds:
#
#     linux/amd64
#     linux/arm64
#
FROM python:3.11-slim


# --------------------------------------------------
# Metadados
# --------------------------------------------------

LABEL project="person-detected"
LABEL description="Edge AI para deteccao de pessoas e classificacao de risco"


# --------------------------------------------------
# Variáveis de ambiente
# --------------------------------------------------
#
# PYTHONDONTWRITEBYTECODE
#     evita gerar arquivos .pyc
#
# PYTHONUNBUFFERED
#     envia logs imediatamente para stdout
#
# PIP_NO_CACHE_DIR
#     reduz cache desnecessário dentro da imagem
#
# YOLO_CONFIG_DIR
#     fornece ao Ultralytics uma área gravável
#
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics


# --------------------------------------------------
# Diretório de trabalho
# --------------------------------------------------

WORKDIR /app


# --------------------------------------------------
# Dependências do sistema
# --------------------------------------------------
#
# libgomp1
#     implementação OpenMP utilizada por bibliotecas
#     numéricas como PyTorch.
#
# Não instalamos X11/Qt porque a aplicação executada
# no container não utiliza cv2.imshow().
#
RUN apt-get update \
    && apt-get install -y \
    --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# --------------------------------------------------
# Dependências Python
# --------------------------------------------------
#
# Copiar requirements antes do código permite
# reaproveitar cache do Docker quando apenas
# arquivos Python são modificados.
#
COPY requirements.txt .


RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt


# --------------------------------------------------
# Normalização do OpenCV para ambiente headless
# --------------------------------------------------
#
# O Ultralytics declara opencv-python como dependência.
#
# Entretanto, dentro de um container de API não usamos:
#
#     cv2.imshow()
#     cv2.namedWindow()
#
# Portanto queremos exclusivamente:
#
#     opencv-python-headless
#
# As variantes OpenCV compartilham o mesmo módulo
# Python chamado "cv2", por isso não é recomendável
# manter opencv-python e opencv-python-headless juntos.
#
# Primeiro removemos a variante desktop que pode ter
# sido instalada pelo Ultralytics.
#
# Depois reinstalamos a variante headless conhecida
# e validada para este ambiente.
#
RUN pip uninstall -y opencv-python \
    && pip install \
    --force-reinstall \
    --no-deps \
    opencv-python-headless==4.12.0.88


# --------------------------------------------------
# Validação do OpenCV durante o build
# --------------------------------------------------
#
# Falhamos o build imediatamente caso cv2 não consiga
# ser importado.
#
# Isso evita descobrir o problema somente quando o
# container já estiver sendo executado.
#
RUN python -c "import cv2; print('OpenCV:', cv2.__version__)"


# --------------------------------------------------
# Modelo YOLO
# --------------------------------------------------
#
# Carregamos o modelo durante o BUILD.
#
# Se os pesos ainda não estiverem presentes,
# o Ultralytics fará o download nesse momento.
#
# Benefício:
#
# o container em produção não precisa baixar o
# modelo no primeiro request.
#
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('YOLO OK')"


# --------------------------------------------------
# Código da aplicação
# --------------------------------------------------
#
# Copiamos apenas os componentes necessários para
# executar a API.
#
COPY app ./app
COPY config ./config


# --------------------------------------------------
# Porta HTTP
# --------------------------------------------------

EXPOSE 8000


# --------------------------------------------------
# Health Check
# --------------------------------------------------
#
# Usamos urllib da própria biblioteca padrão.
#
# Assim não precisamos instalar curl dentro da
# imagem apenas para executar health check.
#
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1


# --------------------------------------------------
# Inicialização
# --------------------------------------------------
#
# 0.0.0.0 é necessário para que o Uvicorn aceite
# conexões vindas de fora do container.
#
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
