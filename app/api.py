from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
)
from app.detector import PersonDetector
from app.zones import RiskZoneClassifier

# -------------------------------------------------
# Caminhos base do projeto
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config" / "zones.json"


# -------------------------------------------------
# Aplicação FastAPI
# -------------------------------------------------

# FastAPI é responsável por:
#
# - receber requisições HTTP
# - validar parâmetros
# - gerar documentação OpenAPI
# - serializar objetos Python para JSON

app = FastAPI(
    title="Person Risk Detection API",
    description=(
        "API de Edge AI para detecção de pessoas e classificação de risco por zonas."
    ),
    version="1.0.0",
)


# -------------------------------------------------
# Componentes da aplicação
# -------------------------------------------------

# O modelo é carregado uma única vez quando
# o processo da API inicia.
#
# Isso é importante.
#
# NÃO queremos carregar o YOLO novamente
# a cada requisição HTTP, porque carregar
# os pesos do modelo possui custo significativo.

detector = PersonDetector()


# O mesmo princípio vale para a configuração
# das zonas.

zone_classifier = RiskZoneClassifier(CONFIG_PATH)


# =================================================
# MODELOS DE RESPOSTA
# =================================================

# Os modelos Pydantic descrevem formalmente
# o JSON devolvido pela API.
#
# Isso traz:
#
# - validação
# - documentação automática
# - contrato explícito da API


# -------------------------------------------------
# Informações da imagem
# -------------------------------------------------


class ImageInfo(BaseModel):
    """
    Dimensões da imagem recebida pela API.
    """

    width: int
    height: int


# -------------------------------------------------
# Bounding Box
# -------------------------------------------------


class BoundingBoxResponse(BaseModel):
    """
    Representa a bounding box da pessoa.

    Formato:

        (x1, y1) --------
           |             |
           |   pessoa    |
           |             |
           -------- (x2, y2)
    """

    x1: int
    y1: int

    x2: int
    y2: int


# -------------------------------------------------
# Ponto na imagem
# -------------------------------------------------


class PointResponse(BaseModel):
    """
    Representa uma coordenada x, y.

    Neste projeto será utilizado principalmente
    para representar o foot_point.
    """

    x: int
    y: int


# -------------------------------------------------
# Detecção individual
# -------------------------------------------------


class PersonDetectionResponse(BaseModel):
    """
    Representa uma pessoa detectada.

    Contém:

        classe
        confiança
        bounding box
        ponto dos pés
        risco
    """

    class_name: str

    confidence: float

    bbox: BoundingBoxResponse

    foot_point: PointResponse

    risk: str


# -------------------------------------------------
# Resposta completa da inferência
# -------------------------------------------------


class InferenceResponse(BaseModel):
    """
    Contrato JSON retornado pelo endpoint.

    Além das pessoas detectadas, incluímos
    algumas informações úteis para avaliação
    de desempenho do sistema.
    """

    filename: str | None

    image: ImageInfo

    model: str

    confidence_threshold: float

    inference_ms: float

    persons_detected: int

    detections: list[PersonDetectionResponse]


# =================================================
# ENDPOINTS
# =================================================


# -------------------------------------------------
# Health Check
# -------------------------------------------------


@app.get("/health")
def health() -> dict:
    """
    Health check simples da aplicação.

    Esse endpoint não executa inferência.

    Ele existe para que:

        Docker
        orquestradores
        monitoramento
        operadores

    consigam verificar se o serviço está ativo.
    """

    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
    }


# -------------------------------------------------
# Inferência JSON
# -------------------------------------------------


@app.post(
    "/api/v1/infer",
    response_model=InferenceResponse,
)
async def infer(
    file: UploadFile = File(...),
) -> InferenceResponse:
    """
    Recebe uma imagem via HTTP e devolve
    o resultado estruturado em JSON.

    Pipeline completo:

        upload HTTP
            ↓
        bytes
            ↓
        NumPy
            ↓
        OpenCV
            ↓
        YOLO
            ↓
        bounding boxes
            ↓
        foot_point
            ↓
        motor de zonas
            ↓
        SEGURO / ALERTA / CRÍTICO
            ↓
        JSON
    """

    # -------------------------------------------------
    # 1. Leitura do arquivo enviado
    # -------------------------------------------------

    # UploadFile representa o arquivo recebido
    # pelo protocolo HTTP multipart/form-data.

    contents = await file.read()

    # Não faz sentido continuar se o cliente
    # enviou um arquivo vazio.

    if not contents:
        raise HTTPException(
            status_code=400,
            detail=("Arquivo de imagem vazio."),
        )

    # -------------------------------------------------
    # 2. Bytes -> NumPy
    # -------------------------------------------------

    # Uma requisição HTTP entrega bytes.
    #
    # OpenCV, porém, trabalha com matrizes.
    #
    # Primeiro transformamos os bytes em
    # um vetor NumPy.

    image_buffer = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    # -------------------------------------------------
    # 3. NumPy -> imagem OpenCV
    # -------------------------------------------------

    # imdecode interpreta os bytes como:
    #
    # JPEG
    # PNG
    # etc.
    #
    # e produz uma matriz BGR.

    image = cv2.imdecode(
        image_buffer,
        cv2.IMREAD_COLOR,
    )

    # Se não for uma imagem válida,
    # imdecode devolve None.

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=("Não foi possível decodificar o arquivo como imagem."),
        )

    # -------------------------------------------------
    # 4. Dimensões da imagem
    # -------------------------------------------------

    height, width = image.shape[:2]

    # -------------------------------------------------
    # 5. Medição de desempenho
    # -------------------------------------------------

    # perf_counter é adequado para medir
    # intervalos de tempo curtos.

    start = perf_counter()

    # -------------------------------------------------
    # 6. Inferência YOLO
    # -------------------------------------------------

    # Aqui está uma diferença importante
    # em relação ao script anterior.
    #
    # Não estamos enviando:
    #
    #     samples/input.jpg
    #
    # Estamos enviando diretamente a matriz
    # da imagem que chegou pela rede.

    detections = detector.detect(image)

    # -------------------------------------------------
    # 7. Tempo de inferência
    # -------------------------------------------------

    inference_ms = (perf_counter() - start) * 1000.0

    # -------------------------------------------------
    # 8. Pós-processamento
    # -------------------------------------------------

    response_detections: list[PersonDetectionResponse] = []

    for detection in detections:
        # ---------------------------------------------
        # Bounding box
        # ---------------------------------------------

        x1, y1, x2, y2 = detection.bbox

        # ---------------------------------------------
        # Foot point
        # ---------------------------------------------

        foot_x, foot_y = detection.foot_point

        # ---------------------------------------------
        # Classificação espacial
        # ---------------------------------------------

        risk = zone_classifier.classify(
            point=(detection.foot_point),
            width=width,
            height=height,
        )

        # ---------------------------------------------
        # Converte o resultado interno para
        # nosso contrato HTTP
        # ---------------------------------------------

        response_detections.append(
            PersonDetectionResponse(
                class_name="person",
                confidence=round(
                    detection.confidence,
                    6,
                ),
                bbox=(
                    BoundingBoxResponse(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                ),
                foot_point=(
                    PointResponse(
                        x=foot_x,
                        y=foot_y,
                    )
                ),
                risk=risk,
            )
        )

    # -------------------------------------------------
    # 9. Resposta HTTP
    # -------------------------------------------------

    return InferenceResponse(
        filename=file.filename,
        image=ImageInfo(
            width=width,
            height=height,
        ),
        model=DEFAULT_MODEL,
        confidence_threshold=(DEFAULT_CONFIDENCE),
        inference_ms=round(
            inference_ms,
            3,
        ),
        persons_detected=len(response_detections),
        detections=(response_detections),
    )
