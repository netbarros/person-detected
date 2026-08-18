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
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
)
from app.detector import (
    PersonDetection,
    PersonDetector,
)
from app.visualizer import annotate_risk_image
from app.zones import RiskZoneClassifier

# -------------------------------------------------
# Caminhos base do projeto
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config" / "zones.json"


# -------------------------------------------------
# Aplicação FastAPI
# -------------------------------------------------

app = FastAPI(
    title="Person Risk Detection API",
    description=(
        "API de Edge AI para detecção de pessoas e classificação de risco por zonas."
    ),
    version="1.0.0",
)


# =================================================
# COMPONENTES DA APLICAÇÃO
# =================================================

# O modelo YOLO é carregado UMA única vez.
#
# Isso é importante em uma API.
#
# Se fizéssemos:
#
#     detector = PersonDetector()
#
# dentro de cada endpoint, o modelo poderia ser
# recarregado a cada requisição.
#
# Isso aumentaria muito a latência.

detector = PersonDetector()


# O classificador de zonas também é criado uma vez.
#
# Ele carrega o arquivo:
#
#     config/zones.json

zone_classifier = RiskZoneClassifier(CONFIG_PATH)


# =================================================
# MODELOS DA RESPOSTA JSON
# =================================================


class ImageInfo(BaseModel):
    """
    Informações básicas da imagem recebida.
    """

    width: int
    height: int


class BoundingBoxResponse(BaseModel):
    """
    Bounding box da pessoa.

    Formato:

        x1, y1
          +----------------+
          |                |
          |     pessoa     |
          |                |
          +----------------+
                         x2, y2
    """

    x1: int
    y1: int
    x2: int
    y2: int


class PointResponse(BaseModel):
    """
    Coordenada de um ponto na imagem.

    Neste projeto usamos para representar
    principalmente o foot_point.
    """

    x: int
    y: int


class PersonDetectionResponse(BaseModel):
    """
    Resultado estruturado para uma pessoa.
    """

    class_name: str
    confidence: float

    bbox: BoundingBoxResponse

    foot_point: PointResponse

    risk: str


class InferenceResponse(BaseModel):
    """
    Contrato completo da resposta JSON.
    """

    filename: str | None

    image: ImageInfo

    model: str

    confidence_threshold: float

    inference_ms: float

    persons_detected: int

    detections: list[PersonDetectionResponse]


# =================================================
# FUNÇÕES AUXILIARES
# =================================================


async def decode_uploaded_image(
    file: UploadFile,
) -> np.ndarray:
    """
    Converte o arquivo recebido via HTTP em
    uma imagem OpenCV.

    Pipeline:

        UploadFile
            ↓
        bytes
            ↓
        numpy.ndarray
            ↓
        cv2.imdecode
            ↓
        imagem BGR


    Essa função é compartilhada pelos dois endpoints:

        /api/v1/infer

        /api/v1/infer/annotated

    Assim evitamos duplicação de código.
    """

    # ---------------------------------------------
    # 1. Lê os bytes do upload
    # ---------------------------------------------

    contents = await file.read()

    # ---------------------------------------------
    # 2. Validação de arquivo vazio
    # ---------------------------------------------

    if not contents:
        raise HTTPException(
            status_code=400,
            detail=("Arquivo de imagem vazio."),
        )

    # ---------------------------------------------
    # 3. Bytes -> vetor NumPy
    # ---------------------------------------------

    image_buffer = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    # ---------------------------------------------
    # 4. NumPy -> imagem OpenCV
    # ---------------------------------------------

    image = cv2.imdecode(
        image_buffer,
        cv2.IMREAD_COLOR,
    )

    # ---------------------------------------------
    # 5. Validação da imagem
    # ---------------------------------------------

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=("Não foi possível decodificar o arquivo como imagem."),
        )

    return image


# -------------------------------------------------
# Inferência + classificação espacial
# -------------------------------------------------


def process_image(
    image: np.ndarray,
) -> tuple[
    list[PersonDetection],
    list[
        tuple[
            PersonDetection,
            str,
        ]
    ],
    float,
]:
    """
    Executa o núcleo da aplicação.

    Essa função é independente de HTTP.

    Pipeline:

        imagem
            ↓
        YOLO
            ↓
        PersonDetection
            ↓
        foot_point
            ↓
        RiskZoneClassifier
            ↓
        SEGURO / ALERTA / CRÍTICO


    Retorna três elementos:

        detections

        risk_results

        inference_ms


    risk_results possui estrutura:

        [
            (
                PersonDetection,
                "SEGURO",
            ),
            (
                PersonDetection,
                "CRÍTICO",
            ),
        ]
    """

    height, width = image.shape[:2]

    # ---------------------------------------------
    # Medição da inferência
    # ---------------------------------------------

    # Aqui medimos especificamente o trecho
    # que executa o detector YOLO.
    #
    # Depois faremos um benchmark mais rigoroso
    # com múltiplas execuções.

    start = perf_counter()

    detections = detector.detect(image)

    inference_ms = (perf_counter() - start) * 1000.0

    # ---------------------------------------------
    # Classificação das zonas
    # ---------------------------------------------

    risk_results: list[
        tuple[
            PersonDetection,
            str,
        ]
    ] = []

    for detection in detections:
        risk = zone_classifier.classify(
            point=(detection.foot_point),
            width=width,
            height=height,
        )

        risk_results.append(
            (
                detection,
                risk,
            )
        )

    return (
        detections,
        risk_results,
        inference_ms,
    )


# -------------------------------------------------
# Conversão para resposta JSON
# -------------------------------------------------


def build_json_response(
    filename: str | None,
    image: np.ndarray,
    risk_results: list[
        tuple[
            PersonDetection,
            str,
        ]
    ],
    inference_ms: float,
) -> InferenceResponse:
    """
    Converte nosso resultado interno para
    o contrato JSON da API.

    Essa separação é interessante porque:

        domínio interno
            PersonDetection

    não precisa ser igual a:

        contrato externo
            PersonDetectionResponse
    """

    height, width = image.shape[:2]

    response_detections: list[PersonDetectionResponse] = []

    for (
        detection,
        risk,
    ) in risk_results:
        x1, y1, x2, y2 = detection.bbox

        foot_x, foot_y = detection.foot_point

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

    return InferenceResponse(
        filename=filename,
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


# =================================================
# ENDPOINTS
# =================================================


# -------------------------------------------------
# Health Check
# -------------------------------------------------


@app.get("/health")
def health() -> dict:
    """
    Verifica se o serviço está ativo.

    Não executa YOLO.
    """

    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
    }


# -------------------------------------------------
# Endpoint 1
#
# Inferência com retorno JSON
# -------------------------------------------------


@app.post(
    "/api/v1/infer",
    response_model=InferenceResponse,
)
async def infer(
    file: UploadFile = File(...),
) -> InferenceResponse:
    """
    Recebe uma imagem e devolve JSON.

    Exemplo conceitual:

        POST /api/v1/infer

            imagem.jpg
                ↓

        {
            "persons_detected": 2,
            "detections": [...]
        }
    """

    # ---------------------------------------------
    # 1. Decodificação
    # ---------------------------------------------

    image = await decode_uploaded_image(file)

    # ---------------------------------------------
    # 2. Inferência + classificação
    # ---------------------------------------------

    (
        _detections,
        risk_results,
        inference_ms,
    ) = process_image(image)

    # ---------------------------------------------
    # 3. Resposta JSON
    # ---------------------------------------------

    return build_json_response(
        filename=file.filename,
        image=image,
        risk_results=risk_results,
        inference_ms=inference_ms,
    )


# -------------------------------------------------
# Endpoint 2
#
# Inferência com retorno PNG
# -------------------------------------------------


@app.post(
    "/api/v1/infer/annotated",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": ("Imagem PNG anotada com zonas e riscos."),
        }
    },
)
async def infer_annotated(
    file: UploadFile = File(...),
) -> Response:
    """
    Recebe uma imagem e devolve outra imagem
    já anotada.

    Pipeline:

        upload
            ↓
        OpenCV
            ↓
        YOLO
            ↓
        RiskZoneClassifier
            ↓
        visualizer
            ↓
        PNG


    Esse endpoint reutiliza exatamente os mesmos
    componentes usados pelo script local.
    """

    # ---------------------------------------------
    # 1. Decodificação
    # ---------------------------------------------

    image = await decode_uploaded_image(file)

    height, width = image.shape[:2]

    # ---------------------------------------------
    # 2. Inferência + classificação
    # ---------------------------------------------

    (
        _detections,
        risk_results,
        _inference_ms,
    ) = process_image(image)

    # ---------------------------------------------
    # 3. Recuperação das zonas
    # ---------------------------------------------

    yellow_polygon = zone_classifier.get_polygon(
        "yellow",
        width,
        height,
    )

    red_polygon = zone_classifier.get_polygon(
        "red",
        width,
        height,
    )

    # ---------------------------------------------
    # 4. Renderização
    # ---------------------------------------------

    annotated = annotate_risk_image(
        image=image,
        risk_results=(risk_results),
        yellow_polygon=(yellow_polygon),
        red_polygon=(red_polygon),
    )

    # ---------------------------------------------
    # 5. Matriz OpenCV -> PNG
    # ---------------------------------------------

    # Até aqui temos uma matriz NumPy.
    #
    # Uma resposta HTTP precisa enviar bytes.
    #
    # cv2.imencode converte a imagem em memória
    # para o formato PNG.

    success, encoded_image = cv2.imencode(
        ".png",
        annotated,
    )

    # ---------------------------------------------
    # 6. Validação da codificação
    # ---------------------------------------------

    if not success:
        raise HTTPException(
            status_code=500,
            detail=("Falha ao gerar imagem PNG anotada."),
        )

    # ---------------------------------------------
    # 7. Resposta HTTP
    # ---------------------------------------------

    return Response(
        content=(encoded_image.tobytes()),
        media_type="image/png",
    )
