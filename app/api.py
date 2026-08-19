"""
API HTTP do projeto Person Detected / Edge AI Risk Zone Monitor.

Este módulo expõe o pipeline completo em duas formas:

1. JSON estruturado
   POST /api/v1/infer

2. Imagem PNG anotada
   POST /api/v1/infer/annotated

Além da detecção e classificação espacial, a API agora também aplica a
política de alertas proporcionais exigida pelo cenário selecionado.

Arquitetura resumida
--------------------

    upload
      ↓
    decode OpenCV
      ↓
    YOLO11n
      ↓
    PersonDetection
      ↓
    foot_point
      ↓
    RiskZoneClassifier
      ↓
    SEGURO / ALERTA / CRÍTICO
      ↓
    AlertDecision
      ↓
    JSON ou PNG

A IA é utilizada para percepção. A decisão espacial e a política de
alerta permanecem determinísticas e testáveis.
"""

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

from app.alerts import (
    AlertDecision,
    AlertDispatcher,
    decide_alert,
    select_highest_risk,
)
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


# ================================================================
# CAMINHOS DO PROJETO
# ================================================================

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "zones.json"


# ================================================================
# APLICAÇÃO FASTAPI
# ================================================================

app = FastAPI(
    title="Person Risk Detection API",
    description=(
        "API de Edge AI para detecção de pessoas, classificação "
        "de risco por zonas e acionamento proporcional de alertas."
    ),
    version="1.1.0",
)


# ================================================================
# COMPONENTES CARREGADOS UMA ÚNICA VEZ
# ================================================================
#
# Por que fora dos endpoints?
#
# Carregar YOLO é uma operação relativamente cara. Se criássemos uma
# nova instância do modelo a cada requisição, introduziríamos latência
# desnecessária e consumo adicional de memória.
#
# Em um processo Uvicorn, estes componentes são carregados uma vez e
# reutilizados pelas requisições atendidas por esse processo.
# ================================================================

detector = PersonDetector()
zone_classifier = RiskZoneClassifier(CONFIG_PATH)
alert_dispatcher = AlertDispatcher()


# ================================================================
# CONTRATOS JSON
# ================================================================


class ImageInfo(BaseModel):
    """Dimensões da imagem recebida pela API."""

    width: int
    height: int


class BoundingBoxResponse(BaseModel):
    """
    Bounding box no formato xyxy.

        (x1, y1)
            +----------------+
            |                |
            |     pessoa     |
            |                |
            +----------------+
                         (x2, y2)
    """

    x1: int
    y1: int
    x2: int
    y2: int


class PointResponse(BaseModel):
    """Ponto bidimensional utilizado para expor o foot_point."""

    x: int
    y: int


class PersonDetectionResponse(BaseModel):
    """Representação HTTP de uma pessoa detectada."""

    class_name: str
    confidence: float
    bbox: BoundingBoxResponse
    foot_point: PointResponse
    risk: str


class AlertResponse(BaseModel):
    """
    Estado agregado do alerta para o frame analisado.

    Se houver várias pessoas, o alerta representa a maior severidade.

    Exemplo:

        pessoa A -> SEGURO
        pessoa B -> CRÍTICO

        alert.level -> CRITICAL
    """

    active: bool
    level: str
    source_risk: str
    action: str
    message: str


class InferenceResponse(BaseModel):
    """Contrato completo do endpoint JSON de inferência."""

    filename: str | None
    image: ImageInfo
    model: str
    confidence_threshold: float
    inference_ms: float
    persons_detected: int
    detections: list[PersonDetectionResponse]
    alert: AlertResponse


# ================================================================
# PRÉ-PROCESSAMENTO HTTP
# ================================================================


async def decode_uploaded_image(
    file: UploadFile,
) -> np.ndarray:
    """
    Converte o upload HTTP em imagem OpenCV BGR.

    Este é o primeiro estágio explícito do pré-processamento:

        UploadFile
            ↓
        bytes
            ↓
        np.frombuffer
            ↓
        cv2.imdecode
            ↓
        imagem BGR

    Depois disso, `PersonDetector.detect()` entrega a imagem ao runner
    do Ultralytics, que executa internamente o pré-processamento
    específico da rede, como adequação ao tamanho de entrada, conversão
    para tensor e normalização compatível com o modelo.

    Evitamos salvar arquivo temporário: a imagem permanece em memória.
    """

    # ------------------------------------------------------------
    # 1. Lê o corpo enviado no campo multipart `file`.
    # ------------------------------------------------------------
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Arquivo de imagem vazio.",
        )

    # ------------------------------------------------------------
    # 2. bytes -> ndarray de bytes.
    # ------------------------------------------------------------
    image_buffer = np.frombuffer(
        contents,
        dtype=np.uint8,
    )

    # ------------------------------------------------------------
    # 3. Decode real do JPEG/PNG/etc. para matriz BGR.
    # ------------------------------------------------------------
    image = cv2.imdecode(
        image_buffer,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Não foi possível decodificar "
                "o arquivo como imagem."
            ),
        )

    return image


# ================================================================
# INFERÊNCIA + PÓS-PROCESSAMENTO ESPACIAL
# ================================================================


def process_image(
    image: np.ndarray,
) -> tuple[
    list[PersonDetection],
    list[tuple[PersonDetection, str]],
    float,
]:
    """
    Executa o núcleo funcional do projeto.

    O método mede somente a chamada do detector para que `inference_ms`
    tenha significado específico e não seja confundido com latência E2E.

    Etapas:

        imagem BGR
            ↓
        YOLO
            ↓
        detecções de person
            ↓
        foot_point
            ↓
        teste geométrico nas zonas
            ↓
        SEGURO / ALERTA / CRÍTICO

    Retorno:

        detections
            detecções de pessoas.

        risk_results
            pares (PersonDetection, risco).

        inference_ms
            duração somente da chamada ao detector YOLO.
    """

    height, width = image.shape[:2]

    # ------------------------------------------------------------
    # Inferência da rede neural.
    # ------------------------------------------------------------
    start = perf_counter()
    detections = detector.detect(image)
    inference_ms = (perf_counter() - start) * 1000.0

    # ------------------------------------------------------------
    # Pós-processamento determinístico.
    # ------------------------------------------------------------
    risk_results: list[tuple[PersonDetection, str]] = []

    for detection in detections:
        risk = zone_classifier.classify(
            point=detection.foot_point,
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


# ================================================================
# POLÍTICA E DESPACHO DE ALERTA
# ================================================================


def evaluate_and_dispatch_alert(
    risk_results: list[tuple[PersonDetection, str]],
) -> AlertDecision:
    """
    Calcula e despacha o alerta global do frame.

    A decisão é proporcional à maior severidade observada:

        SEGURO  -> NONE
        ALERTA  -> WARNING
        CRÍTICO -> CRITICAL

    A lógica de mapeamento está em `app.alerts`, não aqui. A API apenas
    orquestra os componentes.
    """

    highest_risk = select_highest_risk(
        risk
        for _detection, risk in risk_results
    )

    decision = decide_alert(highest_risk)

    # Resposta automática disponível neste protótipo: despacho em log.
    alert_dispatcher.dispatch(decision)

    return decision


# ================================================================
# CONVERSÃO DO ALERTA PARA O CONTRATO HTTP
# ================================================================


def build_alert_response(
    decision: AlertDecision,
) -> AlertResponse:
    """Converte o objeto de domínio em resposta serializável."""

    return AlertResponse(
        active=decision.active,
        level=decision.level.value,
        source_risk=decision.source_risk,
        action=decision.action.value,
        message=decision.message,
    )


# ================================================================
# CONVERSÃO PARA JSON
# ================================================================


def build_json_response(
    filename: str | None,
    image: np.ndarray,
    risk_results: list[tuple[PersonDetection, str]],
    inference_ms: float,
    alert_decision: AlertDecision,
) -> InferenceResponse:
    """
    Converte o resultado interno para o contrato público da API.

    Essa separação evita acoplar o domínio (`PersonDetection`) aos
    modelos HTTP (`PersonDetectionResponse`).
    """

    height, width = image.shape[:2]

    response_detections: list[PersonDetectionResponse] = []

    for detection, risk in risk_results:
        x1, y1, x2, y2 = detection.bbox
        foot_x, foot_y = detection.foot_point

        response_detections.append(
            PersonDetectionResponse(
                class_name="person",
                confidence=round(
                    detection.confidence,
                    6,
                ),
                bbox=BoundingBoxResponse(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                ),
                foot_point=PointResponse(
                    x=foot_x,
                    y=foot_y,
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
        confidence_threshold=DEFAULT_CONFIDENCE,
        inference_ms=round(
            inference_ms,
            3,
        ),
        persons_detected=len(response_detections),
        detections=response_detections,
        alert=build_alert_response(alert_decision),
    )


# ================================================================
# ENDPOINT 0 - HEALTH CHECK
# ================================================================


@app.get("/health")
def health() -> dict:
    """
    Informa que o processo HTTP está ativo.

    Não executa uma inferência, portanto é adequado para healthcheck do
    Docker/Compose sem gerar carga constante no modelo.
    """

    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
    }


# ================================================================
# ENDPOINT 1 - INFERÊNCIA JSON
# ================================================================


@app.post(
    "/api/v1/infer",
    response_model=InferenceResponse,
)
async def infer(
    file: UploadFile = File(...),
) -> InferenceResponse:
    """
    Recebe imagem via multipart/form-data e retorna resultado JSON.

    Além das detecções, o contrato inclui o alerta global do frame.
    """

    # 1. Pré-processamento HTTP/OpenCV.
    image = await decode_uploaded_image(file)

    # 2. Inferência e pós-processamento espacial.
    (
        _detections,
        risk_results,
        inference_ms,
    ) = process_image(image)

    # 3. Resposta automática proporcional ao risco.
    alert_decision = evaluate_and_dispatch_alert(
        risk_results
    )

    # 4. Contrato JSON.
    return build_json_response(
        filename=file.filename,
        image=image,
        risk_results=risk_results,
        inference_ms=inference_ms,
        alert_decision=alert_decision,
    )


# ================================================================
# ENDPOINT 2 - INFERÊNCIA PNG ANOTADO
# ================================================================


@app.post(
    "/api/v1/infer/annotated",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {}
            },
            "description": (
                "Imagem PNG anotada com zonas e riscos."
            ),
        }
    },
)
async def infer_annotated(
    file: UploadFile = File(...),
) -> Response:
    """
    Recebe imagem e devolve PNG anotado.

    O nível de alerta também é exposto em headers ASCII para que o
    consumidor da imagem possa conhecer a decisão do frame sem precisar
    executar uma segunda chamada ao endpoint JSON.
    """

    # 1. Pré-processamento.
    image = await decode_uploaded_image(file)
    height, width = image.shape[:2]

    # 2. Inferência + classificação espacial.
    (
        _detections,
        risk_results,
        _inference_ms,
    ) = process_image(image)

    # 3. Política de alerta proporcional.
    alert_decision = evaluate_and_dispatch_alert(
        risk_results
    )

    # 4. Polígonos na resolução atual da imagem.
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

    # 5. Renderização visual compartilhada com o script local.
    annotated = annotate_risk_image(
        image=image,
        risk_results=risk_results,
        yellow_polygon=yellow_polygon,
        red_polygon=red_polygon,
    )

    # 6. ndarray BGR -> PNG codificado em memória.
    success, encoded_image = cv2.imencode(
        ".png",
        annotated,
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Falha ao gerar imagem PNG anotada.",
        )

    # 7. Headers de alerta usam códigos ASCII para compatibilidade HTTP.
    headers = {
        "X-Alert-Active": (
            "true"
            if alert_decision.active
            else "false"
        ),
        "X-Alert-Level": alert_decision.level.value,
        "X-Alert-Action": alert_decision.action.value,
    }

    return Response(
        content=encoded_image.tobytes(),
        media_type="image/png",
        headers=headers,
    )
