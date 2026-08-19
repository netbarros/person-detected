"""
Testes automatizados do contrato HTTP.

Objetivo pedagógico
-------------------
Uma suíte de testes precisa saber QUAL responsabilidade está validando.
Aqui queremos testar a API, não a qualidade estatística do YOLO.

Por isso a inferência real é substituída por um resultado conhecido:

    HTTP / FastAPI / serialização / PNG / alertas
        -> reais

    inferência YOLO
        -> mock determinístico

A inferência real continua sendo validada pelos testes E2E e benchmarks.
"""

from unittest.mock import patch

import cv2
import numpy as np

from app.detector import (
    PersonDetection,
    PersonDetector,
)


# ================================================================
# IMPORT DA API SEM CARREGAR YOLO REAL
# ================================================================
#
# `app.api` cria PersonDetector no escopo do módulo. Durante o import do
# teste neutralizamos apenas o __init__ para evitar carregar pesos e
# tornar a suíte rápida e independente de hardware.
# ================================================================

with patch.object(
    PersonDetector,
    "__init__",
    return_value=None,
):
    import app.api as api_module


from fastapi.testclient import TestClient


client = TestClient(api_module.app)

TEST_WIDTH = 880
TEST_HEIGHT = 587


# ================================================================
# IMAGEM SINTÉTICA
# ================================================================


def create_test_jpeg() -> bytes:
    """
    Cria uma imagem JPEG válida inteiramente em memória.

    O conteúdo visual é preto porque o modelo será simulado. O que
    interessa aqui é provar que a camada HTTP realmente recebe e
    decodifica uma imagem válida.
    """

    image = np.zeros(
        (
            TEST_HEIGHT,
            TEST_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
    )

    assert success

    return encoded.tobytes()


# ================================================================
# INFERÊNCIA DETERMINÍSTICA PARA O TESTE
# ================================================================


def fake_process_image(
    image: np.ndarray,
) -> tuple[
    list[PersonDetection],
    list[tuple[PersonDetection, str]],
    float,
]:
    """
    Simula uma pessoa em condição CRÍTICA.

    A bounding box foi escolhida para manter o mesmo exemplo didático do
    projeto. O foot_point continua sendo calculado pelo código real de
    `PersonDetection`.
    """

    detection = PersonDetection(
        confidence=0.932,
        bbox=(
            382,
            159,
            505,
            555,
        ),
    )

    return (
        [detection],
        [
            (
                detection,
                "CRÍTICO",
            )
        ],
        12.345,
    )


# ================================================================
# HEALTH
# ================================================================


def test_health() -> None:
    """Healthcheck deve responder rapidamente sem inferência."""

    response = client.get(
        "/health"
    )

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    assert response.json() == {
        "status": "ok",
        "model": "yolo11n.pt",
    }


# ================================================================
# ENDPOINT JSON
# ================================================================


def test_infer_returns_expected_json(
    monkeypatch,
) -> None:
    """
    Valida o contrato JSON incluindo a decisão de alerta proporcional.
    """

    monkeypatch.setattr(
        api_module,
        "process_image",
        fake_process_image,
    )

    response = client.post(
        "/api/v1/infer",
        files={
            "file": (
                "test.jpg",
                create_test_jpeg(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    body = response.json()

    # ------------------------------------------------------------
    # Metadados
    # ------------------------------------------------------------
    assert body["filename"] == "test.jpg"
    assert body["image"] == {
        "width": TEST_WIDTH,
        "height": TEST_HEIGHT,
    }
    assert body["model"] == "yolo11n.pt"
    assert body["confidence_threshold"] == 0.4
    assert body["inference_ms"] == 12.345
    assert body["persons_detected"] == 1

    # ------------------------------------------------------------
    # Detecção
    # ------------------------------------------------------------
    detection = body["detections"][0]

    assert detection["class_name"] == "person"
    assert detection["confidence"] == 0.932
    assert detection["bbox"] == {
        "x1": 382,
        "y1": 159,
        "x2": 505,
        "y2": 555,
    }

    # x = (382 + 505) // 2 = 443
    # y = y2 = 555
    assert detection["foot_point"] == {
        "x": 443,
        "y": 555,
    }
    assert detection["risk"] == "CRÍTICO"

    # ------------------------------------------------------------
    # Alerta proporcional
    # ------------------------------------------------------------
    # CRÍTICO deve gerar a maior severidade definida pelo protótipo.
    assert body["alert"] == {
        "active": True,
        "level": "CRITICAL",
        "source_risk": "CRÍTICO",
        "action": "REQUEST_IMMEDIATE_INTERVENTION",
        "message": (
            "Pessoa detectada na zona vermelha: "
            "solicitar intervenção imediata do sistema responsável."
        ),
    }


# ================================================================
# ENDPOINT PNG
# ================================================================


def test_infer_annotated_returns_valid_png(
    monkeypatch,
) -> None:
    """
    Valida PNG real, dimensões, desenho e headers de alerta.
    """

    monkeypatch.setattr(
        api_module,
        "process_image",
        fake_process_image,
    )

    response = client.post(
        "/api/v1/infer/annotated",
        files={
            "file": (
                "test.jpg",
                create_test_jpeg(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # ------------------------------------------------------------
    # O endpoint de imagem também comunica a decisão do frame.
    # ------------------------------------------------------------
    assert response.headers["x-alert-active"] == "true"
    assert response.headers["x-alert-level"] == "CRITICAL"
    assert (
        response.headers["x-alert-action"]
        == "REQUEST_IMMEDIATE_INTERVENTION"
    )

    # ------------------------------------------------------------
    # Assinatura PNG oficial.
    # ------------------------------------------------------------
    assert response.content.startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    # ------------------------------------------------------------
    # Decodifica a resposta para provar que não é apenas um conjunto de
    # bytes com prefixo correto.
    # ------------------------------------------------------------
    png_buffer = np.frombuffer(
        response.content,
        dtype=np.uint8,
    )

    decoded_image = cv2.imdecode(
        png_buffer,
        cv2.IMREAD_COLOR,
    )

    assert decoded_image is not None

    height, width = decoded_image.shape[:2]

    assert width == TEST_WIDTH
    assert height == TEST_HEIGHT

    # A imagem de entrada era totalmente preta. Como o visualizador
    # desenha zonas, box, ponto e textos, o resultado precisa conter
    # pixels não nulos.
    assert int(decoded_image.sum()) > 0


# ================================================================
# TRATAMENTO DE ERRO
# ================================================================


def test_infer_rejects_invalid_image() -> None:
    """Bytes que não formam uma imagem devem resultar em HTTP 400."""

    response = client.post(
        "/api/v1/infer",
        files={
            "file": (
                "invalid.jpg",
                b"isto-nao-e-uma-imagem",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "Não foi possível decodificar "
        "o arquivo como imagem."
    )
