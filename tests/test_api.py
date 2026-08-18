"""
Testes automatizados da API FastAPI.

Objetivo deste arquivo
----------------------

Aqui queremos validar o CONTRATO HTTP da aplicação.

Isso é diferente de validar o modelo YOLO.

Os testes verificam:

    GET /health

    POST /api/v1/infer

    POST /api/v1/infer/annotated

    tratamento de imagem inválida


Uma decisão importante é NÃO executar inferência YOLO real
durante estes testes.

Motivos:

    1. testes ficam rápidos;

    2. não dependem do peso do modelo;

    3. não dependem do hardware;

    4. não confundimos teste de API com teste de IA;

    5. o resultado passa a ser determinístico.


A inferência real já é validada separadamente por:

    scripts/detect_person.py

    scripts/benchmark.py

    scripts/benchmark_api.py

    testes E2E manuais via curl.
"""


# ============================================================
# IMPORTS
# ============================================================

from unittest.mock import patch

import cv2
import numpy as np
from app.detector import (
    PersonDetection,
    PersonDetector,
)

# ============================================================
# EVITANDO CARREGAMENTO REAL DO YOLO
# ============================================================
#
# Existe um detalhe na arquitetura atual da API:
#
#     app/api.py
#
# possui:
#
#     detector = PersonDetector()
#
# no nível do módulo.
#
# Isso significa que simplesmente importar:
#
#     app.api
#
# faria o YOLO ser carregado.
#
#
# Para o teste de contrato HTTP isso é desnecessário.
#
# Portanto, durante SOMENTE o import de app.api,
# substituímos temporariamente o __init__ do
# PersonDetector.
#
# Assim:
#
#     app.api
#
# é carregado normalmente,
#
# mas:
#
#     YOLO(...)
#
# não é inicializado.
#
#
# Depois substituiremos process_image() por uma
# implementação falsa e determinística.
#
# ============================================================

with patch.object(
    PersonDetector,
    "__init__",
    return_value=None,
):
    import app.api as api_module


# Só depois do módulo estar carregado criamos
# o TestClient.

from fastapi.testclient import TestClient

client = TestClient(api_module.app)


# ============================================================
# CONSTANTES DO TESTE
# ============================================================

TEST_WIDTH = 880
TEST_HEIGHT = 587


# ============================================================
# IMAGEM JPEG SINTÉTICA
# ============================================================


def create_test_jpeg() -> bytes:
    """
    Cria uma pequena imagem de teste em memória.

    Apesar de usarmos as mesmas dimensões da imagem
    real do projeto:

        880 x 587

    não precisamos abrir:

        samples/input.jpg


    Vantagem:

        o teste da API não depende de um arquivo externo.


    Pipeline:

        matriz NumPy
            ↓
        OpenCV
            ↓
        JPEG em memória
            ↓
        bytes
            ↓
        upload HTTP


    A imagem é preta porque o conteúdo visual não é
    importante para este teste.

    A detecção será simulada.
    """

    # --------------------------------------------------------
    # Matriz de imagem
    # --------------------------------------------------------
    #
    # OpenCV trabalha no formato:
    #
    #     altura x largura x canais
    #
    # Portanto:
    #
    #     (587, 880, 3)
    #
    # representa uma imagem BGR.
    #
    # --------------------------------------------------------

    image = np.zeros(
        (
            TEST_HEIGHT,
            TEST_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Codificação JPEG
    # --------------------------------------------------------

    success, encoded = cv2.imencode(
        ".jpg",
        image,
    )

    assert success, "Falha ao criar imagem JPEG sintética para o teste."

    # --------------------------------------------------------
    # OpenCV retorna ndarray.
    #
    # HTTP precisa de bytes.
    # --------------------------------------------------------

    return encoded.tobytes()


# ============================================================
# INFERÊNCIA FALSA
# ============================================================


def fake_process_image(
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
    Substitui process_image() durante os testes.

    Não executa YOLO.

    Criamos manualmente uma pessoa conhecida.

    Isso nos permite saber exatamente qual deveria
    ser a resposta da API.


    Simulação:

        confiança:
            0.932

        bounding box:
            (382, 159, 505, 555)

        foot_point:
            calculado automaticamente por
            PersonDetection

        risco:
            CRÍTICO

        inferência:
            12.345 ms


    Observe uma vantagem:

        PersonDetection continua sendo REAL.

    Estamos simulando somente a inferência, não
    todo o domínio da aplicação.
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

    detections = [detection]

    risk_results = [
        (
            detection,
            "CRÍTICO",
        )
    ]

    inference_ms = 12.345

    return (
        detections,
        risk_results,
        inference_ms,
    )


# ============================================================
# TESTE 1
#
# HEALTH CHECK
# ============================================================


def test_health() -> None:
    """
    Valida o endpoint de saúde da aplicação.

    Esse endpoint não depende da inferência.

    Esperamos:

        HTTP 200

    e:

        {
            "status": "ok",
            "model": "yolo11n.pt"
        }
    """

    response = client.get("/health")

    # --------------------------------------------------------
    # Status HTTP
    # --------------------------------------------------------

    assert response.status_code == 200

    # --------------------------------------------------------
    # Content-Type
    # --------------------------------------------------------

    assert "application/json" in response.headers["content-type"]

    # --------------------------------------------------------
    # Corpo JSON
    # --------------------------------------------------------

    body = response.json()

    assert body["status"] == "ok"

    assert body["model"] == "yolo11n.pt"


# ============================================================
# TESTE 2
#
# ENDPOINT JSON
# ============================================================


def test_infer_returns_expected_json(
    monkeypatch,
) -> None:
    """
    Valida:

        POST /api/v1/infer


    Aqui substituímos:

        process_image()

    pela nossa implementação determinística:

        fake_process_image()


    Portanto este teste NÃO mede se YOLO detecta
    uma pessoa.

    Ele verifica se a API transforma corretamente
    um resultado interno em um contrato HTTP JSON.
    """

    # --------------------------------------------------------
    # Mock da inferência
    # --------------------------------------------------------

    monkeypatch.setattr(
        api_module,
        "process_image",
        fake_process_image,
    )

    # --------------------------------------------------------
    # Imagem enviada ao endpoint
    # --------------------------------------------------------

    image_bytes = create_test_jpeg()

    # --------------------------------------------------------
    # Requisição multipart/form-data
    # --------------------------------------------------------
    #
    # O TestClient monta multipart automaticamente.
    #
    # A estrutura:
    #
    #     (
    #         filename,
    #         conteúdo,
    #         MIME type
    #     )
    #
    # corresponde ao mesmo conceito usado pelo curl:
    #
    #     -F "file=@imagem.jpg"
    #
    # --------------------------------------------------------

    response = client.post(
        "/api/v1/infer",
        files={
            "file": (
                "test.jpg",
                image_bytes,
                "image/jpeg",
            )
        },
    )

    # ========================================================
    # STATUS HTTP
    # ========================================================

    assert response.status_code == 200

    # ========================================================
    # CONTENT-TYPE
    # ========================================================

    assert "application/json" in response.headers["content-type"]

    # ========================================================
    # RESPOSTA JSON
    # ========================================================

    body = response.json()

    # --------------------------------------------------------
    # Arquivo
    # --------------------------------------------------------

    assert body["filename"] == "test.jpg"

    # --------------------------------------------------------
    # Dimensões
    # --------------------------------------------------------

    assert body["image"] == {
        "width": TEST_WIDTH,
        "height": TEST_HEIGHT,
    }

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    assert body["model"] == "yolo11n.pt"

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    assert body["confidence_threshold"] == 0.4

    # --------------------------------------------------------
    # Tempo simulado
    # --------------------------------------------------------

    assert body["inference_ms"] == 12.345

    # --------------------------------------------------------
    # Quantidade de pessoas
    # --------------------------------------------------------

    assert body["persons_detected"] == 1

    # ========================================================
    # DETECÇÃO
    # ========================================================

    assert len(body["detections"]) == 1

    detection = body["detections"][0]

    # --------------------------------------------------------
    # Classe
    # --------------------------------------------------------

    assert detection["class_name"] == "person"

    # --------------------------------------------------------
    # Confiança
    # --------------------------------------------------------

    assert detection["confidence"] == 0.932

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    assert detection["bbox"] == {
        "x1": 382,
        "y1": 159,
        "x2": 505,
        "y2": 555,
    }

    # --------------------------------------------------------
    # Foot point
    # --------------------------------------------------------
    #
    # É calculado por:
    #
    #     x = (382 + 505) // 2
    #
    #     x = 443
    #
    #     y = 555
    #
    # --------------------------------------------------------

    assert detection["foot_point"] == {
        "x": 443,
        "y": 555,
    }

    # --------------------------------------------------------
    # Risco
    # --------------------------------------------------------
    #
    # Observe que no JSON mantemos Unicode:
    #
    #     CRÍTICO
    #
    # A conversão para:
    #
    #     CRITICO
    #
    # acontece somente no desenho OpenCV.
    #
    # --------------------------------------------------------

    assert detection["risk"] == "CRÍTICO"


# ============================================================
# TESTE 3
#
# ENDPOINT PNG
# ============================================================


def test_infer_annotated_returns_valid_png(
    monkeypatch,
) -> None:
    """
    Valida:

        POST /api/v1/infer/annotated


    Queremos provar três coisas:

        1. HTTP 200

        2. Content-Type image/png

        3. os bytes retornados representam
           realmente uma imagem PNG válida


    Novamente, YOLO é simulado.
    """

    # --------------------------------------------------------
    # Mock da inferência
    # --------------------------------------------------------

    monkeypatch.setattr(
        api_module,
        "process_image",
        fake_process_image,
    )

    # --------------------------------------------------------
    # Imagem sintética
    # --------------------------------------------------------

    image_bytes = create_test_jpeg()

    # --------------------------------------------------------
    # Requisição HTTP
    # --------------------------------------------------------

    response = client.post(
        "/api/v1/infer/annotated",
        files={
            "file": (
                "test.jpg",
                image_bytes,
                "image/jpeg",
            )
        },
    )

    # ========================================================
    # STATUS
    # ========================================================

    assert response.status_code == 200

    # ========================================================
    # MIME TYPE
    # ========================================================

    assert response.headers["content-type"] == "image/png"

    # ========================================================
    # ASSINATURA PNG
    # ========================================================
    #
    # Todo PNG começa com estes oito bytes:
    #
    #     89 50 4E 47 0D 0A 1A 0A
    #
    # Em Python:
    #
    #     b"\x89PNG\r\n\x1a\n"
    #
    # Isso já nos dá uma primeira evidência de que
    # recebemos um PNG.
    #
    # ========================================================

    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")

    # ========================================================
    # DECODIFICAÇÃO REAL DO PNG
    # ========================================================
    #
    # Além da assinatura, tentamos abrir os bytes
    # com OpenCV.
    #
    # Se cv2.imdecode retornar None, o conteúdo não
    # representa uma imagem válida.
    #
    # ========================================================

    png_buffer = np.frombuffer(
        response.content,
        dtype=np.uint8,
    )

    decoded_image = cv2.imdecode(
        png_buffer,
        cv2.IMREAD_COLOR,
    )

    assert decoded_image is not None

    # --------------------------------------------------------
    # Dimensões
    # --------------------------------------------------------

    height, width = decoded_image.shape[:2]

    assert width == TEST_WIDTH

    assert height == TEST_HEIGHT

    # ========================================================
    # CONFIRMA QUE HOUVE ANOTAÇÃO
    # ========================================================
    #
    # A imagem original era completamente preta.
    #
    # O visualizador desenha:
    #
    #     zona amarela
    #     zona vermelha
    #     bounding box
    #     foot point
    #     textos
    #
    # Portanto esperamos pixels diferentes de zero.
    #
    # Não fazemos comparação pixel-a-pixel porque
    # detalhes de rasterização podem variar entre
    # versões do OpenCV.
    #
    # ========================================================

    assert int(decoded_image.sum()) > 0


# ============================================================
# TESTE 4
#
# ARQUIVO INVÁLIDO
# ============================================================


def test_infer_rejects_invalid_image() -> None:
    """
    Valida tratamento de erro.

    Enviamos bytes que NÃO representam uma imagem.

    Esperamos:

        HTTP 400

    Isso testa a proteção existente em:

        decode_uploaded_image()
    """

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

    # --------------------------------------------------------
    # HTTP 400 = Bad Request
    # --------------------------------------------------------

    assert response.status_code == 400

    # --------------------------------------------------------
    # Mensagem de erro
    # --------------------------------------------------------

    body = response.json()

    assert body["detail"] == "Não foi possível decodificar o arquivo como imagem."
