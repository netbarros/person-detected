from pathlib import Path

import cv2
from app.zones import RiskZoneClassifier

# ============================================================
# CAMINHOS DO PROJETO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config" / "zones.json"

IMAGE_PATH = ROOT / "samples" / "input.jpg"


# ============================================================
# TESTE DAS TRÊS CLASSIFICAÇÕES
# ============================================================


def test_risk_zone_classification() -> None:
    """
    Valida os três estados possíveis do motor de zonas:

        SEGURO
        ALERTA
        CRÍTICO

    O teste usa pontos conhecidos da imagem utilizada
    na demonstração.

    Isso valida a parte determinística da solução,
    independentemente do YOLO.

    Em outras palavras:

        não estamos testando se a IA detecta alguém;

        estamos testando se, dado um ponto conhecido,
        a regra espacial toma a decisão correta.
    """

    # --------------------------------------------------------
    # Carrega a imagem apenas para obter sua resolução
    # --------------------------------------------------------

    image = cv2.imread(str(IMAGE_PATH))

    assert image is not None, f"Não foi possível abrir a imagem: {IMAGE_PATH}"

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Inicializa o classificador
    # --------------------------------------------------------

    classifier = RiskZoneClassifier(CONFIG_PATH)

    # --------------------------------------------------------
    # Casos conhecidos
    # --------------------------------------------------------
    #
    # Estes pontos foram definidos e validados durante
    # a calibração das zonas.
    #
    # Pessoa da direita:
    #
    #     foot_point=(643, 583)
    #
    # fica fora das zonas.
    #
    # O ponto intermediário:
    #
    #     (443, 470)
    #
    # pertence somente à zona amarela.
    #
    # Pessoa da esquerda:
    #
    #     foot_point=(443, 555)
    #
    # pertence à zona vermelha.
    #
    # --------------------------------------------------------

    cases = [
        (
            (643, 583),
            "SEGURO",
        ),
        (
            (443, 470),
            "ALERTA",
        ),
        (
            (443, 555),
            "CRÍTICO",
        ),
    ]

    # --------------------------------------------------------
    # Execução
    # --------------------------------------------------------

    for (
        point,
        expected_risk,
    ) in cases:
        actual_risk = classifier.classify(
            point=point,
            width=width,
            height=height,
        )

        assert actual_risk == expected_risk, (
            f"Ponto {point}: esperado={expected_risk}, obtido={actual_risk}"
        )


# ============================================================
# TESTE DA PRIORIDADE DA ZONA VERMELHA
# ============================================================


def test_red_zone_has_priority() -> None:
    """
    Valida uma regra de negócio importante:

        vermelho > amarelo

    Como a zona vermelha está contida na amarela,
    um ponto crítico geometricamente também pode estar
    dentro do polígono amarelo.

    A classificação precisa devolver CRÍTICO.

    Isso não depende da ordem visual dos desenhos.
    É uma regra explícita do motor de decisão.
    """

    image = cv2.imread(str(IMAGE_PATH))

    assert image is not None

    height, width = image.shape[:2]

    classifier = RiskZoneClassifier(CONFIG_PATH)

    # Ponto conhecido dentro da região crítica.
    point = (
        443,
        555,
    )

    risk = classifier.classify(
        point=point,
        width=width,
        height=height,
    )

    assert risk == "CRÍTICO"
