from pathlib import Path

import cv2
from app.zones import RiskZoneClassifier

# -------------------------------------------------
# Caminhos base do projeto
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "samples" / "input.jpg"

CONFIG_PATH = ROOT / "config" / "zones.json"


# -------------------------------------------------
# Carregamento da imagem
# -------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Imagem não encontrada: {IMAGE_PATH}")

height, width = image.shape[:2]


# -------------------------------------------------
# Inicialização do classificador
# -------------------------------------------------

classifier = RiskZoneClassifier(CONFIG_PATH)


# -------------------------------------------------
# Casos de teste
# -------------------------------------------------

# Cada item contém:
#
#     (ponto, resultado esperado)
#
# Os três casos representam os três estados
# possíveis do sistema:
#
#     SEGURO
#         fora das zonas
#
#     ALERTA
#         dentro da zona amarela,
#         mas fora da vermelha
#
#     CRÍTICO
#         dentro da zona vermelha

test_points = [
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


# -------------------------------------------------
# Execução dos testes
# -------------------------------------------------

print(f"Imagem: {width}x{height}")

for point, expected in test_points:
    result = classifier.classify(
        point=point,
        width=width,
        height=height,
    )

    print(f"point={point} resultado={result} esperado={expected}")

    # O assert faz o teste falhar imediatamente
    # caso o resultado seja diferente do esperado.
    #
    # Isso é importante porque não queremos
    # apenas "olhar" a saída e decidir manualmente
    # se parece correta.
    assert result == expected, (
        f"Falhou para {point}: esperado={expected}, obtido={result}"
    )


# -------------------------------------------------
# Resultado final
# -------------------------------------------------

print("Teste de zonas aprovado.")
