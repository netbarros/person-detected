from pathlib import Path

import cv2
from app.zones import RiskZoneClassifier

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "samples" / "input.jpg"
CONFIG_PATH = ROOT / "config" / "zones.json"


image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Imagem não encontrada: {IMAGE_PATH}")

height, width = image.shape[:2]

classifier = RiskZoneClassifier(CONFIG_PATH)

test_points = [
    ((643, 583), "SEGURO"),
    ((443, 555), "CRÍTICO"),
]

print(f"Imagem: {width}x{height}")

for point, expected in test_points:
    result = classifier.classify(
        point=point,
        width=width,
        height=height,
    )

    print(f"point={point} resultado={result} esperado={expected}")

    assert result == expected, (
        f"Falhou para {point}: esperado={expected}, obtido={result}"
    )

print("Teste de zonas aprovado.")
