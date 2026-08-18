import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "samples" / "input.jpg"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Não foi possível abrir: {IMAGE_PATH}")

display = image.copy()
points = []
CONFIG_PATH = ROOT / "config" / "zones.json"


def redraw():
    global display

    display = image.copy()

    for point in points:
        cv2.circle(
            display,
            point,
            5,
            (0, 0, 255),
            -1,
        )

    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32)

        cv2.polylines(
            display,
            [pts],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
        )

    cv2.imshow("Marcador de zonas", display)


def save_zone():
    if len(points) < 3:
        print("A zona precisa ter pelo menos 3 pontos.")
        return

    height, width = image.shape[:2]

    normalized_points = [[round(x / width, 6), round(y / height, 6)] for x, y in points]

    data = {"zones": {"red": normalized_points}}

    CONFIG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
        )

    print(f"Zona salva em: {CONFIG_PATH}")
    print(normalized_points)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        print(f"Ponto {len(points)}: ({x}, {y})")

        redraw()


cv2.namedWindow("Marcador de zonas")
cv2.setMouseCallback("Marcador de zonas", on_mouse)

print("Clique nos vértices da zona.")
print("Pressione ESC para sair.")

while True:
    cv2.imshow("Marcador de zonas", display)

    key = cv2.waitKey(20)

    if key == ord("s"):
        save_zone()

    if key == 27:
        break

cv2.destroyAllWindows()
