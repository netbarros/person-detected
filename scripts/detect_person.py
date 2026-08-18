import argparse
from pathlib import Path

import cv2
from app.detector import PersonDetector
from app.zones import RiskZoneClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "zones.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta pessoas em uma imagem usando YOLO."
    )
    parser.add_argument("image", type=Path, help="Caminho da imagem de entrada")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/person-detected.png"),
        help="Caminho da imagem anotada",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")

    image = cv2.imread(str(args.image))

    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {args.image}")

    height, width = image.shape[:2]

    detector = PersonDetector()
    detections = detector.detect(args.image)

    zone_classifier = RiskZoneClassifier(CONFIG_PATH)

    print(f"Pessoas detectadas: {len(detections)}")

    for index, detection in enumerate(detections, start=1):
        risk = zone_classifier.classify(
            point=detection.foot_point,
            width=width,
            height=height,
        )

        print(
            f"{index}: "
            f"confiança={detection.confidence:.3f}, "
            f"bbox={detection.bbox}, "
            f"foot_point={detection.foot_point}, "
            f"risco={risk}"
        )

    detector.annotate(args.image, detections, args.output)
    print(f"Imagem anotada salva em: {args.output}")


if __name__ == "__main__":
    main()
