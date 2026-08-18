import argparse
from pathlib import Path

from app.detector import PersonDetector


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

    detector = PersonDetector()
    detections = detector.detect(args.image)

    print(f"Pessoas detectadas: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        print(
            f"{index}: confiança={detection.confidence:.3f}, "
            f"bbox={detection.bbox}, foot_point={detection.foot_point}"
        )

    detector.annotate(args.image, detections, args.output)
    print(f"Imagem anotada salva em: {args.output}")


if __name__ == "__main__":
    main()
