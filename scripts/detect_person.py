import argparse
from pathlib import Path

import cv2
from app.detector import PersonDetector
from app.zones import RiskZoneClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "zones.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta pessoas e classifica risco por zona."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Caminho da imagem de entrada",
    )

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

    risk_results = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        risk = zone_classifier.classify(
            point=detection.foot_point,
            width=width,
            height=height,
        )

        risk_results.append((detection, risk))

        print(
            f"{index}: "
            f"confiança={detection.confidence:.3f}, "
            f"bbox={detection.bbox}, "
            f"foot_point={detection.foot_point}, "
            f"risco={risk}"
        )

    # Parte diretamente da imagem original.
    # Assim evitamos o rótulo antigo "person 0.xx".
    annotated = image.copy()

    # -------------------------------------------------
    # Desenha a zona vermelha
    # -------------------------------------------------

    red_polygon = zone_classifier.get_polygon(
        "red",
        width,
        height,
    )

    cv2.polylines(
        annotated,
        [red_polygon],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3,
    )

    zone_label = "ZONA VERMELHA"

    zone_x, zone_y, _, _ = cv2.boundingRect(red_polygon)

    zone_font = cv2.FONT_HERSHEY_SIMPLEX
    zone_font_scale = 0.65
    zone_thickness = 2

    (
        (
            zone_text_width,
            zone_text_height,
        ),
        zone_baseline,
    ) = cv2.getTextSize(
        zone_label,
        zone_font,
        zone_font_scale,
        zone_thickness,
    )

    zone_label_x = zone_x

    zone_label_y = max(
        zone_text_height + 10,
        zone_y - 10,
    )

    cv2.rectangle(
        annotated,
        (
            zone_label_x - 5,
            zone_label_y - zone_text_height - 5,
        ),
        (
            zone_label_x + zone_text_width + 5,
            zone_label_y + zone_baseline + 5,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        annotated,
        zone_label,
        (
            zone_label_x,
            zone_label_y,
        ),
        zone_font,
        zone_font_scale,
        (0, 0, 255),
        zone_thickness,
        cv2.LINE_AA,
    )

    # -------------------------------------------------
    # Desenha cada pessoa com um único rótulo
    # -------------------------------------------------

    for detection, risk in risk_results:
        x1, y1, x2, y2 = detection.bbox

        foot_x, foot_y = detection.foot_point

        if risk == "CRÍTICO":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        # Mostra o ponto usado para decidir
        # se a pessoa está dentro da zona.
        cv2.circle(
            annotated,
            (foot_x, foot_y),
            6,
            color,
            -1,
        )

        risk_label = f"{risk} | {detection.confidence:.2f}"

        risk_font = cv2.FONT_HERSHEY_SIMPLEX

        risk_font_scale = 0.7
        risk_thickness = 2

        (
            (
                risk_text_width,
                risk_text_height,
            ),
            risk_baseline,
        ) = cv2.getTextSize(
            risk_label,
            risk_font,
            risk_font_scale,
            risk_thickness,
        )

        risk_label_x = x1

        # Mantemos o texto dentro da caixa.
        # Isso evita cortar o rótulo de pessoas
        # próximas ao topo da imagem.
        risk_label_y = y1 + risk_text_height + 10

        cv2.rectangle(
            annotated,
            (
                risk_label_x,
                risk_label_y - risk_text_height - 5,
            ),
            (
                risk_label_x + risk_text_width + 10,
                risk_label_y + risk_baseline + 5,
            ),
            color,
            -1,
        )

        cv2.putText(
            annotated,
            risk_label,
            (
                risk_label_x + 5,
                risk_label_y,
            ),
            risk_font,
            risk_font_scale,
            (255, 255, 255),
            risk_thickness,
            cv2.LINE_AA,
        )

    # -------------------------------------------------
    # Salva resultado final
    # -------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(args.output),
        annotated,
    ):
        raise RuntimeError(f"Falha ao salvar imagem anotada: {args.output}")

    print(f"Imagem anotada salva em: {args.output}")


if __name__ == "__main__":
    main()
