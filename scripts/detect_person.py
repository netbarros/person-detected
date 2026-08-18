import argparse
from pathlib import Path

import cv2
from app.detector import PersonDetector
from app.zones import RiskZoneClassifier

# -------------------------------------------------
# Caminhos base do projeto
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config" / "zones.json"


# -------------------------------------------------
# Cores das zonas
# -------------------------------------------------

# OpenCV trabalha com BGR.
#
# Vermelho:
#     (0, 0, 255)
#
# Amarelo:
#     (0, 255, 255)

ZONE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}


# -------------------------------------------------
# Normalização dos nomes de risco
# -------------------------------------------------


def normalize_risk_label(
    risk: str,
) -> str:
    """
    Garante que os estados de risco sejam exibidos
    sempre em português.

    Isso permite que o restante do código seja
    tolerante a versões anteriores do classificador.

    Exemplos:

        SAFE      -> SEGURO
        WARNING   -> ALERTA
        CRITICAL  -> CRÍTICO
    """

    mapping = {
        "SAFE": "SEGURO",
        "WARNING": "ALERTA",
        "CRITICAL": "CRÍTICO",
        "SEGURO": "SEGURO",
        "ALERTA": "ALERTA",
        "CRÍTICO": "CRÍTICO",
    }

    return mapping.get(
        risk,
        risk,
    )


# -------------------------------------------------
# Cor correspondente ao nível de risco
# -------------------------------------------------


def get_risk_color(
    risk_label: str,
) -> tuple[int, int, int]:
    """
    Retorna a cor correspondente ao nível de risco.

        SEGURO
            verde

        ALERTA
            amarelo

        CRÍTICO
            vermelho
    """

    if risk_label == "CRÍTICO":
        return (0, 0, 255)

    if risk_label == "ALERTA":
        return (0, 255, 255)

    return (0, 255, 0)


# -------------------------------------------------
# Cor do texto sobre a tarja
# -------------------------------------------------


def get_risk_text_color(
    risk_label: str,
) -> tuple[int, int, int]:
    """
    Define a cor do texto dentro do rótulo.

    Para a tarja amarela usamos texto preto,
    pois branco sobre amarelo possui pouco contraste.

    Para verde e vermelho mantemos texto branco.
    """

    if risk_label == "ALERTA":
        return (0, 0, 0)

    return (255, 255, 255)


# -------------------------------------------------
# Desenho de uma zona
# -------------------------------------------------


def draw_zone_polygon(
    annotated: "cv2.Mat",
    polygon,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """
    Desenha o contorno de uma zona de risco.

    Essa função é genérica.

    Ela pode receber:

        polígono vermelho
        polígono amarelo
        ou qualquer outro polígono futuro
    """

    cv2.polylines(
        annotated,
        [polygon],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


# -------------------------------------------------
# Desenho de uma pessoa
# -------------------------------------------------


def draw_person_risk(
    annotated: "cv2.Mat",
    detection,
    risk_label: str,
) -> None:
    """
    Desenha o resultado de uma detecção.

    Para cada pessoa mostramos:

        bounding box
        ponto dos pés
        nível de risco
        confiança do YOLO

    O ponto dos pés é especialmente importante
    porque ele representa a posição da pessoa
    sobre o piso da área industrial.
    """

    x1, y1, x2, y2 = detection.bbox

    foot_x, foot_y = detection.foot_point

    # ---------------------------------------------
    # Cor correspondente ao risco
    # ---------------------------------------------

    color = get_risk_color(risk_label)

    text_color = get_risk_text_color(risk_label)

    # ---------------------------------------------
    # Bounding box
    # ---------------------------------------------

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    # ---------------------------------------------
    # Foot point
    # ---------------------------------------------

    # Esse é o ponto realmente usado pelo
    # motor geométrico para classificar a zona.

    cv2.circle(
        annotated,
        (foot_x, foot_y),
        6,
        color,
        -1,
    )

    # ---------------------------------------------
    # Rótulo
    # ---------------------------------------------

    label = f"{risk_label} | {detection.confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (
        (
            text_width,
            text_height,
        ),
        baseline,
    ) = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    # Mantemos o texto dentro da bounding box
    # para reduzir risco de corte na borda da imagem.

    label_x = x1

    label_y = y1 + text_height + 10

    # ---------------------------------------------
    # Fundo do rótulo
    # ---------------------------------------------

    cv2.rectangle(
        annotated,
        (
            label_x,
            label_y - text_height - 5,
        ),
        (
            label_x + text_width + 10,
            label_y + baseline + 5,
        ),
        color,
        -1,
    )

    # ---------------------------------------------
    # Texto
    # ---------------------------------------------

    cv2.putText(
        annotated,
        label,
        (
            label_x + 5,
            label_y,
        ),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


# -------------------------------------------------
# Rótulo visual de uma zona
# -------------------------------------------------


def draw_zone_label_front(
    annotated: "cv2.Mat",
    polygon,
    zone_label: str,
    color: tuple[int, int, int],
) -> None:
    """
    Desenha o nome de uma zona.

    Essa função deve ser chamada depois das
    bounding boxes das pessoas.

    Assim, o rótulo da zona fica visualmente
    por cima das demais anotações.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    # ---------------------------------------------
    # Dimensão do texto
    # ---------------------------------------------

    (
        (
            text_width,
            text_height,
        ),
        baseline,
    ) = cv2.getTextSize(
        zone_label,
        font,
        font_scale,
        thickness,
    )

    # ---------------------------------------------
    # Bounding box do polígono
    # ---------------------------------------------

    zone_x, zone_y, _, _ = cv2.boundingRect(polygon)

    # ---------------------------------------------
    # Posição do rótulo
    # ---------------------------------------------

    label_x = zone_x

    label_y = max(
        text_height + 10,
        zone_y - 10,
    )

    padding = 6

    # ---------------------------------------------
    # Fundo preto
    # ---------------------------------------------

    # Usamos fundo preto para garantir contraste
    # independentemente do conteúdo da imagem.

    cv2.rectangle(
        annotated,
        (
            label_x - padding,
            label_y - text_height - padding,
        ),
        (
            label_x + text_width + padding,
            label_y + baseline + padding,
        ),
        (0, 0, 0),
        -1,
    )

    # ---------------------------------------------
    # Texto da zona
    # ---------------------------------------------

    cv2.putText(
        annotated,
        zone_label,
        (
            label_x,
            label_y,
        ),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# -------------------------------------------------
# Programa principal
# -------------------------------------------------


def main() -> None:
    """
    Fluxo completo:

        imagem
            ↓
        YOLO
            ↓
        pessoas detectadas
            ↓
        bounding box
            ↓
        foot_point
            ↓
        motor de zonas
            ↓
        SEGURO / ALERTA / CRÍTICO
            ↓
        imagem anotada
    """

    # ---------------------------------------------
    # 1. Argumentos da linha de comando
    # ---------------------------------------------

    parser = argparse.ArgumentParser(
        description=("Detecta pessoas e classifica risco por zonas.")
    )

    parser.add_argument(
        "image",
        type=Path,
        help=("Caminho da imagem de entrada"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/person-detected.png"),
        help=("Caminho da imagem anotada"),
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # 2. Validação da imagem
    # ---------------------------------------------

    if not args.image.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")

    # ---------------------------------------------
    # 3. Carregamento da imagem
    # ---------------------------------------------

    image = cv2.imread(str(args.image))

    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {args.image}")

    height, width = image.shape[:2]

    # ---------------------------------------------
    # 4. Detecção com YOLO
    # ---------------------------------------------

    detector = PersonDetector()

    detections = detector.detect(args.image)

    # ---------------------------------------------
    # 5. Motor de zonas
    # ---------------------------------------------

    zone_classifier = RiskZoneClassifier(CONFIG_PATH)

    print(f"Pessoas detectadas: {len(detections)}")

    # ---------------------------------------------
    # 6. Classificação individual
    # ---------------------------------------------

    risk_results = []

    for (
        index,
        detection,
    ) in enumerate(
        detections,
        start=1,
    ):
        # O foot_point representa onde a
        # pessoa está tocando o piso.

        risk_raw = zone_classifier.classify(
            point=(detection.foot_point),
            width=width,
            height=height,
        )

        risk_label = normalize_risk_label(risk_raw)

        risk_results.append(
            (
                detection,
                risk_label,
            )
        )

        print(
            f"{index}: "
            f"confiança="
            f"{detection.confidence:.3f}, "
            f"bbox="
            f"{detection.bbox}, "
            f"foot_point="
            f"{detection.foot_point}, "
            f"risco="
            f"{risk_label}"
        )

    # ---------------------------------------------
    # 7. Imagem que receberá as anotações
    # ---------------------------------------------

    annotated = image.copy()

    # ---------------------------------------------
    # 8. Recuperação dos polígonos
    # ---------------------------------------------

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

    # ---------------------------------------------
    # 9. Desenho das zonas
    # ---------------------------------------------

    # Primeiro desenhamos a zona de menor
    # severidade.

    draw_zone_polygon(
        annotated=annotated,
        polygon=yellow_polygon,
        color=(ZONE_COLORS["yellow"]),
        thickness=3,
    )

    # Depois desenhamos a zona crítica.
    #
    # Como a zona vermelha está dentro da
    # amarela, desenhá-la por último ajuda
    # a preservar sua prioridade visual.

    draw_zone_polygon(
        annotated=annotated,
        polygon=red_polygon,
        color=(ZONE_COLORS["red"]),
        thickness=3,
    )

    # ---------------------------------------------
    # 10. Desenho das pessoas
    # ---------------------------------------------

    for (
        detection,
        risk_label,
    ) in risk_results:
        draw_person_risk(
            annotated=annotated,
            detection=detection,
            risk_label=risk_label,
        )

    # ---------------------------------------------
    # 11. Rótulos das zonas
    # ---------------------------------------------

    # Os rótulos são desenhados por último
    # para ficarem na frente das demais
    # anotações.

    draw_zone_label_front(
        annotated=annotated,
        polygon=yellow_polygon,
        zone_label=("ZONA AMARELA"),
        color=(ZONE_COLORS["yellow"]),
    )

    draw_zone_label_front(
        annotated=annotated,
        polygon=red_polygon,
        zone_label=("ZONA VERMELHA"),
        color=(ZONE_COLORS["red"]),
    )

    # ---------------------------------------------
    # 12. Salvamento
    # ---------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(args.output),
        annotated,
    ):
        raise RuntimeError(f"Falha ao salvar imagem anotada: {args.output}")

    # ---------------------------------------------
    # 13. Resultado final
    # ---------------------------------------------

    print(f"Imagem anotada salva em: {args.output}")


if __name__ == "__main__":
    main()
