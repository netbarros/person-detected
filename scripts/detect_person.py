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
# Funções auxiliares
# -------------------------------------------------


def normalize_risk_label(risk: str) -> str:
    """
    Normaliza o rótulo de risco para exibição em português.

    Isso deixa o script mais robusto, porque o classificador
    pode retornar SAFE / CRITICAL ou SEGURO / CRÍTICO,
    e aqui garantimos um padrão visual único.

    Exemplos:
        SAFE      -> SEGURO
        CRITICAL  -> CRÍTICO
        WARNING   -> ALERTA
    """
    mapping = {
        "SAFE": "SEGURO",
        "CRITICAL": "CRÍTICO",
        "WARNING": "ALERTA",
        "SEGURO": "SEGURO",
        "CRÍTICO": "CRÍTICO",
        "ALERTA": "ALERTA",
    }

    return mapping.get(risk, risk)


def get_risk_color(risk_label: str) -> tuple[int, int, int]:
    """
    Define a cor BGR usada para desenhar cada risco.

    OpenCV usa BGR em vez de RGB:
        Verde   -> (0, 255, 0)
        Amarelo -> (0, 255, 255)
        Vermelho-> (0, 0, 255)
    """
    if risk_label == "CRÍTICO":
        return (0, 0, 255)  # vermelho
    if risk_label == "ALERTA":
        return (0, 255, 255)  # amarelo
    return (0, 255, 0)  # verde


def draw_zone_polygon(
    annotated: "cv2.Mat",
    polygon,
    color=(0, 0, 255),
    thickness=3,
) -> None:
    """
    Desenha o contorno do polígono da zona de risco.

    Nesta etapa, estamos desenhando somente a zona vermelha.
    Depois podemos expandir para zona amarela também.
    """
    cv2.polylines(
        annotated,
        [polygon],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


def draw_person_risk(
    annotated: "cv2.Mat",
    detection,
    risk_label: str,
) -> None:
    """
    Desenha na imagem:
    - bounding box da pessoa
    - ponto dos pés (foot_point)
    - rótulo de risco com confiança

    Esse ponto dos pés é importante porque é ele que usamos
    para decidir se a pessoa está dentro ou fora da zona.
    """
    x1, y1, x2, y2 = detection.bbox
    foot_x, foot_y = detection.foot_point

    color = get_risk_color(risk_label)

    # Caixa da pessoa
    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    # Ponto dos pés: referência espacial usada na regra
    cv2.circle(
        annotated,
        (foot_x, foot_y),
        6,
        color,
        -1,
    )

    # Texto do risco + confiança do detector
    label = f"{risk_label} | {detection.confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    # Mantemos o rótulo próximo ao topo da caixa
    label_x = x1
    label_y = y1 + text_height + 10

    # Fundo colorido do rótulo
    cv2.rectangle(
        annotated,
        (label_x, label_y - text_height - 5),
        (label_x + text_width + 10, label_y + baseline + 5),
        color,
        -1,
    )

    # Texto branco sobre a tarja
    cv2.putText(
        annotated,
        label,
        (label_x + 5, label_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_zone_label_front(
    annotated: "cv2.Mat",
    polygon,
    zone_label: str = "ZONA VERMELHA",
) -> None:
    """
    Desenha o rótulo da zona por último, para garantir
    prioridade visual sobre as demais anotações.

    A estratégia é:
    - calcular o boundingRect do polígono
    - posicionar a tarja logo acima da zona
    - desenhar fundo preto para contraste
    - desenhar o texto em vermelho
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        zone_label,
        font,
        font_scale,
        thickness,
    )

    zone_x, zone_y, _, _ = cv2.boundingRect(polygon)

    # Coloca a tarja acima da zona
    label_x = zone_x
    label_y = max(text_height + 10, zone_y - 10)

    padding = 6

    # Fundo preto para contraste
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

    # Texto vermelho
    cv2.putText(
        annotated,
        zone_label,
        (label_x, label_y),
        font,
        font_scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )


# -------------------------------------------------
# Função principal
# -------------------------------------------------


def main() -> None:
    """
    Fluxo principal da aplicação.

    Etapas:
    1. Ler argumentos da linha de comando
    2. Carregar a imagem
    3. Rodar o detector de pessoas (YOLO)
    4. Classificar o risco de cada pessoa pela zona
    5. Desenhar:
       - polígono da zona
       - pessoas com risco
       - rótulo final da zona
    6. Salvar a imagem anotada
    """

    # ---------------------------------------------
    # 1. Argumentos de entrada
    # ---------------------------------------------
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

    # ---------------------------------------------
    # 2. Carregamento da imagem
    # ---------------------------------------------
    if not args.image.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")

    image = cv2.imread(str(args.image))

    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {args.image}")

    height, width = image.shape[:2]

    # ---------------------------------------------
    # 3. Detecção de pessoas
    # ---------------------------------------------
    detector = PersonDetector()
    detections = detector.detect(args.image)

    # ---------------------------------------------
    # 4. Classificação de risco por zona
    # ---------------------------------------------
    zone_classifier = RiskZoneClassifier(CONFIG_PATH)

    print(f"Pessoas detectadas: {len(detections)}")

    risk_results = []

    for index, detection in enumerate(detections, start=1):
        # O classificador recebe:
        # - o ponto dos pés da pessoa
        # - a largura e altura da imagem
        # para transformar as zonas normalizadas em pixels.
        risk_raw = zone_classifier.classify(
            point=detection.foot_point,
            width=width,
            height=height,
        )

        risk_label = normalize_risk_label(risk_raw)

        risk_results.append((detection, risk_label))

        print(
            f"{index}: "
            f"confiança={detection.confidence:.3f}, "
            f"bbox={detection.bbox}, "
            f"foot_point={detection.foot_point}, "
            f"risco={risk_label}"
        )

    # ---------------------------------------------
    # 5. Desenho da imagem anotada
    # ---------------------------------------------
    annotated = image.copy()

    # 5.1. Polígono da zona vermelha
    red_polygon = zone_classifier.get_polygon(
        "red",
        width,
        height,
    )

    draw_zone_polygon(
        annotated=annotated,
        polygon=red_polygon,
        color=(0, 0, 255),
        thickness=3,
    )

    # 5.2. Pessoas com classificação de risco
    for detection, risk_label in risk_results:
        draw_person_risk(
            annotated=annotated,
            detection=detection,
            risk_label=risk_label,
        )

    # 5.3. Rótulo da zona desenhado por último
    draw_zone_label_front(
        annotated=annotated,
        polygon=red_polygon,
        zone_label="ZONA VERMELHA",
    )

    # ---------------------------------------------
    # 6. Salvamento do resultado
    # ---------------------------------------------
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"Falha ao salvar imagem anotada: {args.output}")

    print(f"Imagem anotada salva em: {args.output}")


if __name__ == "__main__":
    main()
