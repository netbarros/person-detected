import cv2
import numpy as np

from app.detector import PersonDetection

# -------------------------------------------------
# Cores
# -------------------------------------------------

# OpenCV utiliza BGR.

COLOR_SAFE = (0, 255, 0)
COLOR_WARNING = (0, 255, 255)
COLOR_CRITICAL = (0, 0, 255)

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)


# -------------------------------------------------
# Cor correspondente ao risco
# -------------------------------------------------


def get_risk_color(
    risk: str,
) -> tuple[int, int, int]:
    """
    Define a cor utilizada para representar
    cada nível de risco.

        SEGURO
            verde

        ALERTA
            amarelo

        CRÍTICO
            vermelho
    """

    if risk == "CRÍTICO":
        return COLOR_CRITICAL

    if risk == "ALERTA":
        return COLOR_WARNING

    return COLOR_SAFE


# -------------------------------------------------
# Cor do texto
# -------------------------------------------------


def get_risk_text_color(
    risk: str,
) -> tuple[int, int, int]:
    """
    Usa texto preto sobre amarelo para melhorar
    o contraste visual.

    Verde e vermelho utilizam texto branco.
    """

    if risk == "ALERTA":
        return COLOR_BLACK

    return COLOR_WHITE


# -------------------------------------------------
# Desenha polígono
# -------------------------------------------------


def draw_zone_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """
    Desenha o contorno de uma zona de risco.
    """

    cv2.polylines(
        image,
        [polygon],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


# -------------------------------------------------
# Desenha pessoa e risco
# -------------------------------------------------


def draw_person_risk(
    image: np.ndarray,
    detection: PersonDetection,
    risk: str,
) -> None:
    """
    Desenha:

        bounding box
        foot_point
        nível de risco
        confiança da detecção
    """

    x1, y1, x2, y2 = detection.bbox

    foot_x, foot_y = detection.foot_point

    color = get_risk_color(risk)

    text_color = get_risk_text_color(risk)

    # ---------------------------------------------
    # Bounding box
    # ---------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    # ---------------------------------------------
    # Foot point
    # ---------------------------------------------

    cv2.circle(
        image,
        (
            foot_x,
            foot_y,
        ),
        6,
        color,
        -1,
    )

    # ---------------------------------------------
    # Texto
    # ---------------------------------------------

    label = f"{risk} | {detection.confidence:.2f}"

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

    label_x = x1

    label_y = y1 + text_height + 10

    # ---------------------------------------------
    # Fundo do rótulo
    # ---------------------------------------------

    cv2.rectangle(
        image,
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
    # Texto do rótulo
    # ---------------------------------------------

    cv2.putText(
        image,
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
# Rótulo da zona
# -------------------------------------------------


def draw_zone_label(
    image: np.ndarray,
    polygon: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """
    Desenha o nome de uma zona próximo ao polígono.

    Essa função deve ser chamada após as pessoas
    para manter o rótulo visualmente na frente.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
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

    zone_x, zone_y, _, _ = cv2.boundingRect(polygon)

    label_x = zone_x

    label_y = max(
        text_height + 10,
        zone_y - 10,
    )

    padding = 6

    # Fundo preto
    cv2.rectangle(
        image,
        (
            label_x - padding,
            label_y - text_height - padding,
        ),
        (
            label_x + text_width + padding,
            label_y + baseline + padding,
        ),
        COLOR_BLACK,
        -1,
    )

    # Texto da zona
    cv2.putText(
        image,
        label,
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
# Renderização completa
# -------------------------------------------------


def annotate_risk_image(
    image: np.ndarray,
    risk_results: list[
        tuple[
            PersonDetection,
            str,
        ]
    ],
    yellow_polygon: np.ndarray,
    red_polygon: np.ndarray,
) -> np.ndarray:
    """
    Gera a imagem final anotada.

    Recebe:

        imagem original

        detecções já classificadas

        polígono amarelo

        polígono vermelho

    Retorna uma nova imagem.

    A imagem original não é modificada.
    """

    annotated = image.copy()

    # ---------------------------------------------
    # 1. Zona amarela
    # ---------------------------------------------

    draw_zone_polygon(
        image=annotated,
        polygon=yellow_polygon,
        color=COLOR_WARNING,
    )

    # ---------------------------------------------
    # 2. Zona vermelha
    # ---------------------------------------------

    # A zona crítica é desenhada depois da amarela
    # para manter a prioridade visual.

    draw_zone_polygon(
        image=annotated,
        polygon=red_polygon,
        color=COLOR_CRITICAL,
    )

    # ---------------------------------------------
    # 3. Pessoas
    # ---------------------------------------------

    for detection, risk in risk_results:
        draw_person_risk(
            image=annotated,
            detection=detection,
            risk=risk,
        )

    # ---------------------------------------------
    # 4. Rótulos das zonas
    # ---------------------------------------------

    draw_zone_label(
        image=annotated,
        polygon=yellow_polygon,
        label="ZONA AMARELA",
        color=COLOR_WARNING,
    )

    draw_zone_label(
        image=annotated,
        polygon=red_polygon,
        label="ZONA VERMELHA",
        color=COLOR_CRITICAL,
    )

    # ---------------------------------------------
    # 5. Resultado
    # ---------------------------------------------

    return annotated
