import cv2
import numpy as np

from app.detector import PersonDetection

# =================================================
# CORES
# =================================================
#
# IMPORTANTE:
#
# O OpenCV trabalha com a ordem:
#
#     BGR
#
# e não RGB.
#
# Portanto:
#
#     (0, 0, 255)
#
# significa vermelho.
#
# -------------------------------------------------

COLOR_SAFE = (0, 255, 0)

COLOR_WARNING = (
    0,
    255,
    255,
)

COLOR_CRITICAL = (
    0,
    0,
    255,
)

COLOR_BLACK = (
    0,
    0,
    0,
)

COLOR_WHITE = (
    255,
    255,
    255,
)


# =================================================
# FORMATAÇÃO VISUAL DO RISCO
# =================================================


def get_risk_display_label(
    risk: str,
) -> str:
    """
    Converte o nome interno do risco para uma forma
    compatível com a renderização do OpenCV.

    Internamente continuamos trabalhando com:

        SEGURO
        ALERTA
        CRÍTICO

    Entretanto, cv2.putText() utiliza as fontes
    Hershey do OpenCV.

    Essas fontes não possuem suporte completo a
    caracteres Unicode.

    Por isso:

        CRÍTICO

    pode aparecer na imagem como:

        CR??TICO

    Não queremos alterar o domínio da aplicação nem
    o contrato JSON apenas por uma limitação visual.

    Portanto fazemos a conversão SOMENTE na camada
    de apresentação:

        CRÍTICO -> CRITICO

    O JSON continua retornando:

        "risk": "CRÍTICO"
    """

    mapping = {
        "SEGURO": "SEGURO",
        "ALERTA": "ALERTA",
        "CRÍTICO": "CRITICO",
    }

    return mapping.get(
        risk,
        risk,
    )


# =================================================
# COR DO RISCO
# =================================================


def get_risk_color(
    risk: str,
) -> tuple[int, int, int]:
    """
    Define a cor correspondente ao nível de risco.

    Convenção visual:

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


# =================================================
# COR DO TEXTO
# =================================================


def get_risk_text_color(
    risk: str,
) -> tuple[int, int, int]:
    """
    Define a cor do texto colocado sobre o fundo
    correspondente ao risco.

    ALERTA possui fundo amarelo.

    Texto branco sobre amarelo possui contraste
    relativamente baixo.

    Por isso usamos:

        ALERTA
            texto preto

        SEGURO
            texto branco

        CRÍTICO
            texto branco
    """

    if risk == "ALERTA":
        return COLOR_BLACK

    return COLOR_WHITE


# =================================================
# DESENHO DO POLÍGONO
# =================================================


def draw_zone_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """
    Desenha o contorno de uma zona de risco.

    A função modifica a imagem recebida.

    polygon possui coordenadas OpenCV no formato:

        [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            ...
        ]
    """

    cv2.polylines(
        image,
        [polygon],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


# =================================================
# DESENHO DA PESSOA
# =================================================


def draw_person_risk(
    image: np.ndarray,
    detection: PersonDetection,
    risk: str,
) -> None:
    """
    Desenha todas as informações visuais associadas
    a uma pessoa detectada.

    São desenhados:

        1. bounding box
        2. foot_point
        3. classificação de risco
        4. confiança do detector


    Exemplo:

        +------------------------+
        | SEGURO | 0.93          |
        +------------------------+
        |                        |
        |         pessoa         |
        |                        |
        |                        |
        +-----------●------------+
                    ^
                foot_point
    """

    # -------------------------------------------------
    # Bounding box
    # -------------------------------------------------

    x1, y1, x2, y2 = detection.bbox

    # -------------------------------------------------
    # Foot point
    # -------------------------------------------------

    foot_x, foot_y = detection.foot_point

    # -------------------------------------------------
    # Cor do risco
    # -------------------------------------------------

    color = get_risk_color(risk)

    # -------------------------------------------------
    # Cor do texto
    # -------------------------------------------------

    text_color = get_risk_text_color(risk)

    # -------------------------------------------------
    # Bounding box
    # -------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    # -------------------------------------------------
    # Foot point
    # -------------------------------------------------
    #
    # O foot_point representa aproximadamente o ponto
    # de contato da pessoa com o piso.
    #
    # É esse ponto que utilizamos para decidir em qual
    # zona espacial a pessoa se encontra.
    #
    # Não usamos o centro da bounding box porque ele
    # representa aproximadamente o tronco da pessoa,
    # não sua posição sobre o chão.
    #
    # -------------------------------------------------

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

    # =================================================
    # RÓTULO DA PESSOA
    # =================================================

    # -------------------------------------------------
    # Conversão apenas para apresentação
    # -------------------------------------------------
    #
    # Aqui está a correção do problema observado:
    #
    #     CRÍTICO
    #
    # aparecia como:
    #
    #     CR??TICO
    #
    # Não mudamos o valor original de "risk".
    #
    # Apenas criamos uma versão própria para desenho.
    #
    # -------------------------------------------------

    display_risk = get_risk_display_label(risk)

    # -------------------------------------------------
    # Texto final
    # -------------------------------------------------

    label = f"{display_risk} | {detection.confidence:.2f}"

    # -------------------------------------------------
    # Configuração da fonte
    # -------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.7

    thickness = 2

    # -------------------------------------------------
    # Medição do texto
    # -------------------------------------------------
    #
    # Precisamos saber o tamanho do texto antes de
    # desenhar seu fundo.
    #
    # getTextSize retorna:
    #
    #     largura
    #     altura
    #     baseline
    #
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Posição do rótulo
    # -------------------------------------------------

    label_x = x1

    label_y = y1 + text_height + 10

    # -------------------------------------------------
    # Fundo do rótulo
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Texto do rótulo
    # -------------------------------------------------

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


# =================================================
# RÓTULO DA ZONA
# =================================================


def draw_zone_label(
    image: np.ndarray,
    polygon: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """
    Desenha o nome da zona próximo ao polígono.

    Exemplos:

        ZONA AMARELA

        ZONA VERMELHA


    Essa função deve ser chamada DEPOIS do desenho
    das pessoas.

    Assim o rótulo da zona permanece visualmente
    sobre outros elementos quando houver sobreposição.
    """

    # -------------------------------------------------
    # Configuração da fonte
    # -------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.75

    thickness = 2

    # -------------------------------------------------
    # Dimensões do texto
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Bounding rectangle do polígono
    # -------------------------------------------------
    #
    # cv2.boundingRect calcula o menor retângulo capaz
    # de conter todo o polígono.
    #
    # Usamos seu canto superior esquerdo como
    # referência para posicionar o texto.
    #
    # -------------------------------------------------

    (
        zone_x,
        zone_y,
        _,
        _,
    ) = cv2.boundingRect(polygon)

    # -------------------------------------------------
    # Posição vertical segura
    # -------------------------------------------------

    label_x = zone_x

    label_y = max(
        text_height + 10,
        zone_y - 10,
    )

    padding = 6

    # -------------------------------------------------
    # Fundo preto
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Nome da zona
    # -------------------------------------------------

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


# =================================================
# RENDERIZAÇÃO COMPLETA
# =================================================


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
    Produz a imagem final anotada.

    Recebe:

        imagem original

        detecções classificadas

        polígono amarelo

        polígono vermelho


    Retorna:

        uma NOVA imagem anotada


    A imagem original não é modificada.

    Isso é importante porque permite reutilizar
    a mesma imagem em outros componentes do sistema.
    """

    # -------------------------------------------------
    # Cópia defensiva
    # -------------------------------------------------

    annotated = image.copy()

    # =================================================
    # 1. ZONA AMARELA
    # =================================================
    #
    # Desenhamos a zona mais ampla primeiro.
    #
    # -------------------------------------------------

    draw_zone_polygon(
        image=annotated,
        polygon=yellow_polygon,
        color=COLOR_WARNING,
    )

    # =================================================
    # 2. ZONA VERMELHA
    # =================================================
    #
    # A zona vermelha possui prioridade.
    #
    # Como ela é desenhada depois, seu contorno fica
    # visível mesmo quando estiver dentro da amarela.
    #
    # -------------------------------------------------

    draw_zone_polygon(
        image=annotated,
        polygon=red_polygon,
        color=COLOR_CRITICAL,
    )

    # =================================================
    # 3. PESSOAS
    # =================================================

    for (
        detection,
        risk,
    ) in risk_results:
        draw_person_risk(
            image=annotated,
            detection=detection,
            risk=risk,
        )

    # =================================================
    # 4. RÓTULOS DAS ZONAS
    # =================================================
    #
    # São desenhados por último para permanecerem
    # visualmente acima dos bounding boxes.
    #
    # -------------------------------------------------

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

    # =================================================
    # 5. RESULTADO
    # =================================================

    return annotated
