import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# -------------------------------------------------
# Caminhos base do projeto
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "samples" / "input.jpg"

CONFIG_PATH = ROOT / "config" / "zones.json"


# -------------------------------------------------
# Configuração visual das zonas
# -------------------------------------------------

# OpenCV trabalha com BGR, não RGB.
#
# Vermelho:
# B = 0
# G = 0
# R = 255
#
# Amarelo:
# B = 0
# G = 255
# R = 255

ZONE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}


ZONE_LABELS = {
    "red": "VERMELHA",
    "yellow": "AMARELA",
}


# -------------------------------------------------
# Leitura da configuração
# -------------------------------------------------


def load_config() -> dict:
    """
    Carrega o arquivo config/zones.json.

    Caso o arquivo ainda não exista, retornamos
    uma estrutura mínima:

        {
            "zones": {}
        }

    Isso permite que o calibrador também funcione
    em um projeto iniciado do zero.
    """

    if not CONFIG_PATH.exists():
        return {"zones": {}}

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # Validação simples da estrutura esperada.
    #
    # O arquivo precisa possuir:
    #
    # {
    #     "zones": {
    #         ...
    #     }
    # }

    if "zones" not in data or not isinstance(
        data["zones"],
        dict,
    ):
        raise ValueError(
            f"Configuração inválida em "
            f"{CONFIG_PATH}: "
            "a chave 'zones' deve existir "
            "e ser um objeto."
        )

    return data


# -------------------------------------------------
# Conversão de coordenadas
# -------------------------------------------------


def normalized_to_pixels(
    normalized_points: list[list[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Converte coordenadas normalizadas para pixels.

    Exemplo:

        imagem = 1000 x 500

        ponto normalizado:
            [0.5, 0.5]

        ponto em pixels:
            (500, 250)

    Essa conversão é necessária para desenhar
    o polígono na imagem.
    """

    points = [
        (
            int(round(x * width)),
            int(round(y * height)),
        )
        for x, y in normalized_points
    ]

    return np.array(
        points,
        dtype=np.int32,
    )


def pixels_to_normalized(
    points: list[tuple[int, int]],
    width: int,
    height: int,
) -> list[list[float]]:
    """
    Converte coordenadas em pixels para
    coordenadas normalizadas entre 0 e 1.

    Exemplo:

        imagem = 1000 x 500

        ponto:
            (500, 250)

        resultado:
            [0.5, 0.5]

    A vantagem de salvar coordenadas normalizadas
    é não acoplar a zona à resolução específica
    da câmera.

    Se amanhã a imagem mudar de 880x587 para
    1920x1080, a mesma configuração ainda poderá
    ser convertida proporcionalmente.
    """

    return [
        [
            round(
                x / width,
                6,
            ),
            round(
                y / height,
                6,
            ),
        ]
        for x, y in points
    ]


# -------------------------------------------------
# Desenho das zonas já existentes
# -------------------------------------------------


def draw_existing_zones(
    canvas: np.ndarray,
    config: dict,
    width: int,
    height: int,
    selected_zone: str,
) -> None:
    """
    Desenha na tela as zonas que já estão
    armazenadas no zones.json.

    Isso é importante porque, ao marcar a zona
    amarela, queremos continuar enxergando a
    zona vermelha como referência.

    Exemplo:

        red    -> já existente
        yellow -> estamos marcando agora

    A zona vermelha continua aparecendo na tela.
    """

    for (
        zone_name,
        normalized_points,
    ) in config["zones"].items():
        # Ignora qualquer zona desconhecida.
        if zone_name not in ZONE_COLORS:
            continue

        # Um polígono precisa ter pelo menos
        # três pontos.
        if len(normalized_points) < 3:
            continue

        polygon = normalized_to_pixels(
            normalized_points,
            width,
            height,
        )

        # Se estivermos editando uma zona que já
        # existe, desenhamos a versão antiga com
        # linha um pouco mais fina.
        thickness = 1 if zone_name == selected_zone else 2

        cv2.polylines(
            canvas,
            [polygon],
            isClosed=True,
            color=ZONE_COLORS[zone_name],
            thickness=thickness,
        )


# -------------------------------------------------
# Programa principal
# -------------------------------------------------


def main() -> None:
    """
    Ferramenta interativa para calibrar
    zonas de risco.

    Uso:

        python -m scripts.mark_zones red

    ou:

        python -m scripts.mark_zones yellow


    Controles:

        clique esquerdo
            adiciona um vértice

        U
            desfaz o último ponto

        C
            limpa os pontos atuais

        S
            salva a zona

        ESC
            fecha a ferramenta
    """

    # ---------------------------------------------
    # Argumento da linha de comando
    # ---------------------------------------------

    parser = argparse.ArgumentParser(
        description=("Marca e salva zonas de risco na imagem de referência.")
    )

    parser.add_argument(
        "zone",
        choices=sorted(ZONE_COLORS.keys()),
        help=("Zona que será calibrada: red ou yellow"),
    )

    args = parser.parse_args()

    selected_zone = args.zone

    # ---------------------------------------------
    # Carregamento da imagem
    # ---------------------------------------------

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {IMAGE_PATH}")

    height, width = image.shape[:2]

    # ---------------------------------------------
    # Carregamento das zonas já existentes
    # ---------------------------------------------

    config = load_config()

    # ---------------------------------------------
    # Pontos da zona que estamos criando
    # ---------------------------------------------

    points: list[tuple[int, int]] = []

    # ---------------------------------------------
    # Nome da janela
    # ---------------------------------------------

    window_name = f"Marcador de zonas - {ZONE_LABELS[selected_zone]}"

    # ---------------------------------------------
    # Redesenho da interface
    # ---------------------------------------------

    def redraw() -> None:
        """
        Reconstrói toda a imagem da janela.

        A ordem é:

        1. imagem original
        2. zonas já salvas
        3. pontos da zona atual
        4. linhas da zona atual
        """

        display = image.copy()

        # Desenha as zonas persistidas.
        draw_existing_zones(
            canvas=display,
            config=config,
            width=width,
            height=height,
            selected_zone=selected_zone,
        )

        # Cor da zona que estamos marcando.
        current_color = ZONE_COLORS[selected_zone]

        # Desenha cada ponto clicado.
        for point in points:
            cv2.circle(
                display,
                point,
                5,
                current_color,
                -1,
            )

        # Com dois pontos já conseguimos
        # desenhar uma linha.
        if len(points) >= 2:
            polygon = np.array(
                points,
                dtype=np.int32,
            )

            # Com três ou mais pontos,
            # fechamos o polígono.
            cv2.polylines(
                display,
                [polygon],
                isClosed=(len(points) >= 3),
                color=current_color,
                thickness=3,
            )

        cv2.imshow(
            window_name,
            display,
        )

    # ---------------------------------------------
    # Salvamento da zona
    # ---------------------------------------------

    def save_zone() -> None:
        """
        Persiste somente a zona que está
        sendo editada.

        Esse ponto é importante.

        Antes tínhamos algo assim:

            data = {
                "zones": {
                    "red": ...
                }
            }

        Isso recriava todo o JSON e poderia
        apagar outra zona.

        Agora fazemos:

            config["zones"][selected_zone] = ...

        Portanto:

            salvar yellow
            NÃO apaga red

        e:

            salvar red
            NÃO apaga yellow
        """

        if len(points) < 3:
            print("A zona precisa ter pelo menos 3 pontos.")

            return

        normalized_points = pixels_to_normalized(
            points,
            width,
            height,
        )

        # Altera somente a zona selecionada.
        config["zones"][selected_zone] = normalized_points

        CONFIG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            CONFIG_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                config,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Zona {ZONE_LABELS[selected_zone]} salva em: {CONFIG_PATH}")

        print(f"Pontos normalizados: {normalized_points}")

    # ---------------------------------------------
    # Evento do mouse
    # ---------------------------------------------

    def on_mouse(
        event,
        x,
        y,
        flags,
        param,
    ) -> None:
        """
        Executada pelo OpenCV sempre que
        ocorre um evento do mouse.

        Neste projeto nos interessa somente:

            EVENT_LBUTTONDOWN

        ou seja:

            clique com botão esquerdo.
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            print(f"Ponto {len(points)}: ({x}, {y})")

            redraw()

    # ---------------------------------------------
    # Criação da janela OpenCV
    # ---------------------------------------------

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        on_mouse,
    )

    # ---------------------------------------------
    # Instruções no terminal
    # ---------------------------------------------

    print(f"Calibrando ZONA {ZONE_LABELS[selected_zone]}.")

    print("Clique nos vértices do polígono.")

    print("U = desfazer | C = limpar | S = salvar | ESC = sair")

    # Primeira renderização.
    redraw()

    # ---------------------------------------------
    # Loop da interface
    # ---------------------------------------------

    while True:
        key = cv2.waitKey(20) & 0xFF

        # -----------------------------------------
        # U -> desfazer último ponto
        # -----------------------------------------

        if key in (
            ord("u"),
            ord("U"),
        ):
            if points:
                removed = points.pop()

                print(f"Ponto removido: {removed}")

                redraw()

        # -----------------------------------------
        # C -> limpar pontos atuais
        # -----------------------------------------

        elif key in (
            ord("c"),
            ord("C"),
        ):
            points.clear()

            print("Pontos atuais removidos.")

            redraw()

        # -----------------------------------------
        # S -> salvar
        # -----------------------------------------

        elif key in (
            ord("s"),
            ord("S"),
        ):
            save_zone()

        # -----------------------------------------
        # ESC -> sair
        # -----------------------------------------

        elif key == 27:
            break

    # ---------------------------------------------
    # Liberação das janelas
    # ---------------------------------------------

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
