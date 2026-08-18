import json
from pathlib import Path

import cv2
import numpy as np


class RiskZoneClassifier:
    """
    Responsável por classificar um ponto da imagem
    de acordo com as zonas de risco configuradas.

    Neste projeto usamos três níveis:

        SEGURO
            ponto fora das zonas de risco

        ALERTA
            ponto dentro da zona amarela

        CRÍTICO
            ponto dentro da zona vermelha

    A zona vermelha possui prioridade sobre a amarela.
    """

    def __init__(
        self,
        config_path: Path,
    ) -> None:
        """
        Carrega o arquivo JSON contendo os polígonos.

        Estrutura esperada:

        {
            "zones": {
                "red": [...],
                "yellow": [...]
            }
        }
        """

        if not config_path.exists():
            raise FileNotFoundError(f"Arquivo de zonas não encontrado: {config_path}")

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        self._validate_config()

    # -------------------------------------------------
    # Validação da configuração
    # -------------------------------------------------

    def _validate_config(
        self,
    ) -> None:
        """
        Faz uma validação básica do zones.json.

        Para este projeto esperamos obrigatoriamente:

            zones.red
            zones.yellow

        Cada zona precisa ter pelo menos três pontos,
        pois três pontos formam o menor polígono possível.
        """

        zones = self.config.get("zones")

        if not isinstance(
            zones,
            dict,
        ):
            raise ValueError(
                "Configuração inválida: a chave 'zones' não foi encontrada."
            )

        for zone_name in (
            "red",
            "yellow",
        ):
            points = zones.get(zone_name)

            if (
                not isinstance(
                    points,
                    list,
                )
                or len(points) < 3
            ):
                raise ValueError(
                    f"Zona '{zone_name}' inválida: "
                    "é necessário informar pelo menos 3 pontos."
                )

    # -------------------------------------------------
    # Conversão de coordenadas
    # -------------------------------------------------

    def _to_pixels(
        self,
        normalized_points: list[list[float]],
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Converte pontos normalizados para pixels.

        Exemplo:

            ponto normalizado:
                [0.5, 0.5]

            imagem:
                880 x 587

            ponto em pixels:
                aproximadamente (440, 294)

        Guardar as coordenadas entre 0 e 1 permite
        reutilizar a mesma configuração em resoluções
        diferentes.
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

    # -------------------------------------------------
    # Recuperação de um polígono
    # -------------------------------------------------

    def get_polygon(
        self,
        zone_name: str,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Retorna uma zona convertida para coordenadas
        em pixels.

        Exemplo:

            get_polygon(
                "red",
                880,
                587,
            )
        """

        zones = self.config["zones"]

        if zone_name not in zones:
            raise KeyError(f"Zona não encontrada: {zone_name}")

        normalized_points = zones[zone_name]

        return self._to_pixels(
            normalized_points,
            width,
            height,
        )

    # -------------------------------------------------
    # Teste ponto dentro do polígono
    # -------------------------------------------------

    @staticmethod
    def _contains(
        polygon: np.ndarray,
        point: tuple[int, int],
    ) -> bool:
        """
        Verifica se um ponto está dentro ou sobre
        a borda de um polígono.

        OpenCV pointPolygonTest retorna:

            > 0
                ponto dentro

            = 0
                ponto na borda

            < 0
                ponto fora

        Para segurança industrial, consideramos
        a borda como pertencente à zona.
        """

        result = cv2.pointPolygonTest(
            polygon,
            point,
            False,
        )

        return result >= 0

    # -------------------------------------------------
    # Classificação de risco
    # -------------------------------------------------

    def classify(
        self,
        point: tuple[int, int],
        width: int,
        height: int,
    ) -> str:
        """
        Classifica o ponto em um dos três níveis:

            CRÍTICO
            ALERTA
            SEGURO

        A ordem abaixo é proposital.

        Como a zona vermelha está dentro da amarela,
        um ponto crítico também estaria geometricamente
        dentro da zona amarela.

        Por isso verificamos primeiro a zona de maior
        severidade.
        """

        red_polygon = self.get_polygon(
            "red",
            width,
            height,
        )

        yellow_polygon = self.get_polygon(
            "yellow",
            width,
            height,
        )

        # ---------------------------------------------
        # Prioridade 1: zona vermelha
        # ---------------------------------------------

        if self._contains(
            red_polygon,
            point,
        ):
            return "CRÍTICO"

        # ---------------------------------------------
        # Prioridade 2: zona amarela
        # ---------------------------------------------

        if self._contains(
            yellow_polygon,
            point,
        ):
            return "ALERTA"

        # ---------------------------------------------
        # Fora das zonas
        # ---------------------------------------------

        return "SEGURO"
