import json
from pathlib import Path

import cv2
import numpy as np


class RiskZoneClassifier:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

    def _to_pixels(
        self,
        normalized_points: list[list[float]],
        width: int,
        height: int,
    ) -> np.ndarray:
        points = [
            (
                int(x * width),
                int(y * height),
            )
            for x, y in normalized_points
        ]

        return np.array(points, dtype=np.int32)

    def get_polygon(
        self,
        zone_name: str,
        width: int,
        height: int,
    ) -> np.ndarray:
        normalized_points = self.config["zones"][zone_name]

        return self._to_pixels(
            normalized_points,
            width,
            height,
        )

    def classify(
        self,
        point: tuple[int, int],
        width: int,
        height: int,
    ) -> str:
        red_points = self.config["zones"]["red"]

        red_polygon = self._to_pixels(
            red_points,
            width,
            height,
        )

        result = cv2.pointPolygonTest(
            red_polygon,
            point,
            False,
        )

        if result >= 0:
            return "CRÍTICO"

        return "SEGURO"
