from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO

from app.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL, PERSON_CLASS_ID


@dataclass(frozen=True)
class PersonDetection:
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def foot_point(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)


class PersonDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, image_path: str | Path) -> list[PersonDetection]:
        results = self.model.predict(
            source=str(image_path),
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            verbose=False,
        )

        detections: list[PersonDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                confidence = float(box.conf[0])

                detections.append(
                    PersonDetection(
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections

    @staticmethod
    def annotate(
        image_path: str | Path,
        detections: Iterable[PersonDetection],
        output_path: str | Path,
    ) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            foot_x, foot_y = detection.foot_point

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (foot_x, foot_y), 5, (0, 0, 255), -1)

            label = f"person {detection.confidence:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Falha ao salvar imagem em: {output_path}")
