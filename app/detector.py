from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
    PERSON_CLASS_ID,
)

# -------------------------------------------------
# Estrutura de uma detecção
# -------------------------------------------------


@dataclass(frozen=True)
class PersonDetection:
    """
    Representa uma pessoa detectada pelo modelo.

    Armazenamos:

        confidence
            confiança da detecção

        bbox
            bounding box no formato:

            (x1, y1, x2, y2)

    Também expomos o foot_point, que será usado
    pelo motor de zonas.
    """

    confidence: float

    bbox: tuple[
        int,
        int,
        int,
        int,
    ]

    # -------------------------------------------------
    # Ponto dos pés
    # -------------------------------------------------

    @property
    def foot_point(
        self,
    ) -> tuple[int, int]:
        """
        Retorna o ponto central inferior da bounding box.

        Exemplo:

            x1, y1
            +-------------+
            |             |
            |   pessoa    |
            |             |
            +------X------+
                  ^
                  |
              foot_point

        Esse ponto representa melhor a posição da pessoa
        no piso do que o centro da bounding box.
        """

        x1, y1, x2, y2 = self.bbox

        foot_x = (x1 + x2) // 2

        foot_y = y2

        return (
            foot_x,
            foot_y,
        )


# -------------------------------------------------
# Detector de pessoas
# -------------------------------------------------


class PersonDetector:
    """
    Encapsula o modelo YOLO usado na aplicação.

    O detector aceita duas formas de entrada:

        1. caminho para uma imagem

            samples/input.jpg

        2. imagem já carregada em memória

            numpy.ndarray

    Essa segunda opção será importante para a API.

    Quando uma imagem chegar via HTTP, podemos
    decodificá-la em memória e enviar diretamente
    para o YOLO, sem criar arquivo temporário.
    """

    # -------------------------------------------------
    # Inicialização
    # -------------------------------------------------

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        """
        Carrega o modelo YOLO.

        O objetivo é criar a instância uma única vez e
        reutilizá-la.

        Isso será especialmente importante na API, porque
        carregar os pesos do modelo a cada requisição
        teria um custo muito alto.
        """

        self.model = YOLO(model_name)

        self.confidence = confidence

    # -------------------------------------------------
    # Inferência
    # -------------------------------------------------

    def detect(
        self,
        source: (str | Path | np.ndarray),
    ) -> list[PersonDetection]:
        """
        Executa a inferência do YOLO.

        O parâmetro source pode ser:

            str
                caminho para arquivo

            Path
                pathlib.Path

            numpy.ndarray
                imagem OpenCV já carregada em memória


        Exemplo usando arquivo:

            detector.detect(
                "samples/input.jpg"
            )


        Exemplo usando imagem em memória:

            image = cv2.imread(
                "samples/input.jpg"
            )

            detector.detect(
                image
            )


        O modelo é configurado para retornar somente
        a classe PERSON_CLASS_ID.

        No conjunto COCO:

            class_id = 0
            class_name = person
        """

        # O Ultralytics aceita string diretamente.
        #
        # Caso recebamos pathlib.Path,
        # convertemos para string.

        if isinstance(
            source,
            Path,
        ):
            source = str(source)

        # ---------------------------------------------
        # Execução do modelo
        # ---------------------------------------------

        results = self.model.predict(
            source=source,
            # Queremos somente pessoas.
            classes=[PERSON_CLASS_ID],
            # Threshold mínimo de confiança.
            conf=(self.confidence),
            # Evita logs internos excessivos.
            verbose=False,
        )

        # ---------------------------------------------
        # Resultado simplificado da aplicação
        # ---------------------------------------------

        detections: list[PersonDetection] = []

        # ---------------------------------------------
        # Percorre os resultados do modelo
        # ---------------------------------------------

        for result in results:
            if result.boxes is None:
                continue

            # -----------------------------------------
            # Cada box representa uma detecção
            # -----------------------------------------

            for box in result.boxes:
                # YOLO retorna:
                #
                # x1, y1, x2, y2

                coords = box.xyxy[0].tolist()

                # Coordenadas de desenho são
                # convertidas para inteiros.

                x1, y1, x2, y2 = map(
                    int,
                    coords,
                )

                # Confiança do modelo.
                confidence = float(box.conf[0])

                # Criamos nosso objeto de domínio.
                #
                # A partir daqui, o restante do projeto
                # não precisa conhecer os objetos internos
                # do Ultralytics.

                detections.append(
                    PersonDetection(
                        confidence=(confidence),
                        bbox=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    )
                )

        return detections

    # -------------------------------------------------
    # Anotador simples
    # -------------------------------------------------

    @staticmethod
    def annotate(
        image_path: str | Path,
        detections: Iterable[PersonDetection],
        output_path: str | Path,
    ) -> None:
        """
        Desenha uma visualização simples das pessoas.

        Esse método foi usado nos primeiros slices.

        O script principal já possui uma visualização
        mais completa com:

            SEGURO
            ALERTA
            CRÍTICO
            zonas de risco

        Mantemos este método porque ele ainda é útil
        como exemplo isolado e para depuração.
        """

        # ---------------------------------------------
        # Carrega imagem
        # ---------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

        # ---------------------------------------------
        # Desenha cada detecção
        # ---------------------------------------------

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            foot_x, foot_y = detection.foot_point

            # Bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Foot point
            cv2.circle(
                image,
                (
                    foot_x,
                    foot_y,
                ),
                5,
                (0, 0, 255),
                -1,
            )

            # Rótulo simples
            label = f"person {detection.confidence:.2f}"

            cv2.putText(
                image,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ---------------------------------------------
        # Prepara saída
        # ---------------------------------------------

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------
        # Salva imagem
        # ---------------------------------------------

        if not cv2.imwrite(
            str(output_path),
            image,
        ):
            raise RuntimeError(f"Falha ao salvar imagem em: {output_path}")
