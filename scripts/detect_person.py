import argparse
from pathlib import Path

import cv2
from app.detector import PersonDetector
from app.visualizer import annotate_risk_image
from app.zones import RiskZoneClassifier

# -------------------------------------------------
# Caminhos base do projeto
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "config" / "zones.json"


# -------------------------------------------------
# Normalização dos nomes de risco
# -------------------------------------------------


def normalize_risk_label(
    risk: str,
) -> str:
    """
    Garante que os níveis de risco sejam exibidos
    sempre no mesmo padrão em português.

    O classificador atual já retorna:

        SEGURO
        ALERTA
        CRÍTICO

    Entretanto, mantemos esta função porque ela
    também aceita nomenclaturas antigas:

        SAFE
        WARNING
        CRITICAL

    Isso deixa o script tolerante a versões
    anteriores do motor de zonas.
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
# Programa principal
# -------------------------------------------------


def main() -> None:
    """
    Executa o fluxo completo de detecção de risco.

    Pipeline:

        imagem
            ↓
        YOLO
            ↓
        pessoas detectadas
            ↓
        bounding boxes
            ↓
        foot_point
            ↓
        motor de zonas
            ↓
        SEGURO / ALERTA / CRÍTICO
            ↓
        visualizador
            ↓
        imagem anotada


    Neste ponto existe uma separação clara:

        PersonDetector
            percepção com IA

        RiskZoneClassifier
            decisão espacial

        visualizer
            apresentação do resultado

        detect_person.py
            orquestração do fluxo
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
    # 2. Validação do arquivo
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
    # 4. Inicialização do detector
    # ---------------------------------------------

    # O PersonDetector encapsula o YOLO.
    #
    # O restante deste script não precisa conhecer
    # detalhes internos do Ultralytics.

    detector = PersonDetector()

    # ---------------------------------------------
    # 5. Inferência
    # ---------------------------------------------

    detections = detector.detect(args.image)

    print(f"Pessoas detectadas: {len(detections)}")

    # ---------------------------------------------
    # 6. Inicialização do motor de zonas
    # ---------------------------------------------

    zone_classifier = RiskZoneClassifier(CONFIG_PATH)

    # ---------------------------------------------
    # 7. Classificação de cada pessoa
    # ---------------------------------------------

    # risk_results será compartilhado com
    # o visualizador.
    #
    # Estrutura:
    #
    # [
    #     (PersonDetection, "SEGURO"),
    #     (PersonDetection, "CRÍTICO"),
    # ]

    risk_results = []

    for (
        index,
        detection,
    ) in enumerate(
        detections,
        start=1,
    ):
        # -----------------------------------------
        # Classificação espacial
        # -----------------------------------------

        # A decisão de risco NÃO é feita pelo YOLO.
        #
        # O YOLO apenas detecta a pessoa.
        #
        # Depois utilizamos o foot_point para
        # descobrir em qual polígono a pessoa está.

        risk = zone_classifier.classify(
            point=(detection.foot_point),
            width=width,
            height=height,
        )

        # -----------------------------------------
        # Normalização do rótulo
        # -----------------------------------------

        risk_label = normalize_risk_label(risk)

        # -----------------------------------------
        # Armazena para o visualizador
        # -----------------------------------------

        risk_results.append(
            (
                detection,
                risk_label,
            )
        )

        # -----------------------------------------
        # Resultado no terminal
        # -----------------------------------------

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
    # 8. Recuperação das zonas em pixels
    # ---------------------------------------------

    # zones.json armazena coordenadas normalizadas.
    #
    # Aqui convertemos novamente para pixels
    # usando a resolução da imagem atual.

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
    # 9. Renderização
    # ---------------------------------------------

    # Agora toda a responsabilidade visual fica
    # dentro de app.visualizer.
    #
    # O script não precisa mais saber:
    #
    # - como desenhar bounding box
    # - qual cor usar
    # - como posicionar rótulos
    # - como desenhar zonas
    #
    # Ele apenas fornece os dados.

    annotated = annotate_risk_image(
        image=image,
        risk_results=risk_results,
        yellow_polygon=(yellow_polygon),
        red_polygon=(red_polygon),
    )

    # ---------------------------------------------
    # 10. Preparação da saída
    # ---------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # 11. Salvamento da imagem
    # ---------------------------------------------

    if not cv2.imwrite(
        str(args.output),
        annotated,
    ):
        raise RuntimeError(f"Falha ao salvar imagem anotada: {args.output}")

    # ---------------------------------------------
    # 12. Resultado final
    # ---------------------------------------------

    print(f"Imagem anotada salva em: {args.output}")


# -------------------------------------------------
# Ponto de entrada
# -------------------------------------------------

if __name__ == "__main__":
    main()
