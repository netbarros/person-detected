from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from time import perf_counter

import cv2
from app.detector import PersonDetector
from app.zones import RiskZoneClassifier

# ============================================================
# CAMINHOS PADRÃO DO PROJETO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE = ROOT / "samples" / "input.jpg"

CONFIG_PATH = ROOT / "config" / "zones.json"


# ============================================================
# FUNÇÃO AUXILIAR: PERCENTIL
# ============================================================


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """
    Calcula um percentil por interpolação linear.

    Exemplo conceitual:

        P95

    significa que aproximadamente 95% das medições
    ficaram abaixo desse valor.

    Isso é importante porque uma média sozinha pode
    esconder execuções muito lentas.

    Implementamos aqui para não adicionar uma
    dependência externa apenas para estatística básica.
    """

    if not values:
        raise ValueError("A lista de valores não pode estar vazia.")

    if not 0 <= percentile_value <= 100:
        raise ValueError("O percentil deve estar entre 0 e 100.")

    ordered = sorted(values)

    # Com somente um elemento não existe necessidade
    # de interpolação.
    if len(ordered) == 1:
        return ordered[0]

    # --------------------------------------------------------
    # Posição teórica dentro da lista ordenada
    # --------------------------------------------------------

    position = (len(ordered) - 1) * (percentile_value / 100.0)

    lower_index = int(position)

    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    fraction = position - lower_index

    # --------------------------------------------------------
    # Interpolação linear
    # --------------------------------------------------------

    return (
        ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


# ============================================================
# RESUMO ESTATÍSTICO
# ============================================================


def summarize(
    label: str,
    values_ms: list[float],
) -> None:
    """
    Exibe estatísticas de uma série de medições.

    Não queremos apresentar somente:

        "levou 45 ms"

    porque uma única execução pode ser enganosa.

    Vamos observar:

        mínimo
        média
        mediana
        P95
        máximo
        desvio padrão

    Também calculamos um FPS TEÓRICO baseado na mediana.

    IMPORTANTE:

    Esse FPS não é automaticamente o FPS real de câmera.

    Ele mede apenas:

        1000 ms / tempo mediano da etapa medida

    Um sistema real ainda pode incluir:

        captura da câmera
        resize
        decodificação
        HTTP
        desenho
        encode
        armazenamento
        rede
    """

    if not values_ms:
        raise ValueError("Não há medições para resumir.")

    mean_ms = statistics.mean(values_ms)

    median_ms = statistics.median(values_ms)

    minimum_ms = min(values_ms)

    maximum_ms = max(values_ms)

    p95_ms = percentile(
        values_ms,
        95,
    )

    # pstdev:
    #
    # desvio padrão populacional das medições
    # observadas neste experimento.
    std_ms = statistics.pstdev(values_ms) if len(values_ms) > 1 else 0.0

    fps_from_median = 1000.0 / median_ms if median_ms > 0 else float("inf")

    print()

    print(label)

    print("-" * len(label))

    print(f"Execuções:      {len(values_ms)}")

    print(f"Mínimo:         {minimum_ms:.3f} ms")

    print(f"Média:          {mean_ms:.3f} ms")

    print(f"Mediana:        {median_ms:.3f} ms")

    print(f"P95:            {p95_ms:.3f} ms")

    print(f"Máximo:         {maximum_ms:.3f} ms")

    print(f"Desvio padrão:  {std_ms:.3f} ms")

    print(f"FPS teórico*:   {fps_from_median:.2f}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================


def main() -> None:
    """
    Mede duas partes do sistema.

    1. DETECTOR

       Mede somente:

           PersonDetector.detect(image)

       Aqui observamos principalmente o custo do YOLO.


    2. PIPELINE

       Mede:

           YOLO
             +
           obtenção do foot_point
             +
           classificação espacial


    Também fazemos warm-up antes das medições.

    Isso é importante porque as primeiras inferências podem
    apresentar custos extras de:

        inicialização
        alocação de memória
        preparação do backend
        cache
    """

    # --------------------------------------------------------
    # Argumentos
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=("Benchmark do detector e do pipeline de classificação de risco.")
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=("Imagem utilizada no benchmark. Padrão: samples/input.jpg"),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help=("Quantidade de execuções de aquecimento. Padrão: 5."),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help=("Quantidade de execuções medidas. Padrão: 30."),
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDAÇÕES
    # ========================================================

    if args.warmup < 0:
        raise ValueError("--warmup não pode ser negativo.")

    if args.runs <= 0:
        raise ValueError("--runs deve ser maior que zero.")

    if not args.image.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")

    # ========================================================
    # CARREGAMENTO DA IMAGEM
    # ========================================================
    #
    # Carregamos a imagem UMA vez.
    #
    # Assim o tempo de acesso ao disco não entra
    # no benchmark.
    # --------------------------------------------------------

    image = cv2.imread(str(args.image))

    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {args.image}")

    height, width = image.shape[:2]

    # ========================================================
    # COMPONENTES
    # ========================================================
    #
    # O modelo também é carregado antes da medição.
    #
    # Queremos benchmark de inferência em regime de execução,
    # não benchmark de inicialização do programa.
    # --------------------------------------------------------

    detector = PersonDetector()

    zone_classifier = RiskZoneClassifier(CONFIG_PATH)

    print("Benchmark do pipeline Edge AI")

    print(f"Imagem:         {args.image}")

    print(f"Resolução:      {width}x{height}")

    print(f"Warm-up:        {args.warmup}")

    print(f"Execuções:      {args.runs}")

    # ========================================================
    # WARM-UP
    # ========================================================
    #
    # Essas execuções NÃO entram no resultado final.
    # --------------------------------------------------------

    print()

    print("Executando warm-up...")

    for _ in range(args.warmup):
        detector.detect(image)

    # ========================================================
    # BENCHMARK 1
    #
    # DETECTOR YOLO
    # ========================================================

    detector_times_ms: list[float] = []

    print("Medindo detector...")

    for _ in range(args.runs):
        start = perf_counter()

        detector.detect(image)

        elapsed_ms = (perf_counter() - start) * 1000.0

        detector_times_ms.append(elapsed_ms)

    # ========================================================
    # BENCHMARK 2
    #
    # PIPELINE FUNCIONAL
    # ========================================================
    #
    # Mede:
    #
    #     YOLO
    #       +
    #     classificação espacial
    #
    #
    # NÃO mede:
    #
    #     leitura JPEG
    #     upload HTTP
    #     multipart
    #     desenho
    #     encode PNG
    #
    # Isso precisa ser declarado quando apresentarmos
    # os resultados.
    # --------------------------------------------------------

    pipeline_times_ms: list[float] = []

    persons_per_run: list[int] = []

    print("Medindo pipeline...")

    for _ in range(args.runs):
        start = perf_counter()

        # ----------------------------------------------------
        # IA: percepção
        # ----------------------------------------------------

        detections = detector.detect(image)

        # ----------------------------------------------------
        # Regra determinística: decisão espacial
        # ----------------------------------------------------

        for detection in detections:
            zone_classifier.classify(
                point=(detection.foot_point),
                width=width,
                height=height,
            )

        elapsed_ms = (perf_counter() - start) * 1000.0

        pipeline_times_ms.append(elapsed_ms)

        persons_per_run.append(len(detections))

    # ========================================================
    # CONSISTÊNCIA FUNCIONAL
    # ========================================================
    #
    # Performance sem estabilidade funcional não basta.
    #
    # Portanto verificamos também se a quantidade de pessoas
    # detectadas permaneceu estável entre as execuções.
    # --------------------------------------------------------

    unique_person_counts = sorted(set(persons_per_run))

    print()

    print(f"Contagens de pessoas observadas: {unique_person_counts}")

    if len(unique_person_counts) != 1:
        print("ATENÇÃO: a quantidade de pessoas variou entre as execuções.")

    # ========================================================
    # RESULTADOS
    # ========================================================

    summarize(
        "1. Detector YOLO",
        detector_times_ms,
    )

    summarize(
        ("2. Pipeline YOLO + classificação de risco"),
        pipeline_times_ms,
    )

    # ========================================================
    # OBSERVAÇÃO SOBRE FPS
    # ========================================================

    print()

    print("* FPS teórico calculado apenas a partir da mediana da etapa medida.")

    print(
        "  Não deve ser apresentado como FPS "
        "real de câmera sem benchmark "
        "do fluxo completo."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()
