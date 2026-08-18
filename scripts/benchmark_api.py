from __future__ import annotations

import argparse
import statistics
import urllib.request
import uuid
from pathlib import Path
from time import perf_counter

# ============================================================
# CAMINHOS PADRÃO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE = ROOT / "samples" / "input.jpg"

DEFAULT_BASE_URL = "http://127.0.0.1:8001"


# ============================================================
# PERCENTIL
# ============================================================


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """
    Calcula um percentil usando interpolação linear.

    Exemplo:

        P95 = 60 ms

    significa que aproximadamente 95% das medições
    ficaram até esse valor.

    Para APIs isso é particularmente importante,
    porque uma média pode esconder requisições lentas.
    """

    if not values:
        raise ValueError("A lista de valores não pode estar vazia.")

    if not 0 <= percentile_value <= 100:
        raise ValueError("O percentil deve estar entre 0 e 100.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * (percentile_value / 100.0)

    lower_index = int(position)

    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    fraction = position - lower_index

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
    Exibe estatísticas das requisições HTTP.
    """

    mean_ms = statistics.mean(values_ms)

    median_ms = statistics.median(values_ms)

    minimum_ms = min(values_ms)

    maximum_ms = max(values_ms)

    p95_ms = percentile(
        values_ms,
        95,
    )

    std_ms = statistics.pstdev(values_ms) if len(values_ms) > 1 else 0.0

    requests_per_second = 1000.0 / median_ms if median_ms > 0 else float("inf")

    print()

    print(label)

    print("-" * len(label))

    print(f"Execuções:              {len(values_ms)}")

    print(f"Mínimo:                 {minimum_ms:.3f} ms")

    print(f"Média:                  {mean_ms:.3f} ms")

    print(f"Mediana:                {median_ms:.3f} ms")

    print(f"P95:                    {p95_ms:.3f} ms")

    print(f"Máximo:                 {maximum_ms:.3f} ms")

    print(f"Desvio padrão:          {std_ms:.3f} ms")

    print(f"Req/s teórico*:         {requests_per_second:.2f}")


# ============================================================
# CORPO MULTIPART
# ============================================================


def build_multipart_body(
    image_path: Path,
) -> tuple[
    bytes,
    str,
]:
    """
    Constrói manualmente um corpo HTTP
    multipart/form-data.

    Normalmente bibliotecas como:

        requests
        httpx

    fariam isso automaticamente.

    Aqui usamos apenas a biblioteca padrão do Python
    para não adicionar uma dependência ao projeto
    exclusivamente para o benchmark.


    O servidor espera um campo chamado:

        file

    porque nossos endpoints utilizam:

        file: UploadFile = File(...)
    """

    # --------------------------------------------------------
    # Carrega a imagem uma única vez
    # --------------------------------------------------------
    #
    # O acesso ao disco NÃO entra em cada medição.
    #
    # Queremos observar principalmente o custo HTTP +
    # servidor, e não o desempenho do SSD.
    # --------------------------------------------------------

    image_bytes = image_path.read_bytes()

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------
    #
    # multipart/form-data separa cada campo utilizando
    # uma string chamada boundary.
    #
    # Geramos uma identificadora única para evitar
    # colisão acidental com o conteúdo da imagem.
    # --------------------------------------------------------

    boundary = "----PersonDetectedBoundary" + uuid.uuid4().hex

    # --------------------------------------------------------
    # Cabeçalho interno do campo "file"
    # --------------------------------------------------------

    header = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; "
        'name="file"; filename="input.jpg"\r\n'
        "Content-Type: image/jpeg\r\n"
        "\r\n"
    ).encode("utf-8")

    # --------------------------------------------------------
    # Fechamento do multipart
    # --------------------------------------------------------

    footer = (f"\r\n--{boundary}--\r\n").encode("utf-8")

    # --------------------------------------------------------
    # Corpo final
    # --------------------------------------------------------

    body = header + image_bytes + footer

    content_type = f"multipart/form-data; boundary={boundary}"

    return (
        body,
        content_type,
    )


# ============================================================
# EXECUÇÃO DE UMA REQUISIÇÃO
# ============================================================


def execute_request(
    url: str,
    body: bytes,
    content_type: str,
) -> tuple[
    float,
    bytes,
    str | None,
]:
    """
    Executa uma requisição POST e mede o tempo E2E
    observado pelo cliente.

    O cronômetro começa imediatamente antes de
    urlopen() e termina somente depois que:

        resposta.read()

    terminou.

    Portanto, a medição inclui o recebimento completo
    da resposta HTTP.
    """

    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": (content_type),
        },
    )

    start = perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        response_body = response.read()

        response_content_type = response.headers.get("Content-Type")

    elapsed_ms = (perf_counter() - start) * 1000.0

    return (
        elapsed_ms,
        response_body,
        response_content_type,
    )


# ============================================================
# BENCHMARK DE UM ENDPOINT
# ============================================================


def benchmark_endpoint(
    label: str,
    url: str,
    body: bytes,
    content_type: str,
    warmup: int,
    runs: int,
    expected_content_type: str,
) -> list[float]:
    """
    Executa warm-up e depois as requisições medidas.

    Também valida o Content-Type retornado.

    Isso evita considerar um erro HTTP ou uma resposta
    inesperada como uma medição válida.
    """

    print()

    print(label)

    print(f"Endpoint: {url}")

    # ========================================================
    # WARM-UP
    # ========================================================

    print(f"Warm-up: {warmup}")

    for _ in range(warmup):
        (
            _elapsed_ms,
            _response_body,
            response_content_type,
        ) = execute_request(
            url=url,
            body=body,
            content_type=content_type,
        )

        if (
            response_content_type is None
            or expected_content_type not in response_content_type
        ):
            raise RuntimeError(
                f"Content-Type inesperado durante warm-up: {response_content_type}"
            )

    # ========================================================
    # MEDIÇÕES
    # ========================================================

    print(f"Medições: {runs}")

    measurements: list[float] = []

    for run_index in range(
        1,
        runs + 1,
    ):
        (
            elapsed_ms,
            response_body,
            response_content_type,
        ) = execute_request(
            url=url,
            body=body,
            content_type=content_type,
        )

        # ----------------------------------------------------
        # Valida tipo de resposta
        # ----------------------------------------------------

        if (
            response_content_type is None
            or expected_content_type not in response_content_type
        ):
            raise RuntimeError(f"Content-Type inesperado: {response_content_type}")

        # ----------------------------------------------------
        # Valida que existe conteúdo
        # ----------------------------------------------------

        if not response_body:
            raise RuntimeError("A API retornou uma resposta vazia.")

        measurements.append(elapsed_ms)

        print(f"  {run_index:02d}: {elapsed_ms:.3f} ms")

    return measurements


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================


def main() -> None:
    """
    Mede a latência HTTP ponta a ponta dos dois
    endpoints principais do projeto.

    Endpoint JSON:

        cliente
          ↓
        HTTP multipart
          ↓
        FastAPI
          ↓
        OpenCV decode
          ↓
        YOLO
          ↓
        classificação espacial
          ↓
        JSON
          ↓
        cliente


    Endpoint PNG:

        cliente
          ↓
        HTTP multipart
          ↓
        FastAPI
          ↓
        OpenCV decode
          ↓
        YOLO
          ↓
        classificação espacial
          ↓
        renderização
          ↓
        encode PNG
          ↓
        cliente
    """

    parser = argparse.ArgumentParser(description=("Benchmark HTTP E2E da API."))

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=("URL base da API. Padrão: http://127.0.0.1:8001"),
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=("Imagem enviada aos endpoints."),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help=("Quantidade de requisições de aquecimento."),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help=("Quantidade de requisições medidas."),
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    if not args.image.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")

    if args.warmup < 0:
        raise ValueError("--warmup não pode ser negativo.")

    if args.runs <= 0:
        raise ValueError("--runs deve ser maior que zero.")

    # ========================================================
    # PREPARAÇÃO DO MULTIPART
    # ========================================================

    (
        body,
        content_type,
    ) = build_multipart_body(args.image)

    print("Benchmark HTTP E2E")

    print(f"Imagem:     {args.image}")

    print(f"Base URL:   {args.base_url}")

    print(f"Tamanho:    {len(body)} bytes")

    # ========================================================
    # ENDPOINT JSON
    # ========================================================

    json_times = benchmark_endpoint(
        label=("1. Inferência com resposta JSON"),
        url=(args.base_url.rstrip("/") + "/api/v1/infer"),
        body=body,
        content_type=content_type,
        warmup=args.warmup,
        runs=args.runs,
        expected_content_type=("application/json"),
    )

    # ========================================================
    # ENDPOINT PNG
    # ========================================================

    png_times = benchmark_endpoint(
        label=("2. Inferência com resposta PNG"),
        url=(args.base_url.rstrip("/") + "/api/v1/infer/annotated"),
        body=body,
        content_type=content_type,
        warmup=args.warmup,
        runs=args.runs,
        expected_content_type=("image/png"),
    )

    # ========================================================
    # RESUMO
    # ========================================================

    summarize(
        "1. HTTP E2E - JSON",
        json_times,
    )

    summarize(
        "2. HTTP E2E - PNG anotado",
        png_times,
    )

    print()

    print("* Req/s teórico considera requisições sequenciais baseadas na mediana.")

    print("  Não representa capacidade sob concorrência.")


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()
