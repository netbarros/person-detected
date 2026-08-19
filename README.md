# Person Detected — Edge AI Risk Zone Monitor

Projeto técnico de **Visão Computacional Embarcada (Edge AI)** para detecção de pessoas, classificação de risco em zonas próximas a máquinas e geração de alertas proporcionais ao nível de perigo.

A solução foi estruturada para demonstrar, de forma reproduzível e explicável:

- detecção de pessoas com modelo pré-treinado;
- pós-processamento espacial por zonas;
- classificação `SEGURO / ALERTA / CRÍTICO`;
- política de alertas proporcional ao risco;
- API HTTP com retorno JSON e PNG anotado;
- containerização com Docker;
- execução com Docker Compose;
- build multi-arquitetura `linux/amd64` e `linux/arm64`;
- análise de viabilidade para Raspberry Pi 5;
- benchmark em CPU sem GPU;
- testes automatizados e validação E2E.

> **Princípio arquitetural:** IA para percepção; regras determinísticas para decisão espacial e política de alerta.

---

## 1. Problema e objetivo

Em ambientes industriais, pessoas podem se aproximar de regiões onde máquinas, equipamentos móveis ou zonas operacionais apresentam risco.

O objetivo deste projeto é analisar uma imagem ou frame de câmera, identificar pessoas e determinar se cada pessoa está:

| Estado    | Significado             |
| --------- | ----------------------- |
| `SEGURO`  | fora das zonas de risco |
| `ALERTA`  | dentro da zona amarela  |
| `CRÍTICO` | dentro da zona vermelha |

A zona vermelha possui prioridade sobre a zona amarela.

Além da classificação, o sistema produz uma **decisão de alerta proporcional ao maior risco observado no frame**.

Fluxo principal:

```text
imagem/frame
    ↓
pré-processamento
    ↓
YOLO11n
    ↓
detecção de person
    ↓
bounding box
    ↓
foot_point
    ↓
motor de zonas
    ↓
SEGURO / ALERTA / CRÍTICO
    ↓
política de alerta
    ↓
JSON / PNG anotado / log de alerta
```

---

## 2. Escopo da solução

O projeto utiliza um modelo YOLO pré-treinado e filtra somente a classe `person`.

Não foi realizado treinamento customizado nesta etapa.

O protótipo contempla:

```text
detecção de pessoas
bounding boxes
confidence
foot_point
zona amarela
zona vermelha
classificação de risco
alerta proporcional
imagem anotada
API HTTP
Docker
Docker Compose
build AMD64
build ARM64
Buildx multiarch
benchmark
testes automatizados
validação E2E
```

O Raspberry Pi 5 é tratado como **alvo de implantação Edge**, mas os benchmarks documentados neste repositório foram medidos em outro hardware. Compatibilidade ARM64 não é confundida com performance real na placa.

---

## 3. Arquitetura

```text
                      +----------------------+
                      |   Upload / Imagem    |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      | Pré-processamento    |
                      | bytes -> OpenCV BGR  |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      |      YOLO11n         |
                      | classe: person       |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      | Bounding Box + Conf. |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      |     Foot Point       |
                      | centro inferior bbox |
                      +----------+-----------+
                                 |
                                 v
                      +----------------------+
                      |   Motor de Zonas     |
                      | pointPolygonTest()   |
                      +----------+-----------+
                                 |
                  +--------------+--------------+
                  |              |              |
                  v              v              v
               SEGURO         ALERTA         CRÍTICO
                  |              |              |
                  +--------------+--------------+
                                 |
                                 v
                      +----------------------+
                      | Política de Alerta   |
                      +----------+-----------+
                                 |
                     +-----------+-----------+
                     |                       |
                     v                       v
                 JSON API                PNG anotado
                     |
                     v
                log / integração
```

A arquitetura separa responsabilidades:

- **YOLO:** percepção probabilística;
- **motor de zonas:** decisão espacial determinística;
- **política de alerta:** decisão operacional determinística;
- **visualizador:** evidência visual;
- **API:** contrato de integração;
- **Docker/Compose:** empacotamento e execução reproduzível.

---

## 4. Pipeline de visão computacional

### 4.1 Pré-processamento

Na API, o arquivo enviado por HTTP é processado em memória:

```text
UploadFile
→ bytes
→ np.frombuffer()
→ cv2.imdecode()
→ imagem BGR
```

Não é necessário criar arquivo temporário no servidor.

Depois disso, a imagem é entregue ao runner do Ultralytics. O pré-processamento específico necessário ao modelo é delegado à biblioteca de inferência, evitando duplicação de transformações dentro da aplicação.

### 4.2 Inferência

```text
imagem BGR
→ YOLO11n
→ filtro da classe person
→ confidence threshold = 0.40
→ bounding boxes
```

O detector da aplicação encapsula os objetos internos do Ultralytics e devolve objetos de domínio `PersonDetection`.

### 4.3 Pós-processamento

```text
bounding box
→ foot_point
→ pointPolygonTest
→ risco espacial
→ política de alerta
→ contrato JSON ou visualização PNG
```

Essa divisão é proposital: a parte probabilística fica restrita à percepção; as regras espaciais e de alerta são explícitas e testáveis.

---

## 5. Escolha do modelo — YOLO11n

O projeto utiliza:

```text
yolo11n.pt
```

A variante Nano foi adotada como **baseline pré-treinado** por oferecer um compromisso adequado para Edge AI entre:

- tamanho do modelo;
- custo computacional;
- latência;
- uso de memória;
- simplicidade de implantação;
- capacidade de detectar a classe `person`.

A escolha **não afirma que YOLO11n seja universalmente o modelo mais preciso**.

Para este protótipo, o objetivo é demonstrar uma solução funcional e viável para execução em CPU/Edge. Em implantação industrial real, a escolha final deve ser baseada em comparação experimental entre modelos e runtimes no hardware e dataset alvo.

### Precisão no cenário industrial

A imagem de demonstração prova o funcionamento do pipeline, mas **não constitui um dataset de validação de acurácia**.

Para medir precisão de forma adequada, seria necessário construir um conjunto rotulado com as condições reais do cenário, incluindo:

```text
câmera elevada
múltiplas pessoas
variação de iluminação
fumaça ou névoa
oclusões
vibração
distância
perspectiva
motion blur
```

As métricas relevantes incluem:

```text
Precision
Recall
F1-score
IoU
mAP
matriz de confusão
```

Em um contexto de monitoramento de risco, falsos negativos merecem atenção especial: se uma pessoa não é detectada, o motor de zonas não consegue avaliar sua posição.

Se a baseline não atingir os critérios definidos para o ambiente real, os próximos passos seriam fine-tuning com dados representativos e comparação com outras variantes/modelos.

---

## 6. Bounding box e foot point

O detector retorna uma bounding box no formato:

```text
(x1, y1)
   +----------------+
   |                |
   |     pessoa     |
   |                |
   +----------------+
                  (x2, y2)
```

Para representar a posição aproximada da pessoa no piso, o projeto utiliza o ponto central inferior:

```python
foot_x = (x1 + x2) // 2
foot_y = y2
```

Visualmente:

```text
   +----------------+
   |                |
   |     pessoa     |
   |                |
   +--------●-------+
            ^
        foot_point
```

O centro da bounding box representa aproximadamente o tronco. Para zonas desenhadas sobre o piso, o `foot_point` é uma aproximação espacial mais adequada.

---

## 7. Zonas de risco

As zonas são armazenadas em:

```text
config/zones.json
```

As coordenadas são normalizadas entre `0` e `1`, permitindo reutilizar a configuração em diferentes resoluções.

O sistema possui:

```text
yellow → ALERTA
red    → CRÍTICO
```

A classificação utiliza:

```python
cv2.pointPolygonTest(...)
```

A borda do polígono é considerada parte da zona.

A prioridade é:

```text
se estiver na zona vermelha
→ CRÍTICO

senão, se estiver na zona amarela
→ ALERTA

senão
→ SEGURO
```

A zona vermelha é verificada primeiro porque pode estar geometricamente contida na zona amarela.

---

## 8. Alertas proporcionais ao risco

Além de classificar o risco espacial, o sistema converte o maior risco observado no frame em uma decisão de alerta.

Política implementada:

| Risco espacial | Nível de alerta | Ação lógica                      |
| -------------- | --------------- | -------------------------------- |
| `SEGURO`       | `NONE`          | `NONE`                           |
| `ALERTA`       | `WARNING`       | `WARN_OPERATOR`                  |
| `CRÍTICO`      | `CRITICAL`      | `REQUEST_IMMEDIATE_INTERVENTION` |

Exemplo com múltiplas pessoas:

```text
Pessoa A → SEGURO
Pessoa B → ALERTA
Pessoa C → CRÍTICO

Alerta global → CRITICAL
```

A política está isolada em `app/alerts.py`.

Nesta versão, o `AlertDispatcher` registra alertas ativos em log. Isso permite demonstrar resposta automática de forma reproduzível sem alegar a existência de hardware físico que não está conectado.

O dispatcher é o ponto de extensão para integrações futuras, por exemplo:

```text
GPIO
sinalizador luminoso/sonoro
MQTT
CLP
relé
sistema supervisório
```

### Limite importante

O projeto **não implementa uma função de segurança certificada** e não deve comandar diretamente parada ou intertravamento de máquina sem a engenharia, análise de risco, validação e certificação aplicáveis.

---

## 9. Estrutura do projeto

```text
person-detected/
│
├── app/
│   ├── __init__.py
│   ├── alerts.py
│   ├── api.py
│   ├── config.py
│   ├── detector.py
│   ├── visualizer.py
│   └── zones.py
│
├── config/
│   └── zones.json
│
├── scripts/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── benchmark_api.py
│   ├── detect_person.py
│   ├── mark_zones.py
│   ├── test_zones.py
│   └── verify_part1.ps1
│
├── tests/
│   ├── __init__.py
│   ├── test_alerts.py
│   ├── test_api.py
│   └── test_zones.py
│
├── samples/
│   └── input.jpg
│
├── outputs/
│   └── person-detected.png
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
├── yolo11n.pt
└── README.md
```

---

## 10. Componentes principais

### `app/detector.py`

Responsável por:

```text
carregar YOLO
executar inferência
filtrar person
converter resultado para PersonDetection
fornecer bbox e foot_point
```

### `app/zones.py`

Responsável por:

```text
carregar zones.json
validar configuração
converter coordenadas normalizadas
construir polígonos
testar foot_point
retornar SEGURO / ALERTA / CRÍTICO
```

### `app/alerts.py`

Responsável por:

```text
selecionar maior risco do frame
converter risco em nível de alerta
definir ação lógica
despachar alerta disponível no protótipo
```

### `app/visualizer.py`

Responsável por desenhar:

```text
zonas
bounding boxes
foot points
risco
confidence
```

### `app/api.py`

Responsável por orquestrar:

```text
upload
decode
inferência
classificação espacial
alerta
JSON / PNG
```

### `scripts/`

Ferramentas de:

```text
calibração
execução local
benchmark
verificação E2E
```

### `tests/`

Testes automatizados da:

```text
API
política de alerta
classificação de zonas
```

---

## 11. Requisitos

O container utiliza Python 3.11.

O ambiente local utilizado durante os testes de desenvolvimento apresentou Python 3.13.5.

Principais dependências:

```text
Ultralytics
OpenCV Headless
NumPy
FastAPI
Uvicorn
python-multipart
pytest
httpx2
```

Instalação:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 12. Execução local do pipeline

Execute:

```powershell
python -m scripts.detect_person samples/input.jpg
```

A saída é gerada em:

```text
outputs/person-detected.png
```

O terminal apresenta, para cada pessoa:

```text
confidence
bounding box
foot_point
risco
```

---

## 13. Calibração das zonas

Zona vermelha:

```powershell
python -m scripts.mark_zones red
```

Zona amarela:

```powershell
python -m scripts.mark_zones yellow
```

Comandos da interface:

```text
clique esquerdo → adiciona ponto
U               → desfaz último ponto
C               → limpa pontos
S               → salva
ESC             → encerra
```

As coordenadas são persistidas em:

```text
config/zones.json
```

---

## 14. API HTTP

Para executar localmente:

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Dentro do container, o Uvicorn escuta em:

```text
0.0.0.0:8000
```

A publicação para o host é feita pelo Docker/Compose.

Os exemplos abaixo usam `localhost`. A API não precisa estar publicada na Internet para demonstrar os endpoints HTTP funcionais.

---

## 15. Endpoint de health check

```http
GET /health
```

Exemplo:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Resposta:

```json
{
  "status": "ok",
  "model": "yolo11n.pt"
}
```

O health check não executa inferência YOLO.

---

## 16. Endpoint 1 — inferência com JSON

```http
POST /api/v1/infer
```

Exemplo:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/api/v1/infer" `
  -F "file=@samples/input.jpg"
```

Contrato de resposta:

```json
{
  "filename": "input.jpg",
  "image": {
    "width": 880,
    "height": 587
  },
  "model": "yolo11n.pt",
  "confidence_threshold": 0.4,
  "inference_ms": 45.383,
  "persons_detected": 2,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.931599,
      "bbox": {
        "x1": 531,
        "y1": 16,
        "x2": 756,
        "y2": 583
      },
      "foot_point": {
        "x": 643,
        "y": 583
      },
      "risk": "SEGURO"
    },
    {
      "class_name": "person",
      "confidence": 0.870626,
      "bbox": {
        "x1": 382,
        "y1": 159,
        "x2": 505,
        "y2": 555
      },
      "foot_point": {
        "x": 443,
        "y": 555
      },
      "risk": "CRÍTICO"
    }
  ],
  "alert": {
    "active": true,
    "level": "CRITICAL",
    "source_risk": "CRÍTICO",
    "action": "REQUEST_IMMEDIATE_INTERVENTION",
    "message": "Pessoa detectada na zona vermelha: solicitar intervenção imediata do sistema responsável."
  }
}
```

`inference_ms` é medido em runtime e varia entre execuções.

---

## 17. Endpoint 2 — imagem PNG anotada

```http
POST /api/v1/infer/annotated
```

Exemplo:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/api/v1/infer/annotated" `
  -F "file=@samples/input.jpg" `
  --output outputs/api-annotated.png
```

Resposta:

```text
Content-Type: image/png
```

A imagem contém:

```text
zona amarela
zona vermelha
bounding boxes
foot points
risco
confidence
```

O endpoint também expõe headers de alerta:

```text
X-Alert-Active
X-Alert-Level
X-Alert-Action
```

Exemplo para ocorrência crítica:

```text
X-Alert-Active: true
X-Alert-Level: CRITICAL
X-Alert-Action: REQUEST_IMMEDIATE_INTERVENTION
```

### Unicode na imagem

O valor interno permanece:

```text
CRÍTICO
```

No desenho OpenCV é exibido:

```text
CRITICO
```

porque as fontes Hershey usadas por `cv2.putText()` não oferecem suporte Unicode completo.

---

## 18. Docker

Build local:

```powershell
docker build -t person-detected:local .
```

Execução direta:

```powershell
docker run --rm `
  -p 8001:8000 `
  --name person-detected `
  person-detected:local
```

Neste exemplo:

```text
host:8001 → container:8000
```

A porta `8001` no host foi usada durante o desenvolvimento para evitar conflito local com outro serviço.

---

## 19. OpenCV headless

O container de API não necessita de interface gráfica.

Por isso utiliza:

```text
opencv-python-headless==4.12.0.88
```

O Dockerfile também remove a variante desktop de OpenCV eventualmente instalada como dependência transitiva e valida o import do `cv2` durante o build.

Essa decisão evita carregar dependências gráficas desnecessárias no runtime do container.

---

## 20. Modelo dentro da imagem Docker

O Dockerfile carrega o YOLO durante o build:

```dockerfile
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('YOLO OK')"
```

O objetivo é evitar que o primeiro start em produção dependa de download do modelo.

---

## 21. Docker Compose

O projeto inclui `docker-compose.yml` com:

```text
build pelo Dockerfile
restart: always
porta configurável
healthcheck
init
```

Validação:

```powershell
docker compose config
```

Subida padrão:

```powershell
docker compose up --build -d
```

Health:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Logs:

```powershell
docker compose logs -f
```

Encerramento:

```powershell
docker compose down
```

Se a porta `8000` estiver ocupada:

```powershell
$env:APP_PORT=8001
docker compose up --build -d
```

Nesse caso:

```text
host:8001 → container:8000
```

O Compose não fixa `platform`, permitindo que a execução use a arquitetura nativa da imagem disponível no host.

---

## 22. Build ARM64

O projeto foi validado para build ARM64 com Buildx:

```powershell
docker buildx build `
  --platform linux/arm64 `
  --tag person-detected:arm64 `
  --load `
  .
```

Validação:

```powershell
docker image inspect `
  person-detected:arm64 `
  --format '{{.Os}}/{{.Architecture}}'
```

Resultado validado:

```text
linux/arm64
```

A imagem ARM64 também foi inicializada em emulação no Docker Desktop e respondeu ao endpoint `/health`.

Essa evidência demonstra compatibilidade funcional de build/runtime ARM64, não desempenho real no Raspberry Pi 5.

---

## 23. Build multi-arquitetura

Validação conjunta:

```powershell
docker buildx build `
  --platform linux/amd64,linux/arm64 `
  --tag person-detected:multiarch `
  --output=type=cacheonly `
  .
```

Para publicação futura em registry:

```powershell
docker buildx build `
  --platform linux/amd64,linux/arm64 `
  --tag USUARIO/person-detected:latest `
  --push `
  .
```

Nesse cenário, o registry publica um manifesto com as variantes:

```text
linux/amd64
linux/arm64
```

A publicação em registry não é necessária para a validação local já executada.

---

## 24. Raspberry Pi 5

O Raspberry Pi 5 é o alvo Edge considerado para implantação.

O projeto já possui evidência de:

```text
build ARM64
execução ARM64 emulada
health check em ARM64
build multiarch
```

Mas:

```text
compatibilidade ARM64
≠
benchmark real no Raspberry Pi 5
```

A validação correta na placa real deve medir:

```text
latência mediana
P95
FPS efetivo do fluxo completo
uso de CPU
uso de memória
temperatura
throttling
estabilidade prolongada
```

---

## 25. Estratégias de otimização Edge

O modelo PyTorch é adequado como baseline funcional, mas pode não ser o runtime final ideal para Raspberry Pi.

Possíveis experimentos posteriores:

```text
redução controlada da resolução de entrada
NCNN
ONNX
TFLite
OpenVINO
FP16
INT8
NPU/acelerador compatível
redução da frequência de renderização PNG
```

Nenhuma dessas alternativas deve ser declarada como superior sem benchmark no hardware alvo.

---

## 26. Benchmark — ambiente utilizado

Benchmark medido em:

```text
CPU: AMD Ryzen 9 5900X
Cores: 12
Threads lógicas: 24
CUDA: não disponível
Execução: CPU
Imagem: 880 x 587
Warm-up: 5 execuções
Medições: 30 execuções
```

### Detector YOLO

| Métrica                  | Resultado |
| ------------------------ | --------: |
| Mínimo                   | 34.432 ms |
| Média                    | 39.793 ms |
| Mediana                  | 39.639 ms |
| P95                      | 44.716 ms |
| Máximo                   | 50.821 ms |
| Desvio padrão            |  2.971 ms |
| FPS teórico pela mediana |     25.23 |

### YOLO + classificação de risco

| Métrica                  | Resultado |
| ------------------------ | --------: |
| Mínimo                   | 36.252 ms |
| Média                    | 40.915 ms |
| Mediana                  | 40.039 ms |
| P95                      | 47.418 ms |
| Máximo                   | 51.364 ms |
| Desvio padrão            |  3.518 ms |
| FPS teórico pela mediana |     24.98 |

Diferença entre medianas:

```text
40.039 - 39.639 ≈ 0.4 ms
```

Neste experimento, a regra geométrica acrescentou custo pequeno quando comparada à inferência do modelo.

---

## 27. Interpretação do FPS

O valor aproximado de:

```text
25 inferências/s
```

é **teórico**, derivado da mediana da etapa medida.

Ele não representa automaticamente FPS real de câmera porque o benchmark do núcleo não inclui integralmente:

```text
captura
rede
armazenamento
pipeline de vídeo
renderização
outros processos concorrentes
```

A forma correta de interpretar o resultado é:

> O núcleo de inferência e classificação apresentou capacidade teórica próxima de 25 execuções por segundo no ambiente medido.

---

## 28. Benchmark HTTP E2E

O benchmark E2E mede a latência percebida pelo cliente HTTP.

### JSON — `POST /api/v1/infer`

| Métrica                    | Resultado |
| -------------------------- | --------: |
| Mínimo                     | 49.259 ms |
| Média                      | 59.336 ms |
| Mediana                    | 53.650 ms |
| P95                        | 76.196 ms |
| Máximo                     | 78.403 ms |
| Desvio padrão              |  9.815 ms |
| Req/s teórico pela mediana |     18.64 |

### PNG — `POST /api/v1/infer/annotated`

| Métrica                    |  Resultado |
| -------------------------- | ---------: |
| Mínimo                     |  67.296 ms |
| Média                      |  85.160 ms |
| Mediana                    |  86.470 ms |
| P95                        |  98.900 ms |
| Máximo                     | 100.889 ms |
| Desvio padrão              |   9.053 ms |
| Req/s teórico pela mediana |      11.56 |

---

## 29. Inferência versus latência E2E

Os números medem coisas diferentes.

### Núcleo

```text
imagem em memória
→ YOLO
→ classificação espacial
```

Mediana:

```text
40.039 ms
```

### HTTP JSON

```text
HTTP
→ multipart
→ decode
→ YOLO
→ zonas
→ alerta
→ serialização JSON
→ resposta HTTP
```

Mediana medida antes da inclusão da camada explícita de alerta:

```text
53.650 ms
```

### HTTP PNG

```text
HTTP
→ multipart
→ decode
→ YOLO
→ zonas
→ alerta
→ desenho
→ encode PNG
→ resposta HTTP
```

Mediana medida antes da inclusão da camada explícita de alerta:

```text
86.470 ms
```

A política de alerta adicionada posteriormente é determinística e de baixo custo, mas os benchmarks E2E devem ser repetidos na versão final antes de declarar números atualizados.

---

## 30. P95 e requisições por segundo

O P95 representa aproximadamente o valor abaixo do qual ficaram 95% das medições observadas.

Exemplo do benchmark JSON:

```text
P95 = 76.196 ms
```

Os valores de `req/s` são calculados teoricamente a partir da mediana:

```text
1000 / mediana_ms
```

Eles representam execução sequencial e **não equivalem a usuários simultâneos ou capacidade de carga concorrente**.

Não foram realizados testes de:

```text
concorrência
stress
carga sustentada
escalabilidade horizontal
```

---

## 31. Discussão de latência aceitável

O teste técnico pede discussão de viabilidade e latência, mas não fixa um limite numérico universal para este caso.

Para o protótipo, pode-se adotar como **meta inicial de engenharia**, a ser validada com o responsável pelo processo industrial:

```text
P95 do pipeline de alerta ≤ 200 ms
```

Esse valor é apenas um critério de projeto para avaliar resposta próxima de tempo real no protótipo.

**Não é um requisito normativo de NR-12 ou ISO 13849 e não representa tempo seguro de parada de máquina.**

Os benchmarks medidos no ambiente local ficaram abaixo dessa referência, porém esses resultados não podem ser extrapolados diretamente para Raspberry Pi 5.

Em hardware real, a meta deve ser reavaliada considerando:

```text
velocidade de aproximação
distância até a zona perigosa
latência de câmera
latência de inferência
comunicação
atuadores
tempo de parada da máquina
margem de segurança
```

Para uma função de segurança real, a latência admissível deve resultar da análise de risco e do projeto funcional completo.

---

## 32. Testes automatizados

Execute:

```powershell
python -m pytest -v
```

A suíte contém testes para:

```text
health check
contrato JSON
retorno PNG válido
imagem inválida
SEGURO / ALERTA / CRÍTICO
prioridade da zona vermelha
mapeamento SEGURO → NONE
mapeamento ALERTA → WARNING
mapeamento CRÍTICO → CRITICAL
seleção da maior severidade
```

Com os arquivos desta versão, o resultado esperado é:

```text
10 passed
```

Antes da entrega final, esse resultado deve ser confirmado no ambiente de validação e somente então tratado como evidência executada.

---

## 33. Estratégia de testes

### Motor de zonas

É determinístico.

Pontos conhecidos:

```text
(643, 583) → SEGURO
(443, 470) → ALERTA
(443, 555) → CRÍTICO
```

### Política de alertas

Também é determinística:

```text
SEGURO  → NONE
ALERTA  → WARNING
CRÍTICO → CRITICAL
```

### API

Os testes automatizados substituem a inferência real por resultados controlados para validar:

```text
contrato HTTP
serialização
PNG
alerta agregado
tratamento de erros
```

### IA

A inferência real é validada separadamente por:

```text
execução E2E
imagem de demonstração
Docker
benchmarks
```

Isso evita transformar toda a suíte em testes caros e dependentes de hardware/modelo.

---

## 34. Verificação E2E da Parte 1

O script:

```text
scripts/verify_part1.ps1
```

organiza a validação final em um único fluxo.

Execução completa:

```powershell
.\scripts\verify_part1.ps1
```

Por padrão ele utiliza a porta `8001` no host.

Etapas:

```text
1. pytest
2. docker compose config
3. docker compose up --build
4. GET /health
5. POST /api/v1/infer
6. POST /api/v1/infer/annotated
7. build multiarch amd64 + arm64
```

O script também grava evidências locais:

```text
outputs/evidence-infer.json
outputs/evidence-annotated.png
```

Para manter o serviço ativo ao final:

```powershell
.\scripts\verify_part1.ps1 -KeepRunning
```

Para iteração rápida, pulando somente a repetição do build multiarch:

```powershell
.\scripts\verify_part1.ps1 -SkipMultiArch
```

A validação final da entrega deve ser executada **sem** `-SkipMultiArch`.

---

## 35. Evidência dos endpoints

A comprovação pode ser feita com o serviço em execução local ou em outro host acessível.

### Health

```powershell
curl.exe http://127.0.0.1:8001/health
```

### JSON

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8001/api/v1/infer" `
  -F "file=@samples/input.jpg"
```

### PNG

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8001/api/v1/infer/annotated" `
  -F "file=@samples/input.jpg" `
  --output outputs/evidence-annotated.png
```

Swagger:

```text
http://127.0.0.1:8001/docs
```

Os endpoints são HTTP reais. Os exemplos usam `localhost` para a demonstração; publicação aberta na Internet não é necessária para provar o funcionamento do serviço.

---

## 36. Limitações atuais

Este projeto é um protótipo técnico.

Limitações relevantes:

```text
oclusão parcial ou total
iluminação variável
baixa resolução
motion blur
fumaça ou névoa
posição e ângulo da câmera
distância
perspectiva
falsos positivos
falsos negativos
mudanças no ambiente
objetos bloqueando o campo de visão
```

Não há tracking temporal nesta versão.

Cada imagem/frame é analisado de forma independente.

---

## 37. Melhorias futuras

Possíveis evoluções:

```text
tracking de pessoas
persistência temporal do risco
histerese de alertas
múltiplas câmeras
múltiplas zonas
configuração remota
MQTT
telemetria
armazenamento de evidências
dashboard
fine-tuning
exportação para runtime otimizado
quantização
NPU/acelerador
benchmark real no Raspberry Pi 5
```

---

## 38. Segurança funcional

Este projeto demonstra **monitoramento visual, classificação de risco e geração de alertas**.

Ele não deve ser tratado como substituto direto de funções ou dispositivos de segurança certificados.

Em aplicação industrial real, requisitos de proteção de máquinas e segurança funcional precisam ser tratados no contexto normativo aplicável, incluindo NR-12 e ISO 13849 quando pertinentes.

A visão computacional pode atuar como camada adicional de:

```text
percepção
monitoramento
evidência
alerta
```

Uma implementação destinada a exercer função de segurança exige engenharia, validação e certificação apropriadas.

---

## 39. Decisão técnica central

A solução pode ser resumida em:

```text
YOLO
→ percebe a pessoa

foot_point
→ aproxima sua posição no piso

polígono
→ representa a zona operacional

regra determinística
→ classifica o risco

política determinística
→ define alerta proporcional
```

A ideia central é evitar usar Machine Learning onde uma regra simples, explicável e testável resolve melhor o problema.

---

## 40. Estado da Parte 1

A implementação desta versão contempla:

```text
detecção de pessoas
bounding boxes
confidence
foot_point
zona amarela
zona vermelha
SEGURO / ALERTA / CRÍTICO
alerta proporcional
calibração das zonas
imagem anotada
API JSON
API PNG
health check
Dockerfile
docker-compose.yml
restart: always
build AMD64
build ARM64
validação multiarch com Buildx
benchmark do núcleo
benchmark HTTP E2E
testes automatizados
script de validação E2E
```

Executar a validação final:

````powershell
python -m pytest -v
.\scripts\verify_part1.ps1


```powershell
.\scripts\verify_part1.ps1

## PowerShell é só política de execução:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python
````

---

## 41. Autor

**Fabiano Barros**

Projeto desenvolvido para avaliação técnica envolvendo Edge AI, visão computacional, APIs, Docker, multi-arquitetura e análise de implantação embarcada.

Agosto de 2026.
