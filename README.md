# Person Detected — Edge AI Risk Zone Monitor

Projeto de Edge AI para detecção de pessoas e classificação de risco em zonas próximas a máquinas e áreas industriais.

A solução combina visão computacional com regras geométricas determinísticas:

```text
imagem
  ↓
YOLO
  ↓
detecção de pessoa
  ↓
bounding box
  ↓
foot_point
  ↓
classificação espacial
  ↓
SEGURO / ALERTA / CRÍTICO
```

O objetivo é demonstrar uma arquitetura simples, explicável e compatível com execução em dispositivos Edge, com Raspberry Pi 5 como alvo de implantação.

---

## 1. Problema

Em ambientes industriais, uma pessoa pode se aproximar de regiões onde existem máquinas, equipamentos móveis ou outros elementos que representam risco operacional.

A proposta deste projeto é monitorar visualmente essas regiões utilizando uma câmera e identificar quando uma pessoa entra em zonas previamente configuradas.

O sistema trabalha com três estados:

| Estado    | Significado                    |
| --------- | ------------------------------ |
| `SEGURO`  | Pessoa fora das zonas de risco |
| `ALERTA`  | Pessoa dentro da zona amarela  |
| `CRÍTICO` | Pessoa dentro da zona vermelha |

A zona vermelha possui prioridade sobre a zona amarela.

---

## 2. Princípio arquitetural

A solução separa duas responsabilidades.

### Percepção com IA

O modelo YOLO responde à pergunta:

> Onde existe uma pessoa na imagem?

O resultado principal é uma `bounding box`.

### Decisão espacial determinística

Depois da detecção, uma regra geométrica responde:

> A pessoa está dentro de qual zona?

Essa decisão não é delegada ao modelo de IA.

Em resumo:

```text
IA para percepção.
Regra determinística para decisão espacial.
```

Essa separação torna a solução mais simples de explicar, testar e manter.

---

## 3. Arquitetura

```text
                   +----------------------+
                   |       Imagem         |
                   +----------+-----------+
                              |
                              v
                   +----------------------+
                   |     YOLO11 Nano      |
                   | detecção de pessoas  |
                   +----------+-----------+
                              |
                              v
                   +----------------------+
                   |    Bounding Box      |
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
                +-------------+-------------+
                |             |             |
                v             v             v
             SEGURO        ALERTA        CRÍTICO
                |             |             |
                +-------------+-------------+
                              |
                 +------------+------------+
                 |                         |
                 v                         v
             JSON API                 PNG anotado
```

---

## 4. Por que YOLO11n

O projeto utiliza:

```text
yolo11n.pt
```

A variante Nano foi escolhida por oferecer um compromisso adequado entre:

- tamanho do modelo;
- custo computacional;
- latência;
- simplicidade de uso;
- possibilidade de implantação em Edge.

O projeto utiliza um modelo pré-treinado e filtra somente a classe `person`.

Não foi realizado treinamento customizado neste estágio.

Para um cenário industrial real, um processo posterior pode incluir fine-tuning com imagens representativas do ambiente específico.

---

## 5. Bounding box e foot point

O detector retorna uma caixa delimitadora:

```text
(x1, y1)
   +----------------+
   |                |
   |     pessoa     |
   |                |
   +----------------+
                  (x2, y2)
```

Para determinar a posição da pessoa em relação ao piso, o projeto não utiliza o centro da bounding box.

É utilizado o ponto central inferior:

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

Essa estratégia aproxima o ponto de contato da pessoa com o chão e é mais adequada para zonas definidas sobre o piso.

---

## 6. Zonas de risco

As zonas são armazenadas em:

```text
config/zones.json
```

As coordenadas são normalizadas entre `0` e `1`.

Isso permite utilizar a mesma configuração em imagens de diferentes resoluções.

O sistema possui:

```text
yellow
→ ALERTA

red
→ CRÍTICO
```

A classificação é realizada com:

```python
cv2.pointPolygonTest(...)
```

A prioridade é:

```text
zona vermelha
    ↓
CRÍTICO

senão, zona amarela
    ↓
ALERTA

senão
    ↓
SEGURO
```

---

## 7. Estrutura do projeto

```text
person-detected/
│
├── app/
│   ├── __init__.py
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
│   └── test_zones.py
│
├── tests/
│   ├── __init__.py
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
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 8. Componentes principais

### `app/detector.py`

Encapsula o modelo YOLO.

Responsabilidades:

```text
carregar modelo
→ executar inferência
→ filtrar person
→ gerar PersonDetection
→ fornecer bbox e foot_point
```

### `app/zones.py`

Implementa a decisão espacial.

Responsabilidades:

```text
carregar zones.json
→ converter coordenadas normalizadas
→ construir polígonos
→ testar foot_point
→ retornar risco
```

### `app/visualizer.py`

Responsável pela apresentação visual.

Desenha:

```text
zonas
bounding boxes
foot points
nível de risco
confiança
```

### `app/api.py`

Expõe o pipeline através de HTTP usando FastAPI.

### `scripts/`

Contém ferramentas de calibração, demonstração e benchmark.

### `tests/`

Contém testes automatizados determinísticos da API e do motor de zonas.

---

## 9. Requisitos

O container de produção utiliza Python 3.11.

O ambiente local utilizado durante parte dos testes utilizou Python 3.13.5.

As principais dependências são:

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

---

## 10. Instalação local

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

## 11. Execução local

Para executar a detecção diretamente sobre a imagem:

```powershell
python -m scripts.detect_person samples/input.jpg
```

A saída é gerada em:

```text
outputs/person-detected.png
```

O terminal também apresenta informações como:

```text
confiança
bounding box
foot_point
risco
```

---

## 12. Calibração das zonas

A ferramenta de calibração permite definir os polígonos visualmente.

Zona vermelha:

```powershell
python -m scripts.mark_zones red
```

Zona amarela:

```powershell
python -m scripts.mark_zones yellow
```

Durante a calibração:

```text
clique esquerdo
→ adiciona ponto

U
→ desfaz último ponto

C
→ limpa pontos

S
→ salva

ESC
→ encerra
```

As coordenadas são armazenadas de forma normalizada em:

```text
config/zones.json
```

---

## 13. API HTTP

Suba a API localmente:

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Documentação Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## 14. Health check

Endpoint:

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

---

## 15. Inferência com resposta JSON

Endpoint:

```http
POST /api/v1/infer
```

Exemplo:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/api/v1/infer" `
  -F "file=@samples/input.jpg"
```

Exemplo de resposta:

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
  ]
}
```

O valor de `inference_ms` varia entre execuções e não deve ser tratado como constante.

---

## 16. Inferência com imagem anotada

Endpoint:

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

O endpoint retorna:

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
confiança
```

Na renderização OpenCV o texto `CRÍTICO` é exibido como:

```text
CRITICO
```

porque as fontes Hershey utilizadas por `cv2.putText()` não possuem suporte Unicode completo.

O valor original continua sendo mantido no domínio e no JSON:

```json
"risk": "CRÍTICO"
```

---

## 17. Docker

Build local:

```powershell
docker build -t person-detected:local .
```

Execução:

```powershell
docker run --rm `
  -p 8001:8000 `
  --name person-detected `
  person-detected:local
```

A porta `8001` é utilizada no host apenas para evitar possíveis conflitos locais.

O serviço continua executando na porta `8000` dentro do container.

Teste:

```powershell
curl.exe http://127.0.0.1:8001/health
```

---

## 18. OpenCV em ambiente headless

O container não necessita de interface gráfica.

Por isso utiliza:

```text
opencv-python-headless
```

em vez da variante desktop do OpenCV.

Isso evita dependências gráficas desnecessárias no runtime da API.

A versão utilizada no container foi fixada em:

```text
opencv-python-headless==4.12.0.88
```

O processo de build também valida explicitamente que o módulo `cv2` pode ser importado antes de carregar o YOLO.

---

## 19. Modelo dentro da imagem Docker

O modelo YOLO é carregado durante o build da imagem:

```dockerfile
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

O objetivo é evitar que o dispositivo Edge precise baixar os pesos no primeiro start da aplicação.

Isso favorece execução em ambientes com conectividade limitada ou controlada.

---

## 20. Multiarquitetura

O projeto foi validado para build:

```text
linux/amd64
linux/arm64
```

A arquitetura ARM64 é relevante para dispositivos como Raspberry Pi 5 executando sistema operacional 64 bits.

Build ARM64:

```powershell
docker buildx build `
  --platform linux/arm64 `
  --tag person-detected:arm64 `
  --load `
  .
```

Validação da arquitetura:

```powershell
docker image inspect `
  person-detected:arm64 `
  --format '{{.Os}}/{{.Architecture}}'
```

Resultado esperado e validado:

```text
linux/arm64
```

A imagem ARM64 também foi inicializada e teve o endpoint `/health` validado utilizando emulação no Docker Desktop.

---

## 21. Build multiarch

Validação simultânea das duas arquiteturas:

```powershell
docker buildx build `
  --platform linux/amd64,linux/arm64 `
  --tag person-detected:multiarch `
  --output=type=cacheonly `
  .
```

Para publicação futura em um registry:

```powershell
docker buildx build `
  --platform linux/amd64,linux/arm64 `
  --tag USUARIO/person-detected:latest `
  --push `
  .
```

Nesse cenário, o registry disponibiliza um manifesto com as duas arquiteturas.

Um host AMD64 pode selecionar a imagem AMD64 e um host ARM64 pode selecionar a variante ARM64.

A publicação multiarch em registry não faz parte da validação realizada até este estágio.

---

## 22. Raspberry Pi 5

O Raspberry Pi 5 é o alvo de implantação Edge considerado para a solução.

O projeto já possui evidência de compatibilidade de build ARM64.

Entretanto:

```text
compatibilidade ARM64
≠
benchmark real no Raspberry Pi 5
```

Os resultados de performance apresentados neste projeto foram medidos em outro hardware e não devem ser extrapolados diretamente para o Raspberry Pi.

A próxima etapa em hardware real deve medir:

```text
latência
FPS
uso de CPU
uso de memória
temperatura
consumo
estabilidade prolongada
```

---

## 23. Estratégias de otimização Edge

O modelo PyTorch utilizado no ambiente de desenvolvimento é adequado para validação funcional, mas não representa necessariamente o formato ideal para implantação final em um Raspberry Pi.

Possíveis etapas posteriores incluem avaliação de:

```text
NCNN
ONNX
TFLite
OpenVINO
FP16
INT8
aceleradores/NPU compatíveis
```

A escolha deve ser feita com benchmark no hardware alvo.

Não se deve assumir que uma otimização será melhor sem medição.

---

## 24. Benchmark do núcleo de inferência

Ambiente medido:

```text
CPU: AMD Ryzen 9 5900X
Cores: 12
Threads lógicas: 24
CUDA: não disponível
Execução do modelo: CPU
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

A diferença entre as medianas foi aproximadamente:

```text
40.039 - 39.639
≈ 0.4 ms
```

Isso indica que, neste experimento, a regra espacial acrescentou custo pequeno quando comparada à inferência do modelo.

---

## 25. Interpretação do FPS

O valor aproximado de:

```text
25 inferências/s
```

é um valor teórico calculado a partir da mediana do trecho medido.

Ele não representa automaticamente FPS real de câmera.

O benchmark não inclui integralmente elementos como:

```text
captura da câmera
rede
armazenamento
pipeline de vídeo
renderização da interface
outros processos concorrentes
```

Portanto a forma correta de interpretar o resultado é:

> O núcleo de inferência e classificação apresentou capacidade teórica próxima de 25 execuções por segundo no ambiente medido.

---

## 26. Benchmark HTTP E2E

Também foi medida a latência percebida por um cliente HTTP.

O benchmark foi executado contra a API em container Docker local utilizando `localhost`.

### Endpoint JSON

```text
POST /api/v1/infer
```

| Métrica                    | Resultado |
| -------------------------- | --------: |
| Mínimo                     | 49.259 ms |
| Média                      | 59.336 ms |
| Mediana                    | 53.650 ms |
| P95                        | 76.196 ms |
| Máximo                     | 78.403 ms |
| Desvio padrão              |  9.815 ms |
| Req/s teórico pela mediana |     18.64 |

### Endpoint PNG

```text
POST /api/v1/infer/annotated
```

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

## 27. Inferência versus latência E2E

Os benchmarks medem conceitos diferentes.

### Núcleo

```text
imagem já em memória
→ YOLO
→ classificação espacial
```

Mediana:

```text
40.039 ms
```

### API JSON

```text
HTTP
→ multipart
→ decode
→ YOLO
→ classificação
→ JSON
→ HTTP response
```

Mediana:

```text
53.650 ms
```

### API PNG

```text
HTTP
→ multipart
→ decode
→ YOLO
→ classificação
→ desenho
→ encode PNG
→ HTTP response
```

Mediana:

```text
86.470 ms
```

Isso demonstra por que tempo de inferência não deve ser tratado como sinônimo de latência total do sistema.

---

## 28. Sobre P95

Além da média e da mediana, foi calculado o percentil 95.

Exemplo do endpoint JSON:

```text
P95 = 76.196 ms
```

Isso significa que aproximadamente 95% das execuções observadas ficaram até esse valor.

Para sistemas interativos ou operacionais, percentis ajudam a identificar o comportamento das requisições mais lentas e são frequentemente mais informativos do que apenas a média.

---

## 29. Requisições por segundo

Os valores de `req/s` mostrados pelo benchmark são calculados a partir da mediana:

```text
1000 / mediana_ms
```

Eles representam somente capacidade teórica de execução sequencial.

Não foram realizados testes de:

```text
concorrência
múltiplos usuários
stress
carga sustentada
escalabilidade horizontal
```

Portanto esses valores não devem ser apresentados como capacidade de usuários simultâneos.

---

## 30. Testes automatizados

A suíte utiliza `pytest`.

Execução:

```powershell
python -m pytest -v
```

Resultado validado:

```text
6 passed
```

Os testes cobrem:

| Área         | Validação                               |
| ------------ | --------------------------------------- |
| Health       | HTTP 200 e contrato básico              |
| API JSON     | estrutura e valores do contrato         |
| API PNG      | retorno `image/png` e PNG decodificável |
| API inválida | rejeição de conteúdo não decodificável  |
| Zonas        | SEGURO, ALERTA e CRÍTICO                |
| Prioridade   | zona vermelha sobre zona amarela        |

---

## 31. Estratégia de testes

A solução separa testes determinísticos da validação do modelo.

### Motor de zonas

É determinístico.

Para um ponto conhecido, o resultado esperado pode ser testado exatamente:

```text
(643, 583)
→ SEGURO

(443, 470)
→ ALERTA

(443, 555)
→ CRÍTICO
```

### API

Os testes automatizados substituem a inferência real por resultados conhecidos.

Isso permite validar rapidamente:

```text
contrato HTTP
serialização JSON
PNG
tratamento de erros
```

sem depender de performance, modelo ou hardware.

### IA

O comportamento real do detector foi validado separadamente através de:

```text
execução E2E
imagem de demonstração
Docker
benchmark
```

Essa separação evita transformar todos os testes em testes caros e não determinísticos de Machine Learning.

---

## 32. Limitações atuais

Este projeto é um protótipo técnico e possui limitações que precisam ser consideradas em um cenário industrial real.

Entre elas estão:

```text
oclusão parcial ou total da pessoa
iluminação variável
baixa resolução
motion blur
fumaça ou névoa
posição e ângulo da câmera
distância da pessoa
perspectiva
falsos positivos
falsos negativos
mudanças no ambiente
objetos bloqueando o campo de visão
```

Também não há tracking temporal neste estágio.

Cada frame é tratado de forma independente.

---

## 33. Melhorias futuras

Possíveis evoluções técnicas incluem:

```text
tracking de pessoas
persistência temporal do risco
histerese de alertas
múltiplas câmeras
múltiplas zonas
configuração remota
telemetria
MQTT
eventos
armazenamento de evidências
dashboard
exportação para formato otimizado
quantização
aceleração por NPU
benchmark real no Raspberry Pi 5
fine-tuning para cenário industrial
```

---

## 34. Métricas de visão computacional

A latência é apenas uma parte da avaliação.

Para validar um detector em um ambiente industrial real também devem ser consideradas métricas como:

```text
Precision
Recall
F1-score
IoU
mAP
matriz de confusão
```

Um sistema rápido, mas com detecção inadequada, não atende ao problema.

Particularmente em aplicações de segurança, falsos negativos merecem atenção especial.

---

## 35. Segurança funcional

Este projeto demonstra monitoramento visual e classificação de risco utilizando Edge AI.

Ele não deve ser tratado como substituto direto de dispositivos ou funções de segurança certificados.

Em um sistema industrial real, requisitos associados à proteção de máquinas e segurança funcional precisam ser analisados no contexto aplicável, incluindo normas e procedimentos relevantes, como NR-12 e ISO 13849 quando pertinentes.

A visão computacional pode atuar como uma camada adicional de percepção, monitoramento, evidência ou alerta.

Uma implementação destinada a exercer função de segurança exige engenharia, validação e certificação apropriadas ao contexto.

---

## 36. Decisão técnica central

A principal decisão arquitetural do projeto pode ser resumida em:

```text
YOLO
→ percebe a pessoa

foot_point
→ representa sua posição no piso

polígono
→ representa a zona operacional

regra determinística
→ decide o risco
```

Isso evita utilizar Machine Learning onde uma regra geométrica simples, explicável e testável resolve melhor o problema.

---

## 37. Estado atual

A implementação já contempla:

```text
detecção de pessoas
bounding boxes
foot_point
zona amarela
zona vermelha
SEGURO / ALERTA / CRÍTICO
calibração das zonas
imagem anotada
API JSON
API PNG
health check
Docker
build AMD64
build ARM64
validação multiarch com Buildx
benchmark do núcleo
benchmark HTTP E2E
testes automatizados
```

O principal trabalho restante para uma implantação real seria a validação no hardware Edge e no ambiente industrial alvo.

---

## 38. Autor

**Fabiano Barros**

Projeto desenvolvido como exercício técnico de arquitetura, Edge AI, visão computacional, APIs e implantação multiplataforma.

Agosto de 2026.
