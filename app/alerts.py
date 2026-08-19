"""
Política e despacho de alertas proporcionais ao risco.

Por que este módulo existe?
---------------------------
O projeto selecionado não pede apenas a classificação da pessoa em
SEGURO / ALERTA / CRÍTICO. A matriz também pede que o sistema acione
alertas proporcionais ao nível de perigo.

Para manter a solução didática e segura, separamos duas responsabilidades:

1. DECISÃO DO ALERTA
   Converte o risco espacial em um nível e uma ação lógica.

2. DESPACHO DO ALERTA
   Executa a resposta automática disponível neste protótipo.
   Aqui a resposta é registrada em log, o que permite demonstrar o
   comportamento sem fingir que existe um relé, GPIO ou intertravamento
   certificado conectado à máquina.

Em uma implantação real, AlertDispatcher é o ponto de extensão para um
adaptador de GPIO, sinalizador visual/sonoro, MQTT, CLP etc.

IMPORTANTE
----------
Este protótipo NÃO implementa uma função de segurança certificada e não
deve ser usado para comandar diretamente uma parada de máquina sem a
engenharia, análise de risco e certificação aplicáveis ao contexto.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


# Logger próprio do domínio de alertas.
#
# O Uvicorn já configura logging para a aplicação; por isso basta usar
# o mecanismo padrão do Python. Isso mantém o componente desacoplado de
# FastAPI, Docker ou qualquer hardware específico.
logger = logging.getLogger("person-detected.alerts")


class AlertLevel(str, Enum):
    """
    Severidade do alerta emitido pelo sistema.

    Os valores são ASCII de propósito: além de serem legíveis, podem ser
    usados sem problemas em headers HTTP, logs, integrações e telemetria.
    """

    NONE = "NONE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertAction(str, Enum):
    """
    Ação lógica associada ao nível de risco.

    Repare que não usamos nomes como STOP_MACHINE. Isso seria uma
    afirmação indevida para um protótipo sem função de segurança
    certificada.

    WARN_OPERATOR
        sinaliza aproximação da zona de risco.

    REQUEST_IMMEDIATE_INTERVENTION
        sinaliza uma ocorrência crítica e solicita intervenção imediata
        do sistema/operador responsável.
    """

    NONE = "NONE"
    WARN_OPERATOR = "WARN_OPERATOR"
    REQUEST_IMMEDIATE_INTERVENTION = "REQUEST_IMMEDIATE_INTERVENTION"


@dataclass(frozen=True)
class AlertDecision:
    """
    Resultado da política de alerta.

    source_risk
        risco espacial que originou a decisão.

    active
        indica se existe um alerta a ser despachado.

    level
        severidade do alerta.

    action
        ação lógica que um adaptador externo poderia executar.

    message
        descrição legível para operador, log ou interface.
    """

    source_risk: str
    active: bool
    level: AlertLevel
    action: AlertAction
    message: str


# Prioridade explícita do domínio.
#
# Quando existem várias pessoas no mesmo frame, o alerta global deve
# refletir o maior risco observado, e não a ordem em que as detecções
# vieram do modelo.
_RISK_PRIORITY = {
    "SEGURO": 0,
    "ALERTA": 1,
    "CRÍTICO": 2,
}


def select_highest_risk(risks: Iterable[str]) -> str:
    """
    Retorna o maior nível de risco observado.

    Exemplo:

        ["SEGURO", "ALERTA", "CRÍTICO"]
            -> "CRÍTICO"

    Se não houver pessoas detectadas, assumimos SEGURO para o estado
    agregado do frame, pois não existe ocorrência humana a sinalizar.
    """

    risks_list = list(risks)

    if not risks_list:
        return "SEGURO"

    unknown = [risk for risk in risks_list if risk not in _RISK_PRIORITY]

    if unknown:
        raise ValueError(
            "Risco desconhecido para política de alerta: "
            + ", ".join(sorted(set(unknown)))
        )

    return max(
        risks_list,
        key=lambda risk: _RISK_PRIORITY[risk],
    )


def decide_alert(risk: str) -> AlertDecision:
    """
    Converte um risco espacial em uma resposta proporcional.

    Mapeamento utilizado no protótipo:

        SEGURO
            -> nenhum alerta

        ALERTA
            -> WARNING
            -> WARN_OPERATOR

        CRÍTICO
            -> CRITICAL
            -> REQUEST_IMMEDIATE_INTERVENTION

    A política está isolada em uma função pura para ser simples de
    testar e fácil de substituir por regras mais sofisticadas no futuro.
    """

    if risk == "SEGURO":
        return AlertDecision(
            source_risk=risk,
            active=False,
            level=AlertLevel.NONE,
            action=AlertAction.NONE,
            message="Nenhum alerta ativo.",
        )

    if risk == "ALERTA":
        return AlertDecision(
            source_risk=risk,
            active=True,
            level=AlertLevel.WARNING,
            action=AlertAction.WARN_OPERATOR,
            message=(
                "Pessoa detectada na zona amarela: "
                "emitir alerta preventivo ao operador."
            ),
        )

    if risk == "CRÍTICO":
        return AlertDecision(
            source_risk=risk,
            active=True,
            level=AlertLevel.CRITICAL,
            action=AlertAction.REQUEST_IMMEDIATE_INTERVENTION,
            message=(
                "Pessoa detectada na zona vermelha: "
                "solicitar intervenção imediata do sistema responsável."
            ),
        )

    raise ValueError(
        f"Risco desconhecido para política de alerta: {risk}"
    )


class AlertDispatcher:
    """
    Executa a resposta automática disponível neste protótipo.

    Nesta versão, o despacho é feito via log. Isso é deliberado:

        - funciona em notebook, VM, Docker e Raspberry Pi;
        - é demonstrável sem hardware externo;
        - não simula uma saída física que não existe;
        - mantém um ponto claro de extensão para integração futura.

    Um projeto de produção poderia trocar/estender esta classe por um
    adapter GPIO, MQTT, CLP ou sinalizador físico, mantendo a política de
    risco independente da tecnologia de saída.
    """

    def dispatch(self, decision: AlertDecision) -> None:
        """
        Despacha o alerta quando ele estiver ativo.

        WARNING é registrado como warning e CRITICAL como error para que
        a severidade também seja visível na infraestrutura de logs.
        """

        if not decision.active:
            return

        log_function = (
            logger.error
            if decision.level == AlertLevel.CRITICAL
            else logger.warning
        )

        log_function(
            "automatic_alert level=%s source_risk=%s action=%s message=%s",
            decision.level.value,
            decision.source_risk,
            decision.action.value,
            decision.message,
        )
