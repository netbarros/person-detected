"""
Testes determinísticos da política de alertas.

Estes testes não executam YOLO, HTTP nem Docker.
O objetivo é provar a regra de negócio exigida pelo projeto selecionado:
alertas proporcionais ao risco detectado.
"""

from app.alerts import (
    AlertAction,
    AlertLevel,
    decide_alert,
    select_highest_risk,
)


def test_safe_risk_does_not_activate_alert() -> None:
    """SEGURO não deve gerar alerta operacional."""

    decision = decide_alert("SEGURO")

    assert decision.active is False
    assert decision.level == AlertLevel.NONE
    assert decision.action == AlertAction.NONE


def test_warning_risk_generates_preventive_alert() -> None:
    """ALERTA deve gerar resposta preventiva proporcional."""

    decision = decide_alert("ALERTA")

    assert decision.active is True
    assert decision.level == AlertLevel.WARNING
    assert decision.action == AlertAction.WARN_OPERATOR


def test_critical_risk_generates_critical_alert() -> None:
    """CRÍTICO deve produzir a maior severidade disponível."""

    decision = decide_alert("CRÍTICO")

    assert decision.active is True
    assert decision.level == AlertLevel.CRITICAL
    assert (
        decision.action
        == AlertAction.REQUEST_IMMEDIATE_INTERVENTION
    )


def test_highest_risk_is_independent_of_detection_order() -> None:
    """
    O estado global do frame deve refletir a pessoa em maior risco,
    independentemente da ordem das detecções retornadas pelo YOLO.
    """

    assert (
        select_highest_risk(
            ["ALERTA", "SEGURO", "CRÍTICO"]
        )
        == "CRÍTICO"
    )

    # Sem pessoas detectadas não existe ocorrência humana a alertar.
    assert select_highest_risk([]) == "SEGURO"
