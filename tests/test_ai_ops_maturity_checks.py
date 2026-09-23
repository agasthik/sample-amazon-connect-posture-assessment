"""Regression tests for evidence-bounded AI operations maturity checks."""

from botocore.exceptions import ClientError

from amazon_connect_assessment.aws_client_factory import AWSClientFactory
from amazon_connect_assessment.checks.ai_ops_maturity_checks import (
    AIGuardrailCoverageCheck,
    AIPromptModelCostCheck,
    BedrockInvocationLoggingCheck,
    CrossRegionInferenceAvailabilityCheck,
    KnowledgeBaseSyncHealthCheck,
    QConnectEncryptionCheck,
    register_ai_ops_maturity_checks,
)
from amazon_connect_assessment.checks.registration import register_all_checks
from amazon_connect_assessment.checks.registry import CheckRegistry
from amazon_connect_assessment.models import CheckStatus


def _wire_access_denied(factory):
    factory.is_access_denied = AWSClientFactory.is_access_denied
    # Default to an assistant with no bound AI agents so attachment evidence is
    # empty unless a test deliberately supplies it.
    factory.get_qconnect_assistant_resilient.return_value = {"assistant": {}}
    factory.list_ai_agents_resilient.return_value = {"aiAgentSummaries": []}
    # Default the individual agent read to "nothing returned", so a test that
    # binds an agent without wiring GetAIAgent lands on Skipped rather than on an
    # accidental Pass built out of mock attributes.
    factory.get_ai_agent_resilient.return_value = {"aiAgent": {}}


def _published_guardrail():
    return {
        "aiGuardrailSummaries": [
            {"name": "published", "status": "ACTIVE", "visibilityStatus": "PUBLISHED"}
        ]
    }


def _bind_agents(factory, **agents_by_type):
    """Bind agent types to agent IDs on the assistant, as GetAssistant reports them."""
    factory.get_qconnect_assistant_resilient.return_value = {
        "assistant": {
            "aiAgentConfiguration": {
                agent_type: {"aiAgentId": agent_id}
                for agent_type, agent_id in agents_by_type.items()
            }
        }
    }


def _agent_summary(agent_id, agent_type, config_key, guardrail_field=None, guardrail_id=None):
    variant = {}
    if guardrail_field and guardrail_id:
        variant[guardrail_field] = guardrail_id
    return {
        "aiAgentId": agent_id,
        "name": f"{agent_type.lower()}-agent",
        "type": agent_type,
        "origin": "CUSTOMER",
        "configuration": {config_key: variant},
    }


def _assistant_association(arn: str = "arn:aws:wisdom:us-east-1:123:assistant/a1"):
    return {
        "IntegrationAssociationSummaryList": [
            {"IntegrationType": "WISDOM_ASSISTANT", "IntegrationArn": arn}
        ]
    }


def _knowledge_base_association(
    arn: str = "arn:aws:wisdom:us-east-1:123:knowledge-base/kb1",
):
    return {
        "IntegrationAssociationSummaryList": [
            {"IntegrationType": "WISDOM_KNOWLEDGE_BASE", "IntegrationArn": arn}
        ]
    }


def _empty_association():
    return {"IntegrationAssociationSummaryList": []}


def _access_denied(operation: str) -> ClientError:
    return ClientError({"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, operation)


def test_ai_guardrail_no_assistant_returns_not_applicable(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    mock_aws_client_factory.list_integration_associations_resilient.return_value = (
        _empty_association()
    )

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.NOT_APPLICABLE
    assert finding.evidence["assistant_count"] == 0


def test_ai_guardrail_draft_only_does_not_pass(make_check_context, mock_aws_client_factory):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    mock_aws_client_factory.list_integration_associations_resilient.return_value = (
        _assistant_association()
    )
    mock_aws_client_factory.list_ai_guardrails_resilient.return_value = {
        "aiGuardrailSummaries": [{"name": "draft", "status": "ACTIVE", "visibilityStatus": "SAVED"}]
    }

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.FAIL
    observation = finding.evidence["assistants_without_active_published_guardrail"][0]
    assert observation["active_published_count"] == 0
    assert observation["observed_states"] == ["ACTIVE/SAVED"]


def test_ai_guardrail_active_published_passes_without_filter_claim(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    mock_aws_client_factory.list_integration_associations_resilient.return_value = (
        _assistant_association()
    )
    mock_aws_client_factory.list_ai_guardrails_resilient.return_value = _published_guardrail()

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS
    assert "does not establish which content" in finding.description.lower()
    assert "content filtering and pii handling are in place" not in finding.description.lower()


def test_ai_guardrail_bound_agent_without_guardrail_fails_despite_published_guardrail(
    make_check_context, mock_aws_client_factory
):
    # Arrange — the availability signal is clean; only attachment is broken.
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, SELF_SERVICE="agent-1")
    factory.list_ai_agents_resilient.return_value = {
        "aiAgentSummaries": [
            _agent_summary("agent-1", "SELF_SERVICE", "selfServiceAIAgentConfiguration")
        ]
    }

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.FAIL
    assert finding.evidence["unguarded_bound_agent_count"] == 1
    assert "assistants_without_active_published_guardrail" not in finding.evidence
    gap = finding.evidence["assistants_with_unguarded_bound_agents"][0]
    assert gap["unguarded_bound_agents"][0]["agent_type"] == "SELF_SERVICE"
    assert gap["unguarded_bound_agents"][0]["guardrail_id"] is None
    assert "agent-1" in finding.structured_remediation.steps[0].instruction


def test_ai_guardrail_bound_agent_with_guardrail_passes(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, ANSWER_RECOMMENDATION="agent-1")
    factory.list_ai_agents_resilient.return_value = {
        "aiAgentSummaries": [
            _agent_summary(
                "agent-1",
                "ANSWER_RECOMMENDATION",
                "answerRecommendationAIAgentConfiguration",
                "answerGenerationAIGuardrailId",
                "gr-1",
            )
        ]
    }

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS
    assert finding.evidence["unguarded_bound_agent_count"] == 0
    observation = finding.evidence["assistant_guardrail_observations"][0]
    assert observation["guarded_bound_agents"][0]["guardrail_id"] == "gr-1"


def test_ai_guardrail_email_agent_without_guardrail_is_excluded_not_failed(
    make_check_context, mock_aws_client_factory
):
    # Arrange — EMAIL_* configurations have no guardrail member to set.
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, EMAIL_RESPONSE="agent-1")
    factory.list_ai_agents_resilient.return_value = {
        "aiAgentSummaries": [
            _agent_summary("agent-1", "EMAIL_RESPONSE", "emailResponseAIAgentConfiguration")
        ]
    }

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS
    observation = finding.evidence["assistant_guardrail_observations"][0]
    assert observation["bound_agent_count"] == 1
    assert observation["guardrail_capable_bound_agents"] == 0


def test_ai_guardrail_unbound_agent_without_guardrail_does_not_fail(
    make_check_context, mock_aws_client_factory
):
    # Arrange — an agent that exists but is not bound cannot reach a caller.
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    factory.list_ai_agents_resilient.return_value = {
        "aiAgentSummaries": [
            _agent_summary("unbound-1", "SELF_SERVICE", "selfServiceAIAgentConfiguration")
        ]
    }

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS
    assert finding.evidence["unguarded_bound_agent_count"] == 0


def test_ai_guardrail_bound_agent_missing_from_list_is_resolved_individually(
    make_check_context, mock_aws_client_factory
):
    # Arrange — a version-pinned binding is absent from ListAIAgents, so the
    # agent is read directly instead of being left unverified.
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, ORCHESTRATION="agent-pinned:3")
    factory.list_ai_agents_resilient.return_value = {"aiAgentSummaries": []}
    factory.get_ai_agent_resilient.return_value = {
        "aiAgent": _agent_summary(
            "agent-pinned:3",
            "ORCHESTRATION",
            "orchestrationAIAgentConfiguration",
            guardrail_field="orchestrationAIGuardrailId",
            guardrail_id="g-1",
        )
    }

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS
    factory.get_ai_agent_resilient.assert_called_once_with("a1", "agent-pinned:3")
    observation = finding.evidence["assistant_guardrail_observations"][0]
    assert observation["guarded_bound_agents"][0]["resolved_by"] == "wisdom:GetAIAgent"
    assert "unresolved_bound_agents" not in observation
    assert finding.evidence["unguarded_bound_agent_count"] == 0


def test_ai_guardrail_resolved_agent_without_guardrail_fails(
    make_check_context, mock_aws_client_factory
):
    # Arrange — the individually-read agent has no guardrail. That is a real
    # attachment gap and must fail, not pass for want of a list entry.
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, SELF_SERVICE="agent-pinned:2")
    factory.list_ai_agents_resilient.return_value = {"aiAgentSummaries": []}
    factory.get_ai_agent_resilient.return_value = {
        "aiAgent": _agent_summary(
            "agent-pinned:2", "SELF_SERVICE", "selfServiceAIAgentConfiguration"
        )
    }

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.FAIL
    assert finding.evidence["unguarded_bound_agent_count"] == 1


def test_ai_guardrail_unresolvable_bound_agent_is_skipped_not_passed(
    make_check_context, mock_aws_client_factory
):
    # Arrange — neither ListAIAgents nor GetAIAgent can read the bound agent.
    # The claim this check makes is about *every* bound agent, so one agent whose
    # configuration was never read makes the claim undetermined. Passing here
    # would be a confident security PASS built on absent evidence.
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, ORCHESTRATION="agent-missing")
    factory.list_ai_agents_resilient.return_value = {"aiAgentSummaries": []}
    factory.get_ai_agent_resilient.side_effect = _access_denied("GetAIAgent")

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    assert finding.evidence["required_permission"] == "wisdom:GetAIAgent"
    observation = finding.evidence["assistants_with_unresolved_bound_agents"][0]
    assert observation["unresolved_bound_agents"] == [
        {"agent_type": "ORCHESTRATION", "ai_agent_id": "agent-missing", "reason": "access_denied"}
    ]
    # The unverified agent still counts as guardrail-capable, so the evidence
    # cannot be read as "no agents needed checking".
    assert observation["guardrail_capable_bound_agents"] == 1


def test_ai_guardrail_real_gap_outranks_an_unresolved_agent(
    make_check_context, mock_aws_client_factory
):
    # Arrange — one bound agent has no guardrail, another cannot be read. The
    # actionable failure is reported rather than masked behind Skipped.
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, SELF_SERVICE="agent-1", ORCHESTRATION="agent-missing")
    factory.list_ai_agents_resilient.return_value = {
        "aiAgentSummaries": [
            _agent_summary("agent-1", "SELF_SERVICE", "selfServiceAIAgentConfiguration")
        ]
    }
    factory.get_ai_agent_resilient.side_effect = _access_denied("GetAIAgent")

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.FAIL
    assert finding.evidence["unguarded_bound_agent_count"] == 1
    assert "assistants_with_unresolved_bound_agents" in finding.evidence


def test_ai_guardrail_get_assistant_access_denied_returns_skipped(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    factory.get_qconnect_assistant_resilient.side_effect = _access_denied("GetAssistant")

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    assert finding.evidence["required_permission"] == "wisdom:GetAssistant"


def test_ai_guardrail_list_agents_access_denied_returns_skipped_not_pass(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, SELF_SERVICE="agent-1")
    factory.list_ai_agents_resilient.side_effect = _access_denied("ListAIAgents")

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    assert finding.evidence["required_permission"] == "wisdom:ListAIAgents"


def test_ai_guardrail_agent_list_is_paginated_with_next_token(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.return_value = _published_guardrail()
    _bind_agents(factory, SELF_SERVICE="agent-page-2")
    factory.list_ai_agents_resilient.side_effect = [
        {"aiAgentSummaries": [], "nextToken": "agent-page-2"},
        {
            "aiAgentSummaries": [
                _agent_summary(
                    "agent-page-2",
                    "SELF_SERVICE",
                    "selfServiceAIAgentConfiguration",
                    "selfServiceAIGuardrailId",
                    "gr-2",
                )
            ]
        },
    ]

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert — the guardrail only becomes visible on page two.
    assert finding.status == CheckStatus.PASS
    assert factory.list_ai_agents_resilient.call_args_list[1].kwargs == {
        "maxResults": 100,
        "nextToken": "agent-page-2",
    }


def test_ai_guardrail_second_page_active_published_changes_result_and_uses_next_token(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_guardrails_resilient.side_effect = [
        {
            "aiGuardrailSummaries": [{"status": "CREATE_FAILED", "visibilityStatus": "SAVED"}],
            "nextToken": "guardrail-page-2",
        },
        {"aiGuardrailSummaries": [{"status": "ACTIVE", "visibilityStatus": "PUBLISHED"}]},
    ]

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS
    assert factory.list_ai_guardrails_resilient.call_args_list[0].kwargs == {"maxResults": 100}
    assert factory.list_ai_guardrails_resilient.call_args_list[1].kwargs == {
        "maxResults": 100,
        "nextToken": "guardrail-page-2",
    }


def test_ai_discovery_second_connect_page_changes_applicability_and_uses_next_token(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.side_effect = [
        {"IntegrationAssociationSummaryList": [], "NextToken": "association-page-2"},
        _assistant_association(),
    ]
    factory.list_ai_guardrails_resilient.return_value = {
        "aiGuardrailSummaries": [{"status": "ACTIVE", "visibilityStatus": "PUBLISHED"}]
    }

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS
    assert factory.list_integration_associations_resilient.call_args_list[0].kwargs == {
        "MaxResults": 100
    }
    assert factory.list_integration_associations_resilient.call_args_list[1].kwargs == {
        "MaxResults": 100,
        "NextToken": "association-page-2",
    }


def test_ai_discovery_partial_page_access_denied_returns_skipped_with_partial_evidence(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.side_effect = [
        {
            **_assistant_association(),
            "NextToken": "association-page-2",
        },
        _access_denied("ListIntegrationAssociations"),
    ]

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    assert finding.evidence["required_permission"] == "connect:ListIntegrationAssociations"
    assert finding.evidence["partial_item_count"] == 1
    factory.list_ai_guardrails_resilient.assert_not_called()


def test_ai_guardrail_partial_assistant_access_denied_returns_skipped_not_pass(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = {
        "IntegrationAssociationSummaryList": [
            {
                "IntegrationType": "WISDOM_ASSISTANT",
                "IntegrationArn": "arn:aws:wisdom:us-east-1:123:assistant/a1",
            },
            {
                "IntegrationType": "WISDOM_ASSISTANT",
                "IntegrationArn": "arn:aws:wisdom:us-east-1:123:assistant/a2",
            },
        ]
    }
    factory.list_ai_guardrails_resilient.side_effect = [
        {"aiGuardrailSummaries": [{"status": "ACTIVE", "visibilityStatus": "PUBLISHED"}]},
        _access_denied("ListAIGuardrails"),
    ]

    # Act
    finding = AIGuardrailCoverageCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    assert finding.evidence["required_permission"] == "wisdom:ListAIGuardrails"
    assert finding.evidence["assistants_evaluated"] == 1


def test_ai_encryption_no_resources_returns_not_applicable(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    mock_aws_client_factory.list_integration_associations_resilient.return_value = (
        _empty_association()
    )

    # Act
    finding = QConnectEncryptionCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.NOT_APPLICABLE


def test_ai_encryption_assistant_access_denied_uses_wisdom_permission(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.side_effect = [
        _assistant_association(),
        _empty_association(),
    ]
    factory.get_qconnect_assistant_resilient.side_effect = _access_denied("GetAssistant")

    # Act
    finding = QConnectEncryptionCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    assert finding.evidence["required_permission"] == "wisdom:GetAssistant"


def test_ai_encryption_missing_customer_key_fails(make_check_context, mock_aws_client_factory):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.side_effect = [
        _empty_association(),
        _knowledge_base_association(),
    ]
    factory.get_qconnect_knowledge_base_resilient.return_value = {
        "knowledgeBase": {"serverSideEncryptionConfiguration": {}}
    }

    # Act
    finding = QConnectEncryptionCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.FAIL
    assert finding.evidence["unencrypted_resources"][0]["resource_type"] == "knowledge_base"


def test_ai_knowledge_base_no_resource_returns_not_applicable(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    mock_aws_client_factory.list_integration_associations_resilient.return_value = (
        _empty_association()
    )

    # Act
    finding = KnowledgeBaseSyncHealthCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.NOT_APPLICABLE


def test_ai_knowledge_base_sync_failed_fails_with_bounded_reasons(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _knowledge_base_association()
    factory.get_qconnect_knowledge_base_resilient.return_value = {
        "knowledgeBase": {
            "name": "kb1",
            "status": "ACTIVE",
            "ingestionStatus": "SYNC_FAILED",
            "ingestionFailureReasons": ["x" * 400],
        }
    }

    # Act
    finding = KnowledgeBaseSyncHealthCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.FAIL
    failed = finding.evidence["failed_knowledge_bases"][0]
    assert failed["ingestion_status"] == "SYNC_FAILED"
    assert len(failed["ingestion_failure_reasons"][0]) == 300


def test_ai_knowledge_base_active_missing_ingestion_returns_skipped(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _knowledge_base_association()
    factory.get_qconnect_knowledge_base_resilient.return_value = {
        "knowledgeBase": {"name": "kb1", "status": "ACTIVE"}
    }

    # Act
    finding = KnowledgeBaseSyncHealthCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    observation = finding.evidence["incomplete_knowledge_bases"][0]
    assert observation["status"] == "ACTIVE"
    assert observation["ingestion_status"] is None


def test_ai_knowledge_base_transient_ingestion_returns_skipped(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _knowledge_base_association()
    factory.get_qconnect_knowledge_base_resilient.return_value = {
        "knowledgeBase": {
            "name": "kb1",
            "status": "ACTIVE",
            "ingestionStatus": "SYNCING_IN_PROGRESS",
        }
    }

    # Act
    finding = KnowledgeBaseSyncHealthCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    assert finding.evidence["incomplete_knowledge_bases"][0]["ingestion_status"] == (
        "SYNCING_IN_PROGRESS"
    )


def test_ai_knowledge_base_active_and_sync_success_passes(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _knowledge_base_association()
    factory.get_qconnect_knowledge_base_resilient.return_value = {
        "knowledgeBase": {
            "name": "kb1",
            "status": "ACTIVE",
            "ingestionStatus": "SYNC_SUCCESS",
        }
    }

    # Act
    finding = KnowledgeBaseSyncHealthCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS


def test_ai_prompt_no_prompts_returns_not_applicable(make_check_context, mock_aws_client_factory):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_prompts_resilient.return_value = {"aiPromptSummaries": []}

    # Act
    finding = AIPromptModelCostCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.NOT_APPLICABLE


def test_ai_prompt_second_page_premium_model_fails_and_uses_next_token(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_ai_prompts_resilient.side_effect = [
        {
            "aiPromptSummaries": [{"name": "cheap", "modelId": "amazon.nova-lite-v1"}],
            "nextToken": "prompt-page-2",
        },
        {"aiPromptSummaries": [{"name": "premium", "modelId": "anthropic.claude-opus-4-v1"}]},
    ]

    # Act
    finding = AIPromptModelCostCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.FAIL
    assert finding.evidence["prompts_checked"] == 2
    assert factory.list_ai_prompts_resilient.call_args_list[1].kwargs["nextToken"] == (
        "prompt-page-2"
    )


def test_ai_bedrock_logging_non_ai_instance_is_not_applicable_without_bedrock_call(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _empty_association()

    # Act
    finding = BedrockInvocationLoggingCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.NOT_APPLICABLE
    assert finding.evidence["bedrock_api_called"] is False
    factory.get_model_invocation_logging_configuration_resilient.assert_not_called()


def test_ai_bedrock_logging_q_enabled_without_destination_fails(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.get_model_invocation_logging_configuration_resilient.return_value = {
        "loggingConfig": {}
    }

    # Act
    finding = BedrockInvocationLoggingCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.FAIL


def test_ai_bedrock_logging_access_denied_returns_skipped(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.get_model_invocation_logging_configuration_resilient.side_effect = _access_denied(
        "GetModelInvocationLoggingConfiguration"
    )

    # Act
    finding = BedrockInvocationLoggingCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.SKIPPED
    assert finding.evidence["required_permission"] == (
        "bedrock:GetModelInvocationLoggingConfiguration"
    )


def test_ai_cross_region_non_ai_instance_is_not_applicable_without_bedrock_call(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _empty_association()

    # Act
    finding = CrossRegionInferenceAvailabilityCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.NOT_APPLICABLE
    assert finding.evidence["bedrock_api_called"] is False
    factory.list_inference_profiles_resilient.assert_not_called()


def test_ai_cross_region_second_page_profiles_affect_count_and_use_next_token(
    make_check_context, mock_aws_client_factory
):
    # Arrange
    _wire_access_denied(mock_aws_client_factory)
    factory = mock_aws_client_factory
    factory.list_integration_associations_resilient.return_value = _assistant_association()
    factory.list_inference_profiles_resilient.side_effect = [
        {
            "inferenceProfileSummaries": [{"inferenceProfileId": "profile-1"}],
            "nextToken": "profile-page-2",
        },
        {"inferenceProfileSummaries": [{"inferenceProfileId": "profile-2"}]},
    ]

    # Act
    finding = CrossRegionInferenceAvailabilityCheck().execute(make_check_context())

    # Assert
    assert finding.status == CheckStatus.PASS
    assert finding.evidence["cross_region_profile_count"] == 2
    assert factory.list_inference_profiles_resilient.call_args_list[1].kwargs == {
        "typeEquals": "SYSTEM_DEFINED",
        "maxResults": 100,
        "nextToken": "profile-page-2",
    }


def test_ai_registration_registers_exact_workstream_checks():
    # Arrange
    registry = CheckRegistry()

    # Act
    register_ai_ops_maturity_checks(registry)

    # Assert
    assert set(registry.list_check_ids()) == {
        "ai-ops-guardrail-001",
        "ai-ops-encryption-001",
        "ai-ops-kb-sync-001",
        "ai-ops-model-cost-001",
        "ai-ops-bedrock-logging-001",
        "ai-ops-cross-region-001",
    }


def test_ai_registration_skip_flow_analysis_keeps_instance_checks():
    # Arrange
    registry = CheckRegistry()

    # Act
    register_all_checks(registry, skip_flow_analysis=True)

    # Assert
    registered = set(registry.list_check_ids())
    assert {
        "ai-ops-guardrail-001",
        "ai-ops-encryption-001",
        "ai-ops-kb-sync-001",
        "ai-ops-model-cost-001",
        "ai-ops-bedrock-logging-001",
        "ai-ops-cross-region-001",
    } <= registered
