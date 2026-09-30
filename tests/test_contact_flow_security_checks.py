"""
Tests for contact-flow security checks (Task 5 / Requirements 20-23, 26, 39).

These checks consume the parser so we build flow JSON using conftest helpers.
"""

from amazon_connect_assessment.checks.contact_flow_security_checks import (
    DynamicPromptInjectionCheck,
    ExternalTransferTollFraudCheck,
    LambdaResponseValidationCheck,
    PIIInPromptsCheck,
    SensitiveDataInAttributesCheck,
    register_contact_flow_security_checks,
)
from amazon_connect_assessment.checks.registry import CheckRegistry
from amazon_connect_assessment.models import CheckStatus, ContactFlow, Severity
from tests.conftest import build_action, build_contact_flow


def _instance_with_flow(instance, flow_json, name="TestFlow", flow_type="CONTACT_FLOW"):
    """Attach a single parsed flow to the instance fixture."""
    instance.contact_flows = [
        ContactFlow(
            id="f1",
            arn="arn:aws:connect:us-east-1:123:instance/i/flow/f1",
            name=name,
            type=flow_type,
            state="ACTIVE",
            content=flow_json,
        )
    ]
    return instance


# --- Prompt injection (sec-prompt-inject-001) ---


class TestDynamicPromptInjection:
    def test_prompt_check_attributes_ssml_returns_medium_review_candidate(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "MessageParticipant", {"SSML": "Hello $.Attributes.Name"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)
        check = DynamicPromptInjectionCheck()

        # Act
        finding = check.execute(make_check_context(instance=instance))

        # Assert
        assert check.check_id == "sec-prompt-inject-001"
        assert check.name == "Potential Unsafe Dynamic Content in Prompts"
        assert check.severity == Severity.MEDIUM
        assert finding.status == CheckStatus.FAIL
        assert finding.severity == Severity.MEDIUM
        candidate = finding.evidence["actionable_review_candidates"][0]
        assert candidate == finding.evidence["flagged_prompts"][0]
        assert candidate["action_type"] == "MessageParticipant"
        assert candidate["dynamic_refs"] == ["$.Attributes.Name"]
        assert candidate["source_categories"] == ["attributes_unknown"]
        assert candidate["prompt_preview"] == "Hello $.Attributes.Name"
        assert candidate["interpreted_as"] == "ssml"
        assert candidate["flow_type"] == "CONTACT_FLOW"
        assert candidate["reachable"] is True
        assert candidate["audience_basis"] == (
            "flow type does not establish an agent/other audience"
        )
        assert candidate["sanitization_assessed"] is False
        assert "does not prove" in finding.description
        assert "unknown unless its writer is traced" in finding.description

    def test_prompt_check_external_ssml_classifies_external_result(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "PlayPrompt", {"SSML": "Status $.External.Result"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        candidate = finding.evidence["actionable_review_candidates"][0]
        assert finding.severity == Severity.MEDIUM
        assert candidate["dynamic_refs"] == ["$.External.Result"]
        assert candidate["source_categories"] == ["external_or_lambda_result"]

    def test_prompt_check_supported_reference_roots_records_source_categories(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        prompt = (
            "$.Lex.Slots.Name $.Media.InitialMessage $.SegmentAttributes.Topic "
            "$.Media.Sip.Headers.X-Test $.Customer.Name $.Attributes.Name "
            "$.CustomSource.Value"
        )
        flow = build_contact_flow([build_action("a1", "PlayPrompt", {"SSML": prompt})])
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        candidate = finding.evidence["actionable_review_candidates"][0]
        assert candidate["source_categories"] == [
            "lex",
            "media_initial_message",
            "segment_attributes",
            "sip_metadata",
            "customer",
            "attributes_unknown",
            "unknown",
        ]

    def test_prompt_check_plain_text_generic_flow_returns_pass_with_information(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "MessageParticipant", {"Text": "Hello $.Attributes.Name"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.PASS
        assert finding.evidence["actionable_review_candidates"] == []
        assert len(finding.evidence["informational_dynamic_references"]) == 1
        assert "No reachable prompt met the review threshold" in finding.description

    def test_prompt_check_agent_whisper_plain_text_returns_medium_review_candidate(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "MessageParticipant", {"Text": "Reason $.Attributes.Reason"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow, flow_type="AGENT_WHISPER")

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.FAIL
        assert finding.severity == Severity.MEDIUM
        assert finding.evidence["actionable_review_candidates"][0]["interpreted_as"] == "text"

    def test_prompt_check_system_root_prefix_collision_returns_review_candidate(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "PlayPrompt", {"SSML": "Hello $.AgentControlled.Name"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        candidate = finding.evidence["actionable_review_candidates"][0]
        assert finding.status == CheckStatus.FAIL
        assert candidate["source_categories"] == ["unknown"]
        assert finding.evidence["system_attribute_prompts_excluded"] == 0

    def test_prompt_check_stored_customer_input_ssml_returns_pass_as_constrained(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "PlayPrompt", {"SSML": "Digits $.StoredCustomerInput"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.PASS
        assert finding.evidence["actionable_review_candidates"] == []
        assert finding.evidence["constrained_reference_prompts_excluded"] == 1

    def test_prompt_check_customer_endpoint_address_ssml_returns_pass_as_constrained(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "PlayPrompt", {"SSML": "Number $.CustomerEndpoint.Address"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.PASS
        excluded = finding.evidence["excluded_reference_prompts"][0]
        assert excluded["source_categories"] == ["constrained_customer_endpoint_address"]

    def test_prompt_check_unreachable_dynamic_prompt_returns_pass_without_evidence(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [
                build_action("entry", "DisconnectParticipant"),
                build_action("orphan", "PlayPrompt", {"SSML": "Hello $.External.Name"}),
            ],
            start_action="entry",
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.PASS
        assert finding.evidence["actionable_review_candidates"] == []
        assert finding.evidence["informational_dynamic_references"] == []
        assert finding.evidence["excluded_reference_prompts"] == []

    def test_prompt_check_missing_entry_returns_skipped_with_disclosure(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "PlayPrompt", {"SSML": "Hello $.External.Name"})],
            start_action="missing",
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.SKIPPED
        assert finding.evidence["flows_analyzed"] == 0
        assert finding.evidence["analysis_complete"] is False
        assert finding.evidence["unanalyzed_flows"][0]["reason"] == (
            "entry action is missing or invalid"
        )
        assert "did not scan every action as a fallback" in finding.description

    def test_prompt_check_unavailable_input_flow_returns_skipped(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        instance = _instance_with_flow(sample_connect_instance, None)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.SKIPPED
        assert finding.evidence["flows_parsed"] == 0
        assert finding.evidence["unanalyzed_flows"][0]["reason"] == "flow content unavailable"

    def test_prompt_check_partial_entry_failure_discloses_incomplete_analysis(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        valid_flow = build_contact_flow(
            [build_action("candidate", "PlayPrompt", {"SSML": "Hello $.External.Name"})]
        )
        invalid_flow = build_contact_flow(
            [build_action("hidden", "PlayPrompt", {"SSML": "Hello $.External.Hidden"})],
            start_action="missing",
        )
        instance = _instance_with_flow(sample_connect_instance, valid_flow, name="ValidFlow")
        instance.contact_flows.append(
            ContactFlow(
                id="f2",
                arn="arn:aws:connect:us-east-1:123:instance/i/flow/f2",
                name="InvalidFlow",
                type="CONTACT_FLOW",
                state="ACTIVE",
                content=invalid_flow,
            )
        )

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.FAIL
        assert finding.evidence["flows_analyzed"] == 1
        assert finding.evidence["flows_unanalyzed"] == 1
        assert finding.evidence["analysis_complete"] is False
        assert [row["action_id"] for row in finding.evidence["actionable_review_candidates"]] == [
            "candidate"
        ]
        assert "Analysis was incomplete for 1 flow(s)" in finding.description

    def test_prompt_check_partial_analysis_without_candidate_returns_skipped(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        valid_flow = build_contact_flow(
            [build_action("entry", "MessageParticipant", {"Text": "Hello"})]
        )
        invalid_flow = build_contact_flow(
            [build_action("hidden", "PlayPrompt", {"SSML": "Hello $.External.Hidden"})],
            start_action="missing",
        )
        instance = _instance_with_flow(sample_connect_instance, valid_flow, name="ValidFlow")
        instance.contact_flows.append(
            ContactFlow(
                id="f2",
                arn="arn:aws:connect:us-east-1:123:instance/i/flow/f2",
                name="InvalidFlow",
                type="CONTACT_FLOW",
                state="ACTIVE",
                content=invalid_flow,
            )
        )

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.SKIPPED
        assert finding.evidence["flows_analyzed"] == 1
        assert finding.evidence["flows_unanalyzed"] == 1
        assert "could not inspect every input contact flow" in finding.description

    def test_prompt_check_mixed_rows_targets_only_actionable_action(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [
                build_action(
                    "info",
                    "MessageParticipant",
                    {"Text": "Hello $.Attributes.Name"},
                    next_action="candidate",
                ),
                build_action(
                    "candidate", "MessageParticipant", {"SSML": "Status $.External.Result"}
                ),
            ]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert [row["action_id"] for row in finding.evidence["actionable_review_candidates"]] == [
            "candidate"
        ]
        assert [
            row["action_id"] for row in finding.evidence["informational_dynamic_references"]
        ] == ["info"]
        assert finding.structured_remediation.target_resources == ["candidate"]

    def test_prompt_check_review_description_uses_conditional_customer_language(
        self, make_check_context, sample_connect_instance
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "PlayPrompt", {"SSML": "Hello $.Attributes.Name"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        lowered = finding.description.lower()
        assert "comes from caller" not in lowered
        assert "without sanitizing" not in lowered
        assert "</speak><speak>" not in lowered
        assert "allows arbitrary text" in lowered
        assert "without the required escaping" in lowered
        assert "another person" in lowered
        assert "error branch" in lowered
        assert "trusted ivr or agent instructions" in lowered
        assert "code execution or account takeover" in lowered
        assert '<break time="10s"/>' in finding.description
        assert "runtime behavior still needs validation" in lowered
        remediation = finding.remediation
        assert "DTMF" in remediation
        assert "fixed enum" in remediation
        assert "trusted constant" in remediation
        assert "XML-escape &, <, and >" in remediation
        assert "Error branch" in remediation
        assert "Check contact attributes" not in remediation

    def test_prompt_check_execution_avoids_polly_and_network_calls(
        self, make_check_context, sample_connect_instance, monkeypatch
    ):
        # Arrange
        flow = build_contact_flow(
            [build_action("a1", "PlayPrompt", {"SSML": "Hello $.External.Name"})]
        )
        instance = _instance_with_flow(sample_connect_instance, flow)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("unit check attempted an external call")

        monkeypatch.setattr("boto3.client", fail_if_called)
        monkeypatch.setattr("socket.create_connection", fail_if_called)

        # Act
        finding = DynamicPromptInjectionCheck().execute(make_check_context(instance=instance))

        # Assert
        assert finding.status == CheckStatus.FAIL
        assert finding.severity == Severity.MEDIUM


# --- Lambda response validation (sec-lambda-validation-001) ---


class TestLambdaResponseValidation:
    def test_branch_without_default_fails(self, make_check_context, sample_connect_instance):
        flow = build_contact_flow(
            [
                build_action(
                    "a1",
                    "InvokeLambdaFunction",
                    {"FunctionArn": "arn:...:fn"},
                    conditions=[
                        {
                            "NextAction": "a2",
                            "Condition": {"Operator": "Equals", "Operands": ["1"]},
                        }
                    ],
                ),
                build_action("a2", "DisconnectParticipant"),
            ]
        )
        inst = _instance_with_flow(sample_connect_instance, flow)
        finding = LambdaResponseValidationCheck().execute(make_check_context(instance=inst))
        assert finding.status == CheckStatus.FAIL

    def test_lambda_with_default_passes(self, make_check_context, sample_connect_instance):
        flow = build_contact_flow(
            [
                build_action(
                    "a1",
                    "InvokeLambdaFunction",
                    {"FunctionArn": "arn:...:fn"},
                    next_action="a3",
                    conditions=[
                        {
                            "NextAction": "a2",
                            "Condition": {"Operator": "Equals", "Operands": ["1"]},
                        }
                    ],
                ),
                build_action("a2", "DisconnectParticipant"),
                build_action("a3", "DisconnectParticipant"),
            ]
        )
        inst = _instance_with_flow(sample_connect_instance, flow)
        finding = LambdaResponseValidationCheck().execute(make_check_context(instance=inst))
        assert finding.status == CheckStatus.PASS


# --- Toll fraud (sec-toll-fraud-001) ---


class TestExternalTransferTollFraud:
    def test_dynamic_transfer_fails(self, make_check_context, sample_connect_instance):
        flow = build_contact_flow(
            [
                build_action(
                    "a1",
                    "TransferContactToPhoneNumber",
                    {"PhoneNumber": "$.Attributes.DestNumber"},
                )
            ]
        )
        inst = _instance_with_flow(sample_connect_instance, flow)
        finding = ExternalTransferTollFraudCheck().execute(make_check_context(instance=inst))
        assert finding.status == CheckStatus.FAIL
        assert finding.severity.value == "critical"

    def test_static_transfer_passes(self, make_check_context, sample_connect_instance):
        flow = build_contact_flow(
            [
                build_action(
                    "a1",
                    "TransferContactToPhoneNumber",
                    {"PhoneNumber": "+18005551234"},
                )
            ]
        )
        inst = _instance_with_flow(sample_connect_instance, flow)
        finding = ExternalTransferTollFraudCheck().execute(make_check_context(instance=inst))
        assert finding.status == CheckStatus.PASS


# --- Sensitive data in attributes (sec-sensitive-data-001) ---


class TestSensitiveDataInAttributes:
    def test_ssn_attribute_fails(self, make_check_context, sample_connect_instance):
        flow = build_contact_flow(
            [
                build_action(
                    "a1",
                    "UpdateContactAttributes",
                    {"Attributes": {"CustomerSSN": "123-45-6789"}},
                )
            ]
        )
        inst = _instance_with_flow(sample_connect_instance, flow)
        finding = SensitiveDataInAttributesCheck().execute(make_check_context(instance=inst))
        assert finding.status == CheckStatus.FAIL
        assert "ssn" in finding.evidence["flagged_attributes"][0]["attribute_name"].lower()

    def test_safe_attribute_passes(self, make_check_context, sample_connect_instance):
        flow = build_contact_flow(
            [
                build_action(
                    "a1",
                    "UpdateContactAttributes",
                    {"Attributes": {"Language": "en-US"}},
                )
            ]
        )
        inst = _instance_with_flow(sample_connect_instance, flow)
        finding = SensitiveDataInAttributesCheck().execute(make_check_context(instance=inst))
        assert finding.status == CheckStatus.PASS


# --- PII in prompts (sec-pii-prompts-001) ---


class TestPIIInPrompts:
    def test_unmasked_account_number_fails(self, make_check_context, sample_connect_instance):
        flow = build_contact_flow(
            [
                build_action(
                    "a1",
                    "MessageParticipant",
                    {"Text": "Your AccountNumber is $.Attributes.AccountNumber"},
                )
            ]
        )
        inst = _instance_with_flow(sample_connect_instance, flow)
        finding = PIIInPromptsCheck().execute(make_check_context(instance=inst))
        assert finding.status == CheckStatus.FAIL

    def test_masked_reference_passes(self, make_check_context, sample_connect_instance):
        flow = build_contact_flow(
            [
                build_action(
                    "a1",
                    "MessageParticipant",
                    {"Text": "Account ending in last4 $.Attributes.AcctLast4"},
                )
            ]
        )
        inst = _instance_with_flow(sample_connect_instance, flow)
        finding = PIIInPromptsCheck().execute(make_check_context(instance=inst))
        assert finding.status == CheckStatus.PASS


# --- Registration ---


def test_register_contact_flow_security_checks():
    registry = CheckRegistry()
    register_contact_flow_security_checks(registry)
    ids = {c.check_id for c in registry.get_all_checks()}
    expected = {
        "sec-prompt-inject-001",
        "sec-lambda-validation-001",
        "sec-toll-fraud-001",
        "sec-sensitive-data-001",
        "sec-pii-prompts-001",
    }
    assert expected <= ids
