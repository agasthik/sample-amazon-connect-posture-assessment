"""
Blast-radius security checks for Lambda functions reachable from contact flows.

- sec-excessive-agency-001 : Lambda execution role overly broad (OWASP LLM06)

The check combines flow-content analysis (parser) with AWS API inspection
(Lambda configs, IAM roles), and degrades to SKIPPED on access denied.

Two checks were removed from this module rather than left in place:

``sec-ai-lex-001`` failed every Lex integration it found, unconditionally. It
could not read a bot's configuration, so a deployment with correctly guarded
bots received the same HIGH finding as one with none. Reimplementing it means
reading intent and slot configuration (``lex:ListIntents``, ``DescribeIntent``,
``ListSlots``, ``DescribeSlot``), which the assessment policy does not currently
grant.

``sec-ai-cascade-001`` asked a sound question — whether one model's output feeds
another's input with nothing validating in between — but identified AI stages by
substring-matching the Lambda ARN against hints including ``"ai"`` and ``"ml"``,
which match unrelated names such as ``ClaimLookup`` or ``EmailHandler`` while
missing any AI Lambda whose name does not advertise itself. Reimplementing it
means deriving AI involvement from the execution role's granted actions, the way
``ExcessiveAgencyCheck`` below already resolves roles.

Both are absent rather than approximated: a check that cannot distinguish a
healthy configuration from a broken one does not belong in a report someone
makes decisions from.
"""

from typing import Optional

from ..models import (
    CheckStatus,
    ContactFlow,
    ContactFlowGraph,
    Pillar,
    Remediation,
    RemediationReference,
    RemediationStep,
    Severity,
)
from ..parsers import ContactFlowParser
from .base import BaseCheck, CheckContext

_PARSER = ContactFlowParser()

_SENSITIVE_SERVICE_PREFIXES = (
    "iam:",
    "kms:Decrypt",
    "s3:Put",
    "s3:Delete",
    "dynamodb:Delete",
    "sqs:Send",
    "sns:Publish",
    "secretsmanager:",
    "organizations:",
)


def _parse_flow(flow: ContactFlow) -> Optional[ContactFlowGraph]:
    if not flow.content or not isinstance(flow.content, dict):
        return None
    try:
        return _PARSER.parse(flow.content)
    except Exception:
        return None


def _role_name_from_arn(arn: str) -> Optional[str]:
    if not arn or ":role/" not in arn:
        return None
    return arn.split(":role/", 1)[1].split("/")[-1]


class ExcessiveAgencyCheck(BaseCheck):
    """Detect Lambda functions with overly broad IAM roles (Req 27 / OWASP LLM06)."""

    def __init__(self):
        super().__init__(
            check_id="sec-excessive-agency-001",
            name="Excessive Agency / Lambda Privilege Scope",
            pillar=Pillar.SECURITY,
            severity=Severity.HIGH,
            description=(
                "Identifies Lambda functions invoked by contact flows whose "
                "execution roles grant overly broad or sensitive permissions, "
                "reducing blast radius if the flow is manipulated."
            ),
        )

    def execute(self, context: CheckContext):
        instance = context.instance
        factory = context.aws_client_factory
        flagged = []

        # Collect unique Lambda ARNs from all flows.
        lambda_arns = set()
        for flow in instance.contact_flows:
            graph = _parse_flow(flow)
            if not graph:
                continue
            for action in graph.actions.values():
                if action.action_type == "InvokeLambdaFunction":
                    arn = action.parameters.get("FunctionArn")
                    if arn:
                        lambda_arns.add(arn)

        if not lambda_arns:
            return self.create_finding(
                status=CheckStatus.PASS,
                resource_id=instance.instance_id,
                resource_type="ContactFlow",
                description="No Lambda integrations to evaluate.",
                evidence={"lambda_count": 0},
            )

        for fn_arn in lambda_arns:
            try:
                fn_resp = factory.get_lambda_function_resilient(fn_arn)
            except Exception as e:
                if factory.is_access_denied(e):
                    return self.skipped_for_access_denied(context, "lambda:GetFunction")
                continue

            role_arn = fn_resp.get("Configuration", {}).get("Role", "")
            role_name = _role_name_from_arn(role_arn)
            if not role_name:
                continue

            # Check inline policies for broad permissions.
            try:
                inline_names = factory.list_role_policies_resilient(role_name).get(
                    "PolicyNames", []
                )
                for policy_name in inline_names:
                    doc = factory.get_role_policy_resilient(role_name, policy_name).get(
                        "PolicyDocument", {}
                    )
                    for stmt in doc.get("Statement", []) if isinstance(doc, dict) else []:
                        if stmt.get("Effect") != "Allow":
                            continue
                        actions = stmt.get("Action", [])
                        if isinstance(actions, str):
                            actions = [actions]
                        for a in actions:
                            if a == "*" or any(
                                a.startswith(p) for p in _SENSITIVE_SERVICE_PREFIXES
                            ):
                                flagged.append(
                                    {
                                        "function_arn": fn_arn,
                                        "role_name": role_name,
                                        "excessive_action": a,
                                        "policy_name": policy_name,
                                    }
                                )
            except Exception as e:
                if factory.is_access_denied(e):
                    return self.skipped_for_access_denied(context, "iam:GetRolePolicy")
                continue

        if flagged:
            return self.create_finding(
                status=CheckStatus.FAIL,
                resource_id=instance.instance_id,
                resource_type="LambdaFunction",
                description=(
                    f"{len(flagged)} excessive permission(s) detected in Lambda "
                    "execution roles used by contact flows."
                ),
                evidence={"excessive_permissions": flagged},
                structured_remediation=Remediation(
                    summary="Scope down Lambda execution roles to least privilege.",
                    target_resources=list({f["function_arn"] for f in flagged}),
                    steps=[
                        RemediationStep(
                            order=1,
                            instruction=(
                                "For each flagged Lambda role, remove or constrain "
                                "sensitive-service permissions (iam:*, kms:Decrypt, "
                                "s3:Put/Delete, dynamodb:Delete, sqs:Send) to only "
                                "the specific resources the function needs."
                            ),
                            command=(
                                f"aws iam list-role-policies --role-name {flagged[0]['role_name']}"
                            ),
                        ),
                        RemediationStep(
                            order=2,
                            instruction=(
                                "Consider using a dedicated execution role per "
                                "Lambda rather than sharing a broad role across "
                                "multiple functions."
                            ),
                        ),
                    ],
                    references=[
                        RemediationReference(
                            title="OWASP LLM06: Excessive Agency",
                            url="https://owasp.org/www-project-top-10-for-large-language-model-applications/",  # noqa: E501
                        )
                    ],
                    applies_if="Lambda functions handle untrusted contact-flow data.",
                ),
            )

        return self.create_finding(
            status=CheckStatus.PASS,
            resource_id=instance.instance_id,
            resource_type="LambdaFunction",
            description=(
                f"Evaluated {len(lambda_arns)} Lambda execution role(s); no "
                "excessive permissions detected."
            ),
            evidence={"lambda_count": len(lambda_arns)},
        )


def register_ai_agent_security_checks(registry) -> None:
    """Register all AI/agentic security checks."""
    registry.register_check(ExcessiveAgencyCheck())
