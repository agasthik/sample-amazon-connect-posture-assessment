"""
Contact-flow security checks (Phase 2 / Task 5).

These checks parse contact flow JSON content (via the parser package) and
detect security vulnerabilities in the flow logic:

- sec-prompt-inject-001       : SSML markup / spoken-content injection in prompts
- sec-lambda-validation-001   : Lambda response used for branching without validation
- sec-toll-fraud-001          : External transfer to dynamic phone number (toll fraud)
- sec-sensitive-data-001      : Sensitive data stored in contact attributes
- sec-pii-prompts-001         : PII read back in voice prompts without masking

Each check operates on a parsed ContactFlowGraph derived from the flow content.
"""

import re
from typing import Dict, List, Optional

from ..models import (
    CheckStatus,
    ContactFlow,
    ContactFlowGraph,
    FlowAction,
    Pillar,
    Remediation,
    RemediationReference,
    RemediationStep,
    Severity,
)
from ..parsers import ContactFlowParser
from .base import BaseCheck, CheckContext

_PARSER = ContactFlowParser()

# Patterns indicating sensitive data in attribute names (case-insensitive).
# These are DETECTION substrings for flagging PII/PCI/PHI stored in contact
# attributes — not recommended attribute names for customer use. A match should
# trigger a compliance review (HIPAA, PCI-DSS, or GDPR as applicable).
_SENSITIVE_ATTR_PATTERNS = (
    "ssn",
    "social",
    "creditcard",
    "cardnumber",
    "cvv",
    "pin",
    "password",
    "passcode",
    "dob",
    "dateofbirth",
    "accountnumber",
    "routingnumber",
    "bankaccount",
    "taxid",
    "passportnumber",
)

# Transfer-to-phone-number action type variants.
_PHONE_TRANSFER_TYPES = {
    "TransferContactToPhoneNumber",
    "TransferToPhoneNumber",
}


def _parse_flow(flow: ContactFlow) -> Optional[ContactFlowGraph]:
    """Parse a ContactFlow's content into a graph; return None on failure."""
    if not flow.content or not isinstance(flow.content, dict):
        return None
    try:
        return _PARSER.parse(flow.content)
    except Exception:
        return None


def _is_dynamic_reference(value) -> bool:
    """True if the value references a contact attribute or external source."""
    if not isinstance(value, str):
        return False
    return value.startswith("$.") or value.startswith("$[")


# Matches a JSONPath-style contact attribute reference embedded anywhere in
# free text, e.g. the "$.Attributes.Name" in "Hello, $.Attributes.Name.".
# Captures $.<segment>(.<segment>|[...])* — stops at whitespace or a
# JSONPath-illegal character so trailing punctuation in prose ("...Name.")
# isn't swallowed into the reference.
_DYNAMIC_REF_PATTERN = re.compile(r"\$(?:\.[A-Za-z0-9_]+|\[[^\]]*\])+")


# JSONPath root segments for Amazon Connect's *system* attributes — see
# https://docs.aws.amazon.com/connect/latest/adminguide/connect-attrib-list.html.
# These are predefined, Connect-populated values that a flow author cannot
# redefine and a caller cannot set to arbitrary text (queue/agent names
# configured by the admin, region/channel/contact-id metadata, telephony
# SIP headers from the carrier). They are excluded from
# DynamicPromptInjectionCheck's flags: a $.Queue.Name reference in a
# prompt is not an injection vector the way a value sourced from caller
# DTMF input, a Lex slot, or a Lambda/CRM lookup is, because nothing the
# caller says or does changes what these resolve to. Listed by root
# segment so nested paths (e.g. $.Queue.OutboundCallerId.Address) match
# too.
_SYSTEM_ATTRIBUTE_ROOTS = (
    "$.awsregion",
    "$.systemendpoint",
    "$.queue",
    "$.agent",
    "$.contactid",
    "$.initialcontactid",
    "$.taskcontactid",
    "$.previouscontactid",
    "$.channel",
    "$.instancearn",
    "$.initiationmethod",
    "$.languagecode",
    "$.tags",
    "$.media.sip",
)


def _is_system_attribute_reference(value: str) -> bool:
    """
    True if ``value`` is a Connect *system* attribute reference (queue
    name, agent name, region, channel, contact ID, carrier SIP metadata,
    etc) rather than a value that originated from the caller, a bot slot,
    or a Lambda/external lookup.

    Used to keep DynamicPromptInjectionCheck focused on genuine injection
    risk: system attributes are admin-configured or Connect-populated and
    a caller cannot influence what they resolve to, so speaking one in a
    prompt carries none of the SSML-injection risk this check exists to
    catch.
    """
    lowered = value.lower()
    return any(lowered.startswith(root) for root in _SYSTEM_ATTRIBUTE_ROOTS)


# Flow types whose prompts are spoken to somebody other than the caller who
# supplied the value. This is the difference between a caller injecting text
# into their own call — where they are both attacker and audience, which is
# not an attack — and injecting text that an *agent* then hears as though the
# platform authored it.
_OTHER_AUDIENCE_FLOW_TYPES = {
    "AGENT_WHISPER",
    "OUTBOUND_WHISPER",
    "AGENT_HOLD",
    "AGENT_TRANSFER",
}


def _prompt_text_and_markup(action: FlowAction) -> tuple[str, bool]:
    """
    Return ``(text, interpreted_as_ssml)`` for a prompt-playing action.

    This distinction decides whether markup in a substituted value is
    *parsed* or merely *spoken*, which is the difference between an
    injection vulnerability and a cosmetic oddity. Amazon Connect emits the
    flow designer's "Interpret as" choice as a separate parameter key —
    ``SSML`` for SSML, ``Text`` for plain text — so the key that carries the
    string also tells us how Polly will treat it.

    Some flow revisions additionally carry an explicit discriminator
    (``TextType``, the name Polly's own API uses, or ``InterpretAs``). Both
    are honoured when present so a flow that spells it out is not misread.
    The fallback is plain text, which is the flow designer's default and the
    safer assumption to make: it under-states severity rather than inventing
    a markup-parsing vulnerability that isn't there.
    """
    params = action.parameters or {}

    ssml_value = params.get("SSML")
    if ssml_value:
        return str(ssml_value), True

    text_value = str(params.get("Text", "") or "")
    discriminator = str(params.get("TextType") or params.get("InterpretAs") or "")
    return text_value, discriminator.strip().lower() == "ssml"


def _get_phone_destination(action: FlowAction) -> Optional[str]:
    """Extract the phone number destination from a transfer action."""
    params = action.parameters or {}
    return (
        params.get("PhoneNumber")
        or params.get("ContactFlowId")  # some older formats
        or params.get("Endpoint", {}).get("Address")
        if isinstance(params.get("Endpoint"), dict)
        else params.get("PhoneNumber")
    )


class DynamicPromptInjectionCheck(BaseCheck):
    """
    Detect caller- or externally-sourced values spoken in voice prompts
    (Req 20).

    Severity is not uniform, because exploitability is not uniform. Three
    cases, and conflating them was this check's original defect — it reported
    all of them as HIGH "prompt injection":

    1. **The prompt is interpreted as SSML.** Polly parses the markup in the
       substituted value, so ``</speak><speak>...`` smuggled through an
       attribute changes what the platform says. A genuine injection
       vulnerability: HIGH.
    2. **Plain text, spoken to somebody other than the caller** (an agent
       whisper, a transfer or hold flow). No markup is parsed, but text the
       caller authored reaches an agent as though Connect authored it, which
       is a workable social-engineering path: MEDIUM.
    3. **Plain text, spoken back to the caller who supplied it.** The caller
       hears their own value. Attacker and audience are the same person, so
       there is no attack unless an external system supplied the value —
       worth reporting so it can be traced, not worth paging anyone: LOW.

    Note the check reports what it can prove from flow content. It cannot
    prove where an attribute's value *originated* — Connect does not record
    that in the flow — so it excludes references a caller provably cannot
    influence (Connect system attributes) and asks the reader to trace the
    rest. That is stated in the finding rather than papered over.
    """

    def __init__(self):
        super().__init__(
            check_id="sec-prompt-inject-001",
            # Named for the mechanism, not the fashionable phrase. "Prompt
            # injection" now reads as an LLM attack; this is SSML markup
            # injection into Amazon Polly and spoken-content spoofing, and
            # no model is involved. The check ID is unchanged so --diff
            # against existing baselines keeps working.
            name="Voice Prompt Injection (SSML Markup and Spoken Content)",
            pillar=Pillar.SECURITY,
            # Declared severity is the worst case (SSML parsing). Individual
            # findings downgrade per the ladder in the class docstring.
            severity=Severity.HIGH,
            description=(
                "Detects contact flows that speak caller- or external-system-"
                "sourced values in voice prompts without sanitizing them, and "
                "rates each by whether Polly parses the value as SSML markup "
                "and who actually hears it."
            ),
        )

    def execute(self, context: CheckContext):
        instance = context.instance
        flagged = []
        system_attr_refs_seen = 0

        for flow in instance.contact_flows:
            graph = _parse_flow(flow)
            if not graph:
                continue
            other_audience = (flow.type or "").upper() in _OTHER_AUDIENCE_FLOW_TYPES
            for action in graph.actions.values():
                if action.action_type not in (
                    "MessageParticipant",
                    "PlayPrompt",
                    "PlayAudio",
                ):
                    continue
                text, is_ssml = _prompt_text_and_markup(action)
                dynamic_refs = _DYNAMIC_REF_PATTERN.findall(text)
                if not dynamic_refs:
                    continue
                # $.Queue.Name, $.Agent.*, and the other system attributes
                # in _SYSTEM_ATTRIBUTE_ROOTS are Connect-populated and
                # admin-configured — a caller cannot set what they resolve
                # to, so speaking one carries none of the SSML-injection
                # risk this check exists to catch. Skip a prompt whose
                # *only* dynamic references are system attributes; a
                # prompt mixing a system attribute with a caller-sourced
                # one is still flagged, since the caller-sourced part is
                # the actual risk.
                if all(_is_system_attribute_reference(ref) for ref in dynamic_refs):
                    system_attr_refs_seen += 1
                    continue
                if is_ssml:
                    tier, tier_severity = "ssml_parsed", Severity.HIGH
                elif other_audience:
                    tier, tier_severity = "other_audience", Severity.MEDIUM
                else:
                    tier, tier_severity = "spoken_to_source", Severity.LOW
                flagged.append(
                    {
                        "flow": flow.name,
                        "flow_id": flow.id,
                        "flow_type": flow.type,
                        "action_id": action.action_id,
                        "dynamic_ref": text[:120],
                        "interpreted_as": "ssml" if is_ssml else "text",
                        "risk_tier": tier,
                        "severity": tier_severity.value,
                    }
                )

        if flagged:
            ssml_hits = [f for f in flagged if f["risk_tier"] == "ssml_parsed"]
            other_audience_hits = [f for f in flagged if f["risk_tier"] == "other_audience"]
            self_audience_hits = [f for f in flagged if f["risk_tier"] == "spoken_to_source"]

            # The finding carries the worst tier present, so one SSML prompt is
            # not diluted to LOW by a dozen harmless plain-text ones alongside.
            if ssml_hits:
                finding_severity = Severity.HIGH
            elif other_audience_hits:
                finding_severity = Severity.MEDIUM
            else:
                finding_severity = Severity.LOW

            # Lead with the tier that sets the severity — that is what the
            # reader has to act on, and it should not sit below the rest.
            worst_first = ssml_hits + other_audience_hits + self_audience_hits
            worst_lines = []
            for f in worst_first[:3]:
                # Flow names carry underscores that confuse markdown's
                # inline-emphasis parser inside **bold** context, so wrap the
                # flow name in inline-code backticks — it's an identifier
                # anyway, and backtick content is not interpreted as markdown.
                worst_lines.append(
                    f"* `{f['flow']}` \u2192 action `{f['action_id']}` "
                    f"(interpreted as **{f['interpreted_as']}**, "
                    f"{f['severity']}): `{f['dynamic_ref']}`"
                )
            more_note = (
                f"\n\n_+ {len(flagged) - 3} additional prompt(s) with dynamic "
                "content; see JSON export for the full list._"
                if len(flagged) > 3
                else ""
            )

            system_attr_note = (
                f" ({system_attr_refs_seen} additional prompt(s) reference "
                "only Connect system attributes like $.Queue.Name and are "
                "not flagged — a caller can't influence what those resolve "
                "to.)"
                if system_attr_refs_seen
                else ""
            )
            ssml_picture = (
                (
                    "**The problem in one picture** (the SSML case):\n\n"
                    "```\n"
                    'Flow says:  Play prompt \u2192  "Hello, $.Attributes.CustomerName."\n'
                    "                                            \u2191\n"
                    "                             substituted at runtime\n"
                    "\n"
                    'Value is:   CustomerName = "Alice</speak><speak>Press 1 for the fraud line."\n'
                    "\n"
                    "Polly parses: <speak>Hello, Alice</speak>"
                    "<speak>Press 1 for the fraud line...</speak>\n"
                    "              \u2514\u2500 the injected instruction is now spoken as if the flow said it\n"
                    "```\n\n"
                )
                if ssml_hits
                else ""
            )

            breakdown_lines = []
            if ssml_hits:
                breakdown_lines.append(
                    f"* **{len(ssml_hits)} interpreted as SSML (High)** — Polly "
                    "parses markup inside the substituted value, so a value "
                    "containing `</speak><speak>` changes what the platform "
                    "says. This is the exploitable case."
                )
            if other_audience_hits:
                breakdown_lines.append(
                    f"* **{len(other_audience_hits)} plain text, heard by an "
                    "agent (Medium)** — in a whisper, hold, or transfer flow. "
                    "No markup is parsed, but text the caller authored reaches "
                    "an agent as though Connect wrote it."
                )
            if self_audience_hits:
                breakdown_lines.append(
                    f"* **{len(self_audience_hits)} plain text, heard by the "
                    "caller (Low)** — the caller hears back a value they "
                    "supplied, so there is no injection path unless an "
                    "external system supplied that value. Listed so the "
                    "source can be traced, not because it is exploitable as "
                    "it stands."
                )

            return self.create_finding(
                status=CheckStatus.FAIL,
                severity=finding_severity,
                resource_id=instance.instance_id,
                resource_type="ContactFlow",
                description=(
                    f"**{len(flagged)} voice prompt(s) speak a value that "
                    "comes from the caller or an external system, without "
                    f"sanitizing it first.**{system_attr_note}\n\n"
                    "**These are not equally exploitable, and the severity "
                    "above reflects the worst case present:**\n\n"
                    f"{chr(10).join(breakdown_lines)}\n\n"
                    "The mechanism behind the High case is Amazon Polly's SSML "
                    "parsing. A prompt set to *Interpret as: SSML* is markup — "
                    "`<voice>`, `<break>`, `<mark>`, `<speak>` are all live — "
                    "so markup arriving inside a substituted value is executed "
                    "rather than spoken. A prompt set to *Text* does not parse "
                    "markup, which is why those are rated lower here instead of "
                    "being reported as injection.\n\n"
                    "**Only caller-influenced values are flagged.** A "
                    "reference like `$.Queue.Name` or `$.Agent.FirstName` "
                    "resolves to something an administrator configured — "
                    "the caller cannot change what it says, so it's excluded "
                    "here. What's flagged below are values that trace back "
                    "to something the caller said (a Lex slot, free-form "
                    "transcription) or that an external system (Lambda, CRM "
                    "lookup) returned.\n\n"
                    "**Trace the source before treating any of these as a "
                    "bug.** Connect does not record where an attribute's value "
                    "came from, so this check cannot tell a speech slot from a "
                    "DTMF digit capture. That distinction decides "
                    "exploitability: a value captured by **Store customer "
                    "input** over DTMF can only hold digits, and digits cannot "
                    "carry SSML markup, so such a prompt is not exploitable "
                    "however it is interpreted. Free-form speech slots and "
                    "external lookups are what can carry markup.\n\n"
                    f"{ssml_picture}"
                    f"**Flagged prompts (top {min(3, len(worst_first))}, "
                    f"worst first):**\n\n"
                    f"{chr(10).join(worst_lines)}{more_note}\n\n"
                    "**Fix (in the flow designer):**\n"
                    "1. Open each flagged prompt and check its **Interpret "
                    "as** setting. If it is SSML and does not need to be, "
                    "switch it to Text \u2014 that alone closes the "
                    "markup-parsing path.\n"
                    "2. If it must stay SSML, sanitize upstream with either a "
                    "**Check contact attributes** block that only passes "
                    "values matching a known-safe pattern (digits, an enum), "
                    "or an **Invoke Lambda function** that strips `<`, `>`, "
                    "`&` and unmatched quotes, truncates to a safe length, and "
                    "returns a `SafeCustomerName` attribute the prompt "
                    "references instead of the raw one."
                ),
                evidence={
                    "flagged_prompts": flagged,
                    "system_attribute_prompts_excluded": system_attr_refs_seen,
                },
                structured_remediation=Remediation(
                    summary=(
                        "Switch prompts off SSML where markup isn't needed, "
                        "and sanitize dynamic content reaching the ones that "
                        "keep it — SSML is markup Polly interprets, and "
                        "unchecked substitution into it is an injection vector."
                    ),
                    target_resources=[f["action_id"] for f in flagged],
                    steps=[
                        RemediationStep(
                            order=1,
                            instruction=(
                                "Check each flagged prompt's 'Interpret as' "
                                "setting. Prompts set to Text do not parse "
                                "markup; prompts set to SSML are the "
                                "injectable ones. Switch SSML to Text "
                                "wherever the prompt does not actually use "
                                "markup — the cheapest fix, and it removes "
                                "the vector outright rather than filtering it."
                            ),
                        ),
                        RemediationStep(
                            order=2,
                            instruction=(
                                "Trace what sets each flagged attribute. A "
                                "DTMF 'Store customer input' capture holds "
                                "digits only and cannot carry markup, so "
                                "those need no sanitization. Free-form speech "
                                "slots and Lambda/CRM lookups can, and are "
                                "what the remaining steps address."
                            ),
                        ),
                        RemediationStep(
                            order=3,
                            instruction=(
                                "For values that must stay in an SSML prompt, "
                                "insert a Lambda immediately upstream that: "
                                "strips `<`, `>`, `&`; rejects unbalanced "
                                "quotes; truncates to a safe length (e.g. 60 "
                                "chars for a name); returns the sanitized "
                                "string as a new attribute. The prompt then "
                                "references the sanitized attribute, not the "
                                "raw one."
                            ),
                        ),
                        RemediationStep(
                            order=4,
                            instruction=(
                                "For values that should match a fixed set "
                                "(department names, product codes), use a "
                                "Check contact attributes block to route on "
                                "an allowlist instead of speaking the raw "
                                "value."
                            ),
                        ),
                    ],
                    references=[
                        RemediationReference(
                            title="Amazon Polly SSML reference (interpreted tags)",
                            url="https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html",
                        ),
                        RemediationReference(
                            title="Using contact attributes in Amazon Connect",
                            url="https://docs.aws.amazon.com/connect/latest/adminguide/connect-attrib-list.html",  # noqa: E501
                        ),
                    ],
                    applies_if=(
                        "prompts speak values that originated from callers or "
                        "external systems (as opposed to hard-coded strings)."
                    ),
                ),
            )

        system_attr_note = (
            f" ({system_attr_refs_seen} prompt(s) reference only Connect "
            "system attributes like $.Queue.Name, which are excluded since "
            "a caller can't influence them.)"
            if system_attr_refs_seen
            else ""
        )
        return self.create_finding(
            status=CheckStatus.PASS,
            resource_id=instance.instance_id,
            resource_type="ContactFlow",
            description=(
                f"None of the {len(instance.contact_flows)} contact flow(s) "
                "analyzed have a voice prompt that speaks a caller- or "
                f"external-system-sourced value.{system_attr_note} Every "
                "prompt either uses a hard-coded string, or its only "
                "dynamic reference is a Connect system attribute the caller "
                "cannot influence. Neither is a vector for what this check "
                "looks for: `</speak><speak>` markup smuggled into an "
                "SSML-interpreted prompt, or attacker-authored text spoken to "
                "an agent as though the flow wrote it."
            ),
            evidence={
                "flows_analyzed": len(instance.contact_flows),
                "system_attribute_prompts_excluded": system_attr_refs_seen,
            },
        )


class LambdaResponseValidationCheck(BaseCheck):
    """Detect Lambda returns used for branching without validation (Req 21)."""

    def __init__(self):
        super().__init__(
            check_id="sec-lambda-validation-001",
            name="Lambda Response Validation",
            pillar=Pillar.SECURITY,
            severity=Severity.MEDIUM,
            description=(
                "Detects contact flows that branch on Lambda return values "
                "without validating the response shape or providing a default."
            ),
        )

    def execute(self, context: CheckContext):
        instance = context.instance
        flagged = []

        for flow in instance.contact_flows:
            graph = _parse_flow(flow)
            if not graph:
                continue
            for action in graph.actions.values():
                if action.action_type != "InvokeLambdaFunction":
                    continue
                # Check if action's transitions include conditions (branching
                # on the return). If conditions exist but no default/error path,
                # it's unvalidated branching.
                cond_targets = [t for t in action.transitions if t.transition_type == "condition"]
                has_default = any(t.transition_type == "default" for t in action.transitions)
                if cond_targets and not has_default:
                    flagged.append(
                        {
                            "flow": flow.name,
                            "flow_id": flow.id,
                            "action_id": action.action_id,
                            "lambda_arn": action.parameters.get("FunctionArn", "unknown"),
                        }
                    )

        if flagged:
            return self.create_finding(
                status=CheckStatus.FAIL,
                resource_id=instance.instance_id,
                resource_type="ContactFlow",
                description=(
                    f"{len(flagged)} Lambda invocation(s) branch on return "
                    "values without a default fallback path."
                ),
                evidence={"flagged_lambdas": flagged},
                structured_remediation=Remediation(
                    summary="Add default/fallback branches after Lambda invocations.",
                    target_resources=[f["action_id"] for f in flagged],
                    steps=[
                        RemediationStep(
                            order=1,
                            instruction=(
                                "For each flagged Lambda action, add a 'Default' "
                                "transition that handles unexpected return values "
                                "safely (e.g., route to an error prompt or retry)."
                            ),
                        ),
                    ],
                    applies_if="Lambda functions may return unexpected data.",
                ),
            )

        return self.create_finding(
            status=CheckStatus.PASS,
            resource_id=instance.instance_id,
            resource_type="ContactFlow",
            description="Lambda branching includes default paths.",
            evidence={"flows_analyzed": len(instance.contact_flows)},
        )


class ExternalTransferTollFraudCheck(BaseCheck):
    """Detect dynamic phone-number transfers (toll fraud risk, Req 22)."""

    def __init__(self):
        super().__init__(
            check_id="sec-toll-fraud-001",
            name="External Transfer Toll Fraud Risk",
            pillar=Pillar.SECURITY,
            severity=Severity.CRITICAL,
            description=(
                "Detects contact flows that transfer calls to dynamically "
                "determined phone numbers without a validation step, "
                "exposing the instance to toll fraud."
            ),
        )

    def execute(self, context: CheckContext):
        instance = context.instance
        flagged = []
        static_count = 0

        for flow in instance.contact_flows:
            graph = _parse_flow(flow)
            if not graph:
                continue
            for action in graph.actions.values():
                if action.action_type not in _PHONE_TRANSFER_TYPES:
                    continue
                dest = _get_phone_destination(action)
                if dest and _is_dynamic_reference(dest):
                    flagged.append(
                        {
                            "flow": flow.name,
                            "flow_id": flow.id,
                            "action_id": action.action_id,
                            "dynamic_source": dest,
                        }
                    )
                else:
                    static_count += 1

        if flagged:
            return self.create_finding(
                status=CheckStatus.FAIL,
                resource_id=instance.instance_id,
                resource_type="ContactFlow",
                description=(
                    f"{len(flagged)} external transfer(s) use a dynamic phone "
                    "number without validation — toll fraud risk."
                ),
                evidence={
                    "dynamic_transfers": flagged,
                    "static_transfers": static_count,
                },
                structured_remediation=Remediation(
                    summary="Constrain dynamic transfer destinations to an allowlist.",
                    target_resources=[f["action_id"] for f in flagged],
                    steps=[
                        RemediationStep(
                            order=1,
                            instruction=(
                                "Add a Check Attribute or Lambda validation "
                                "action before the transfer that confirms the "
                                "destination number is on a pre-approved list."
                            ),
                        ),
                        RemediationStep(
                            order=2,
                            instruction=(
                                "Alternatively, replace the dynamic reference "
                                "with a static, hardcoded number for each "
                                "known destination."
                            ),
                        ),
                    ],
                    references=[
                        RemediationReference(
                            title="Transfer contacts to a phone number",
                            url="https://docs.aws.amazon.com/connect/latest/adminguide/transfer-to-phone-number.html",  # noqa: E501
                        )
                    ],
                ),
            )

        return self.create_finding(
            status=CheckStatus.PASS,
            resource_id=instance.instance_id,
            resource_type="ContactFlow",
            description=(f"All {static_count} external transfer(s) use static phone numbers."),
            evidence={"static_transfers": static_count},
        )


class SensitiveDataInAttributesCheck(BaseCheck):
    """Detect sensitive data stored in contact attributes (Req 23)."""

    def __init__(self):
        super().__init__(
            check_id="sec-sensitive-data-001",
            name="Sensitive Data in Contact Attributes",
            pillar=Pillar.SECURITY,
            severity=Severity.HIGH,
            description=(
                "Detects contact flows that store potentially sensitive data "
                "(PII, credentials) in contact attributes, which are visible "
                "in CTRs, logs, and reporting."
            ),
        )

    def execute(self, context: CheckContext):
        instance = context.instance
        flagged = []

        for flow in instance.contact_flows:
            graph = _parse_flow(flow)
            if not graph:
                continue
            for action in graph.actions.values():
                if action.action_type not in (
                    "UpdateContactAttributes",
                    "SetContactAttributes",
                ):
                    continue
                attrs = action.parameters.get("Attributes", {})
                if isinstance(attrs, dict):
                    for key in attrs:
                        if any(p in key.lower() for p in _SENSITIVE_ATTR_PATTERNS):
                            flagged.append(
                                {
                                    "flow": flow.name,
                                    "flow_id": flow.id,
                                    "action_id": action.action_id,
                                    "attribute_name": key,
                                }
                            )

        if flagged:
            # Group flagged items by flow so the reader can jump to each
            # flow once rather than scanning a de-duplicated action list.
            by_flow: Dict[str, List[Dict[str, str]]] = {}
            for f in flagged:
                by_flow.setdefault(f["flow"], []).append(f)

            flow_lines = []
            for flow_name, entries in list(by_flow.items())[:5]:
                attr_names = ", ".join(sorted({f"`{e['attribute_name']}`" for e in entries}))
                flow_lines.append(f"* `{flow_name}` — sets: {attr_names}")
            more_note = (
                f"\n\n_+ {len(by_flow) - 5} more flow(s) with flagged attributes; see JSON export._"
                if len(by_flow) > 5
                else ""
            )

            return self.create_finding(
                status=CheckStatus.FAIL,
                resource_id=instance.instance_id,
                resource_type="ContactFlow",
                description=(
                    f"**{len(flagged)} `Set contact attributes` action(s) in "
                    f"{len(by_flow)} flow(s) store data under names that "
                    "look like PII or credentials.**\n\n"
                    "The attribute name is the giveaway — this check "
                    "watches for keys like `ssn`, `dob`, `creditcard`, "
                    "`cvv`, `pin`, `password`, `accountnumber`, "
                    "`bankaccount`, `taxid`, `passportnumber` (full list in "
                    "the source). If a Connect flow writes those names into "
                    "contact attributes, the values end up in three places "
                    "you probably don't want them:\n\n"
                    "1. **Contact Trace Records (CTRs)** — attributes are "
                    "part of the CTR JSON exported to your Kinesis or S3 "
                    "stream after every contact. Anyone with read on that "
                    "bucket sees the raw value.\n"
                    "2. **Agent workspace** — supervisors and agents with "
                    "'View contact record' permission can see attributes "
                    "on a completed contact.\n"
                    "3. **Flow logs / CloudWatch** — if flow logging is "
                    "enabled, every attribute change writes a log line "
                    "containing the value.\n\n"
                    "**What the check flagged:**\n\n"
                    f"{chr(10).join(flow_lines)}{more_note}\n\n"
                    "**Fix (per attribute):** keep the *reference*, drop "
                    "the *value*. In the flow, replace:\n\n"
                    "```\n"
                    "Set contact attribute:  ssn        = <raw 9-digit value>\n"
                    "```\n\n"
                    "with:\n\n"
                    "```\n"
                    "Set contact attribute:  ssn_last4  = <last 4 digits only>       # safe to voice/log\n"
                    "Set contact attribute:  customer_token = <Lambda-returned UUID>  # opaque handle\n"
                    "```\n\n"
                    "Store the full value in Amazon Connect Customer "
                    "Profiles (encrypted at rest, access-scoped) and "
                    "resolve it via Lambda only when a specific step needs "
                    "the full number. The attribute in the flow then "
                    "carries only the token — logs and CTRs stay clean."
                ),
                evidence={"flagged_attributes": flagged},
                structured_remediation=Remediation(
                    summary=(
                        "Replace raw-PII contact attributes with tokenized "
                        "references; resolve the full value via Lambda "
                        "only when a step needs it."
                    ),
                    target_resources=[f["action_id"] for f in flagged],
                    steps=[
                        RemediationStep(
                            order=1,
                            instruction=(
                                "For each flagged attribute, decide the "
                                "smallest form the flow actually needs: "
                                "last-4 digits for confirmation prompts, "
                                "an opaque token for downstream Lambdas, "
                                "or nothing at all if the value was "
                                "written but never read."
                            ),
                        ),
                        RemediationStep(
                            order=2,
                            instruction=(
                                "Move the full value into Amazon Connect "
                                "Customer Profiles (or your own encrypted "
                                "store) keyed by a UUID. Update the flow "
                                "to store only the UUID as a contact "
                                "attribute; write a small Lambda that "
                                "returns the full value on demand for the "
                                "one or two blocks that need it."
                            ),
                            console_path=("Connect console -> Customer Profiles"),
                        ),
                        RemediationStep(
                            order=3,
                            instruction=(
                                "If you can't avoid attributes at all, at "
                                "least enable Contact Lens sensitive-data "
                                "redaction so the values are scrubbed "
                                "before CTRs and recordings are exported."
                            ),
                            console_path=("Connect console -> Analytics -> Contact Lens settings"),
                        ),
                    ],
                    references=[
                        RemediationReference(
                            title="Customer Profiles for Amazon Connect",
                            url="https://docs.aws.amazon.com/connect/latest/adminguide/customer-profiles.html",  # noqa: E501
                        ),
                        RemediationReference(
                            title="Contact Lens sensitive data redaction",
                            url="https://docs.aws.amazon.com/connect/latest/adminguide/sensitive-data-redaction.html",  # noqa: E501
                        ),
                    ],
                ),
            )

        return self.create_finding(
            status=CheckStatus.PASS,
            resource_id=instance.instance_id,
            resource_type="ContactFlow",
            description=(
                f"None of the {len(instance.contact_flows)} flow(s) "
                "analyzed store data under attribute names that look like "
                "PII or credentials (ssn, dob, creditcard, cvv, pin, "
                "accountnumber, etc.). If PII passes through a flow at "
                "all, this pattern keeps it out of CTRs and flow logs — "
                "which is where accidental exposure usually happens."
            ),
            evidence={"flows_analyzed": len(instance.contact_flows)},
        )


class PIIInPromptsCheck(BaseCheck):
    """Detect PII read back in voice prompts without masking (Req 39)."""

    def __init__(self):
        super().__init__(
            check_id="sec-pii-prompts-001",
            name="PII Exposure in Voice Prompts",
            pillar=Pillar.SECURITY,
            severity=Severity.HIGH,
            description=(
                "Detects contact flows that read back sensitive customer data "
                "(account numbers, SSN, etc.) in voice prompts without masking."
            ),
        )

    def execute(self, context: CheckContext):
        instance = context.instance
        flagged = []

        for flow in instance.contact_flows:
            graph = _parse_flow(flow)
            if not graph:
                continue
            for action in graph.actions.values():
                if action.action_type not in ("MessageParticipant", "PlayPrompt"):
                    continue
                raw_text = str(action.parameters.get("Text", ""))
                text = raw_text.lower()
                for pattern in _SENSITIVE_ATTR_PATTERNS:
                    if pattern in text:
                        # Check if masking is applied (heuristic: "last4",
                        # "ending in", "substring" in same text).
                        has_mask = any(
                            m in text
                            for m in (
                                "last4",
                                "lastfour",
                                "ending in",
                                "substring",
                                "mask",
                                "redact",
                            )
                        )
                        if not has_mask:
                            # Preserve a truncated copy of the raw prompt
                            # text so the finding can show the reader
                            # what actually got flagged.
                            excerpt = raw_text.replace("\n", " ").strip()
                            if len(excerpt) > 140:
                                excerpt = excerpt[:137] + "\u2026"
                            flagged.append(
                                {
                                    "flow": flow.name,
                                    "flow_id": flow.id,
                                    "action_id": action.action_id,
                                    "attribute_pattern": pattern,
                                    "prompt_text": excerpt,
                                }
                            )
                        break  # one flag per action is enough

        if flagged:
            worst_lines = []
            for f in flagged[:3]:
                worst_lines.append(
                    f"* `{f['flow']}` \u2192 matches `{f['attribute_pattern']}` "
                    f"in prompt: \u201c{f['prompt_text']}\u201d"
                )
            more_note = (
                f"\n\n_+ {len(flagged) - 3} more prompt(s); see JSON export._"
                if len(flagged) > 3
                else ""
            )

            return self.create_finding(
                status=CheckStatus.FAIL,
                resource_id=instance.instance_id,
                resource_type="ContactFlow",
                description=(
                    f"**{len(flagged)} voice prompt(s) speak a piece of "
                    "sensitive data back to the caller without masking "
                    "it.**\n\n"
                    "The check reads the `Text` parameter of every "
                    "`Play prompt` / `Message participant` action, looks "
                    "for references to values that sound like PII "
                    "(`ssn`, `dob`, `creditcard`, `accountnumber`, "
                    "`taxid`, `passportnumber`, and similar), and checks "
                    "whether the prompt also contains a masking hint "
                    "nearby (`last4`, `ending in`, `substring`, `mask`, "
                    "`redact`). If the sensitive reference is present but "
                    "no masking hint is, the prompt gets flagged.\n\n"
                    "**Unmasked vs masked, side by side:**\n\n"
                    "```\n"
                    '\u274c  Play prompt: "Your account number is $.Attributes.AccountNumber."\n'
                    "         \u2514\u2500 caller hears all 12 digits, anyone nearby hears them too.\n"
                    "\n"
                    '\u2705  Play prompt: "Your account ending in $.Attributes.AccountNumberLast4."\n'
                    "         \u2514\u2500 last 4 digits only, enough for the caller to recognize.\n"
                    "```\n\n"
                    "**Why this matters.** Callers phone from open "
                    "offices, cars, and public spaces. Whatever the "
                    "prompt says gets heard by anyone in earshot AND is "
                    "captured verbatim in the call recording. Recordings "
                    "sit in S3 for weeks or years. If a support team, "
                    "auditor, or breached recording bucket touches the "
                    "recordings later, the PII is right there in the "
                    "audio.\n\n"
                    "**What the check flagged:**\n\n"
                    f"{chr(10).join(worst_lines)}{more_note}\n\n"
                    "**Fix (per prompt):** replace the full attribute "
                    "reference with a `*Last4` variant, or a "
                    "confirmation pattern that doesn't voice the value "
                    "at all (\u201cThe account ending in 4 3 2 1, is that "
                    "correct?\u201d). Compute the last-4 attribute with a "
                    "small `Set contact attributes` block upstream of "
                    "the prompt. Additionally, turn on **Contact Lens "
                    "sensitive-data redaction** so if a caller says the "
                    "full number back, it's redacted from the recording "
                    "and transcript."
                ),
                evidence={"flagged_prompts": flagged},
                structured_remediation=Remediation(
                    summary=(
                        "Voice only the last 4 digits (or a tokenized "
                        "reference) instead of the full value, and turn "
                        "on Contact Lens redaction as a safety net."
                    ),
                    target_resources=[f["action_id"] for f in flagged],
                    steps=[
                        RemediationStep(
                            order=1,
                            instruction=(
                                "For each flagged prompt, compute a "
                                "last-4 attribute upstream (Set contact "
                                "attributes \u2192 "
                                "`AccountNumberLast4 = $.Attributes.AccountNumber` "
                                "with a substring transform), then edit "
                                "the prompt to reference the last-4 "
                                "attribute instead of the full one."
                            ),
                            console_path="Connect console -> Routing -> Flows",
                        ),
                        RemediationStep(
                            order=2,
                            instruction=(
                                "Enable Contact Lens sensitive-data "
                                "redaction on the instance. This scrubs "
                                "numeric PII patterns (account numbers, "
                                "SSNs, credit cards) from call "
                                "recordings and transcripts before they "
                                "land in S3."
                            ),
                            console_path=("Connect console -> Analytics -> Contact Lens settings"),
                        ),
                    ],
                    references=[
                        RemediationReference(
                            title="Contact Lens sensitive data redaction",
                            url="https://docs.aws.amazon.com/connect/latest/adminguide/sensitive-data-redaction.html",  # noqa: E501
                        )
                    ],
                    applies_if=("prompts include values that identify or authenticate a customer."),
                ),
            )

        return self.create_finding(
            status=CheckStatus.PASS,
            resource_id=instance.instance_id,
            resource_type="ContactFlow",
            description=(
                f"None of the {len(instance.contact_flows)} flow(s) "
                "analyzed voice sensitive attributes (ssn, dob, "
                "creditcard, accountnumber, etc.) to callers without a "
                "masking hint nearby (`last4`, `ending in`, `substring`, "
                "`mask`, `redact`). This is the pattern that keeps PII "
                "out of call recordings and stops passers-by in the "
                "caller's environment from overhearing account numbers."
            ),
            evidence={"flows_analyzed": len(instance.contact_flows)},
        )


def register_contact_flow_security_checks(registry) -> None:
    """Register all contact-flow security checks."""
    registry.register_check(DynamicPromptInjectionCheck())
    registry.register_check(LambdaResponseValidationCheck())
    registry.register_check(ExternalTransferTollFraudCheck())
    registry.register_check(SensitiveDataInAttributesCheck())
    registry.register_check(PIIInPromptsCheck())
