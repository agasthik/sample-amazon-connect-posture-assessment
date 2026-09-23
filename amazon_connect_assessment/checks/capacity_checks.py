"""
Service-quota and capacity headroom checks.

- res-quota-config-001   : Configuration-object counts vs. their per-instance quotas
- res-quota-headroom-001 : Peak concurrent calls vs. the concurrent-calls quota
- res-quota-growth-001   : Call-volume growth trend vs. projected time-to-quota

Pillar note: these are capacity checks, but they are registered under
``Pillar.RESILIENCE`` rather than as a separate capacity pillar, because
Well-Architected already owns this ground — REL01 is "Manage Service Quotas and
Constraints", and REL01-BP06 is specifically "ensure sufficient gap between
quota and maximum usage". Filing them anywhere else would misrepresent the
framework the tool claims to assess against.

Why these are worth checking at all: Amazon Connect enforces hard per-instance
limits, and none of them are surfaced in the Connect console. Hitting one is
not a cost inefficiency, it is an outage — callers get busy signals once
concurrent calls are capped, and administrators cannot create users, queues, or
flows once those ceilings are reached. The failure mode is also seasonal, so an
instance that looks healthy in March can breach in November.

All three checks degrade to SKIPPED on AccessDenied and emit structured
remediation.
"""

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    CheckStatus,
    Pillar,
    Remediation,
    RemediationReference,
    RemediationStep,
    Severity,
)
from .base import BaseCheck, CheckContext

# Amazon Connect's Service Quotas service code.
_QUOTA_SERVICE_CODE = "connect"

# Utilization at or above this percentage is reported as a finding. 80% is the
# conventional headroom alarm point and leaves room to either request an
# increase (which is not instantaneous) or shed load before the ceiling.
_UTILIZATION_WARN_PCT = 80.0

# Above this, the severity is raised: there is no longer enough headroom to
# absorb a normal seasonal peak while an increase request is processed.
_UTILIZATION_CRITICAL_PCT = 95.0

# Bounded pagination, matching the convention used elsewhere in the package.
_QUOTA_PAGE_LIMIT = 20
_QUOTA_PAGE_SIZE = 100
_PHONE_NUMBER_PAGE_LIMIT = 20
_PHONE_NUMBER_PAGE_SIZE = 100
_USER_PAGE_LIMIT = 20
_USER_PAGE_SIZE = 100

# Amazon Connect publishes ConcurrentCalls under *both* of these dimensions, and
# CloudWatch keys every metric on its complete dimension set. A query naming
# InstanceId alone therefore matches no metric and returns an empty datapoint
# list — which these checks would read as "this instance carried no call
# traffic" on an instance that is in fact busy.
_CONCURRENT_CALLS_METRIC_GROUP = "VoiceCalls"

# Metric window for the headroom check.
_HEADROOM_LOOKBACK_DAYS = 30

# Metric window for the growth-trend check, and the projection horizon that
# counts as urgent. 90 days is the shortest window that yields enough weekly
# points for a trend to mean anything; 26 weeks (two quarters) is the horizon
# because a Service Quotas increase for Connect is a support-case workflow,
# not a console toggle, so a quarter of lead time is the practical minimum.
_GROWTH_LOOKBACK_DAYS = 90
_GROWTH_HORIZON_WEEKS = 26
_MIN_GROWTH_DATA_POINTS = 4

_QUOTA_DOC_URL = (
    "https://docs.aws.amazon.com/connect/latest/adminguide/amazon-connect-service-limits.html"
)
_QUOTA_INCREASE_URL = (
    "https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html"
)


class QuotaLookupDenied(Exception):
    """Raised when every Service Quotas read is denied, so checks can SKIP."""


@dataclass(frozen=True)
class _QuotaSubject:
    """A countable Connect resource and the quota name that bounds it."""

    key: str
    label: str
    keywords: Tuple[str, ...]


# Matched on quota *name* rather than quota code because the codes are opaque
# (``L-...``) and are not documented per resource, while the names are stable,
# human-readable, and worth showing in evidence so a reader can verify which
# quota a percentage was computed against.
_CONFIG_SUBJECTS: Tuple[_QuotaSubject, ...] = (
    _QuotaSubject("users", "Users", ("users", "per instance")),
    _QuotaSubject("queues", "Queues", ("queues", "per instance")),
    _QuotaSubject("routing_profiles", "Routing profiles", ("routing profiles", "per instance")),
    _QuotaSubject("security_profiles", "Security profiles", ("security profiles", "per instance")),
    _QuotaSubject("flows", "Flows", ("flows", "per instance")),
    _QuotaSubject("phone_numbers", "Phone numbers", ("phone numbers", "per instance")),
)

_CONCURRENT_CALLS_KEYWORDS: Tuple[str, ...] = ("concurrent", "calls", "per instance")


# ---------------------------------------------------------------------------
# Quota lookup, cached per region
#
# Every check in this module needs the same quota table, and the checks run in
# parallel, so the table is fetched once and shared. Without the cache a
# three-check run makes the same paginated Service Quotas calls three times.
# ---------------------------------------------------------------------------
_QUOTA_CACHE: Dict[str, Dict[str, float]] = {}
_QUOTA_CACHE_LOCK = threading.Lock()


def _list_quotas(factory: Any, operation: str) -> Dict[str, float]:
    """Paginate one Service Quotas list operation into ``{quota name: value}``."""
    client = factory.get_client("service-quotas")
    quotas: Dict[str, float] = {}
    next_token: Optional[str] = None

    for _ in range(_QUOTA_PAGE_LIMIT):
        kwargs: Dict[str, Any] = {
            "ServiceCode": _QUOTA_SERVICE_CODE,
            "MaxResults": _QUOTA_PAGE_SIZE,
        }
        if next_token:
            kwargs["NextToken"] = next_token

        response = factory.call_api_with_resilience(client, operation, "service-quotas", **kwargs)
        for quota in response.get("Quotas") or []:
            name = quota.get("QuotaName")
            value = quota.get("Value")
            if name and isinstance(value, (int, float)):
                quotas[name] = float(value)

        next_token = response.get("NextToken")
        if not next_token:
            break

    return quotas


def get_connect_quotas(factory: Any) -> Dict[str, float]:
    """
    Return every Connect quota for the assessed region as ``{name: value}``.

    Applied quotas are overlaid on AWS defaults. Both calls are made because
    an applied quota only exists once a customer has requested an increase —
    reading applied quotas alone returns an empty table for an instance still
    on defaults, which is exactly the population most likely to be near a
    ceiling.

    Raises:
        QuotaLookupDenied: if both reads are denied, so the caller can emit a
            SKIPPED finding naming the missing permission rather than a
            misleading PASS.
    """
    region = getattr(factory, "region", "default")
    if region in _QUOTA_CACHE:
        return _QUOTA_CACHE[region]

    with _QUOTA_CACHE_LOCK:
        if region in _QUOTA_CACHE:
            return _QUOTA_CACHE[region]

        merged: Dict[str, float] = {}
        denied = 0
        for operation in ("list_aws_default_service_quotas", "list_service_quotas"):
            try:
                merged.update(_list_quotas(factory, operation))
            except Exception as exc:
                if factory.is_access_denied(exc):
                    denied += 1
                    continue
                raise

        if denied == 2:
            raise QuotaLookupDenied("Service Quotas reads denied")

        _QUOTA_CACHE[region] = merged
        return merged


def reset_quota_cache() -> None:
    """Clear the per-region quota cache. Exposed for tests."""
    with _QUOTA_CACHE_LOCK:
        _QUOTA_CACHE.clear()


def _match_quota(
    quotas: Dict[str, float], keywords: Tuple[str, ...]
) -> Optional[Tuple[str, float]]:
    """
    Return the ``(name, value)`` of the quota whose name contains every keyword.

    When several names match, the shortest wins: AWS names the base quota most
    simply ("Queues per instance") and derives longer names for narrower
    variants ("Queues per routing profile per instance"), so the shortest match
    is the instance-wide ceiling we want.
    """
    matches = sorted(
        (name for name in quotas if all(k in name.lower() for k in keywords)),
        key=lambda name: (len(name), name),
    )
    if not matches:
        return None
    return matches[0], quotas[matches[0]]


def _utilization_pct(count: float, quota: float) -> Optional[float]:
    """Percentage of a quota consumed, or None when the quota is unusable."""
    if quota <= 0:
        return None
    return round(count / quota * 100, 1)


def _severity_for(worst_pct: float) -> Severity:
    """Escalate severity once there is no room to absorb a seasonal peak."""
    return Severity.HIGH if worst_pct >= _UTILIZATION_CRITICAL_PCT else Severity.MEDIUM


def _quota_references() -> List[RemediationReference]:
    """Shared documentation links for every check in this module."""
    return [
        RemediationReference(
            title="Amazon Connect service quotas",
            url=_QUOTA_DOC_URL,
        ),
        RemediationReference(
            title="Requesting a quota increase",
            url=_QUOTA_INCREASE_URL,
        ),
    ]


def _daily_peaks(
    factory: Any, instance_id: str, lookback_days: int
) -> List[Tuple[datetime, float]]:
    """
    Return ``(timestamp, daily peak concurrent calls)`` ordered oldest first.

    Sampled at a one-day period and aggregated locally rather than asking
    CloudWatch for weekly buckets, because GetMetricStatistics caps the period
    at one day. Raises on error so callers can distinguish AccessDenied from
    "this instance took no calls".
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    response = factory.call_api_with_resilience(
        factory.get_cloudwatch_client(),
        "get_metric_statistics",
        "cloudwatch",
        Namespace="AWS/Connect",
        MetricName="ConcurrentCalls",
        Dimensions=[
            {"Name": "InstanceId", "Value": instance_id},
            {"Name": "MetricGroup", "Value": _CONCURRENT_CALLS_METRIC_GROUP},
        ],
        StartTime=start,
        EndTime=end,
        Period=86400,
        Statistics=["Maximum"],
    )

    points = [
        (point["Timestamp"], float(point["Maximum"]))
        for point in response.get("Datapoints") or []
        if point.get("Timestamp") is not None and point.get("Maximum") is not None
    ]
    points.sort(key=lambda item: item[0])
    return points


def _weekly_peaks(points: List[Tuple[datetime, float]]) -> List[Tuple[int, float]]:
    """
    Collapse daily peaks into ``(week index, peak)`` pairs, oldest first.

    The week index is carried rather than discarded because the buckets are not
    necessarily consecutive — CloudWatch returns no datapoint for a week the
    instance took no calls, so a 90-day window can contain gaps. Regressing
    against list position instead of elapsed weeks would compress each gap into
    a single step and overstate the growth rate by the size of the gap.
    """
    if not points:
        return []
    origin = points[0][0]
    buckets: Dict[int, float] = {}
    for timestamp, value in points:
        week = (timestamp - origin).days // 7
        buckets[week] = max(buckets.get(week, value), value)
    return [(week, buckets[week]) for week in sorted(buckets)]


def _linear_slope(points: List[Tuple[int, float]]) -> float:
    """
    Least-squares slope of weekly peak against elapsed week index.

    Used instead of comparing first and last points so a single anomalous week
    (an outage, a marketing spike) cannot by itself manufacture or erase a
    trend. Weeks with no datapoint are absent from ``points`` rather than
    present as zero: CloudWatch reporting nothing is not the same claim as the
    instance having taken no calls, and zero-filling would drag the fitted slope
    toward whichever side of the window the gap falls on.
    """
    n = len(points)
    if n == 0:
        return 0.0
    mean_x = sum(week for week, _ in points) / n
    mean_y = sum(value for _, value in points) / n
    numerator = sum((week - mean_x) * (value - mean_y) for week, value in points)
    denominator = sum((week - mean_x) ** 2 for week, _ in points)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _list_phone_number_count(factory: Any, target_arn: str) -> int:
    """
    Count every phone number claimed to an instance.

    Paginated locally, following the same convention as the engine and the
    advanced resilience checks: each call site bounds its own page pull rather
    than depending on a shared helper.
    """
    total = 0
    next_token: Optional[str] = None
    for _ in range(_PHONE_NUMBER_PAGE_LIMIT):
        kwargs: Dict[str, Any] = {
            "TargetArn": target_arn,
            "MaxResults": _PHONE_NUMBER_PAGE_SIZE,
        }
        if next_token:
            kwargs["NextToken"] = next_token
        response = factory.call_api_with_resilience(
            factory.get_connect_client(),
            "list_phone_numbers_v2",
            "connect",
            **kwargs,
        )
        total += len(response.get("ListPhoneNumbersSummaryList") or [])
        next_token = response.get("NextToken")
        if not next_token:
            break
    return total


def _list_user_count(factory: Any, instance_id: str) -> int:
    """
    Count every user configured on an instance.

    Counted from the API here rather than read off ``ConnectInstance.users``
    because no analyzer in the assessment pipeline calls ListUsers — that
    collection is empty on every ordinary run, so deriving the count from it
    reported the users quota as unmeasured for every customer. ``connect:
    ListUsers`` is already part of the granted read set.
    """
    total = 0
    next_token: Optional[str] = None
    for _ in range(_USER_PAGE_LIMIT):
        kwargs: Dict[str, Any] = {
            "InstanceId": instance_id,
            "MaxResults": _USER_PAGE_SIZE,
        }
        if next_token:
            kwargs["NextToken"] = next_token
        response = factory.call_api_with_resilience(
            factory.get_connect_client(),
            "list_users",
            "connect",
            **kwargs,
        )
        total += len(response.get("UserSummaryList") or [])
        next_token = response.get("NextToken")
        if not next_token:
            break
    return total


class ConfigurationQuotaUtilizationCheck(BaseCheck):
    """Compare configuration-object counts against their per-instance quotas."""

    def __init__(self):
        super().__init__(
            check_id="res-quota-config-001",
            name="Configuration Object Quota Utilization",
            pillar=Pillar.RESILIENCE,
            severity=Severity.MEDIUM,
            description=(
                "Compares the number of users, queues, routing profiles, security "
                "profiles, flows, and claimed phone numbers against the Service "
                "Quotas ceiling for each, and reports any that are close enough to "
                "the limit that further growth would be blocked."
            ),
        )

    def _measured_counts(self, context: CheckContext) -> Tuple[Dict[str, int], List[str]]:
        """
        Return countable resources plus the keys that could not be measured.

        An empty collection is treated as unmeasured rather than as a count of
        zero. Every live Connect instance has at least one user, queue, routing
        profile, and security profile, so an empty list means discovery was
        denied or skipped — and reporting 0% utilization in that case would be a
        false PASS on precisely the instance whose data is missing.

        Users and phone numbers are counted directly from their list APIs, both
        of which this check owns because no analyzer collects them.
        """
        instance = context.instance
        factory = context.aws_client_factory
        candidates = {
            "queues": len(instance.queues),
            "routing_profiles": len(instance.routing_profiles),
            "security_profiles": len(instance.security_profiles),
            "flows": len(instance.contact_flows),
        }
        counts = {key: value for key, value in candidates.items() if value > 0}
        unmeasured = [key for key, value in candidates.items() if value == 0]

        # Each API-backed subject is counted in its own try block so a denial on
        # one does not discard the other, or the four subjects already measured
        # from collected data. Unlike an empty collection, a count of zero from a
        # call that succeeded is a real measurement: a new instance can legitimately
        # hold no claimed numbers.
        for key, counter in (
            ("users", lambda: _list_user_count(factory, instance.instance_id)),
            ("phone_numbers", lambda: _list_phone_number_count(factory, instance.instance_arn)),
        ):
            try:
                counts[key] = counter()
            except Exception as exc:
                if not factory.is_access_denied(exc):
                    raise
                unmeasured.append(key)

        return counts, unmeasured

    def execute(self, context: CheckContext):
        instance = context.instance
        factory = context.aws_client_factory

        try:
            quotas = get_connect_quotas(factory)
        except QuotaLookupDenied:
            return self.skipped_for_access_denied(context, "servicequotas:ListServiceQuotas")

        counts, unmeasured = self._measured_counts(context)

        measured: Dict[str, Dict[str, Any]] = {}
        unmatched: List[str] = []
        for subject in _CONFIG_SUBJECTS:
            if subject.key not in counts:
                continue
            match = _match_quota(quotas, subject.keywords)
            if match is None:
                # Service Quotas does not publish a ceiling for this resource in
                # this region, or has renamed it. Recorded rather than silently
                # dropped so the gap is visible to the reader.
                unmatched.append(subject.key)
                continue
            quota_name, quota_value = match
            pct = _utilization_pct(counts[subject.key], quota_value)
            if pct is None:
                unmatched.append(subject.key)
                continue
            measured[subject.key] = {
                "label": subject.label,
                "count": counts[subject.key],
                "quota_name": quota_name,
                "quota_value": quota_value,
                "utilization_pct": pct,
            }

        evidence: Dict[str, Any] = {
            "instance_alias": instance.instance_alias,
            "warn_threshold_pct": _UTILIZATION_WARN_PCT,
            "measured": measured,
            "unmeasured_resources": sorted(unmeasured),
            "quotas_without_published_limit": sorted(unmatched),
        }

        if not measured:
            return self.create_finding(
                status=CheckStatus.SKIPPED,
                resource_id=instance.instance_id,
                resource_type="ConnectInstance",
                description=(
                    "Skipped: no configuration-object count could be compared against a "
                    "quota. This usually means Connect discovery was denied or run with "
                    "--skip-flow-analysis, or that Service Quotas published no matching "
                    "per-instance limits for this region."
                ),
                evidence=evidence,
            )

        breaches = {
            key: data
            for key, data in measured.items()
            if data["utilization_pct"] >= _UTILIZATION_WARN_PCT
        }

        if not breaches:
            worst = max(measured.values(), key=lambda data: data["utilization_pct"])
            return self.create_finding(
                status=CheckStatus.PASS,
                resource_id=instance.instance_id,
                resource_type="ConnectInstance",
                description=(
                    f"All {len(measured)} measurable configuration quotas for instance "
                    f"{instance.display_name} have headroom. The closest is "
                    f"{worst['label'].lower()} at {worst['utilization_pct']}% of "
                    f"{int(worst['quota_value'])}."
                ),
                evidence=evidence,
            )

        worst_pct = max(data["utilization_pct"] for data in breaches.values())
        detail = ", ".join(
            f"{data['label'].lower()} at {data['utilization_pct']}% "
            f"({data['count']} of {int(data['quota_value'])})"
            for data in sorted(breaches.values(), key=lambda d: -d["utilization_pct"])
        )

        return self.create_finding(
            status=CheckStatus.FAIL,
            resource_id=instance.instance_id,
            resource_type="ConnectInstance",
            severity=_severity_for(worst_pct),
            description=(
                f"Instance {instance.display_name} is within "
                f"{100 - worst_pct:.1f}% of a hard Connect quota: {detail}. Once a "
                "ceiling is reached, administrators can no longer create that resource "
                "type, which blocks routine changes and onboarding."
            ),
            evidence=evidence,
            structured_remediation=Remediation(
                summary=(
                    "Request quota increases for the resources listed below before they "
                    "reach their ceiling, and remove objects that are no longer in use."
                ),
                steps=[
                    RemediationStep(
                        order=1,
                        instruction=(
                            "Review the current quota values for Amazon Connect in this region."
                        ),
                        console_path="Service Quotas > AWS services > Amazon Connect",
                        command=("aws service-quotas list-service-quotas --service-code connect"),
                    ),
                    RemediationStep(
                        order=2,
                        instruction=(
                            "Request an increase for each quota above "
                            f"{_UTILIZATION_WARN_PCT:.0f}% utilization. Increases are "
                            "processed as support cases, so raise them before the "
                            "ceiling is reached rather than at it."
                        ),
                        console_path=(
                            "Service Quotas > AWS services > Amazon Connect > "
                            "select quota > Request increase at account level"
                        ),
                    ),
                    RemediationStep(
                        order=3,
                        instruction=(
                            "Reclaim headroom by deleting unused objects — disabled "
                            "users, empty queues, and superseded flow versions all "
                            "count against these quotas."
                        ),
                    ),
                ],
                target_resources=[data["quota_name"] for data in breaches.values()],
                references=_quota_references(),
            ),
        )


class ConcurrentCallsHeadroomCheck(BaseCheck):
    """Compare peak concurrent calls against the concurrent-calls quota."""

    def __init__(self):
        super().__init__(
            check_id="res-quota-headroom-001",
            name="Concurrent Calls Quota Headroom",
            pillar=Pillar.RESILIENCE,
            severity=Severity.HIGH,
            description=(
                "Compares the peak concurrent calls observed over the last 30 days "
                "against the instance's concurrent-active-calls quota. Callers receive "
                "busy signals once this ceiling is reached, so headroom here is a "
                "direct availability control."
            ),
        )

    def execute(self, context: CheckContext):
        instance = context.instance
        factory = context.aws_client_factory

        try:
            quotas = get_connect_quotas(factory)
        except QuotaLookupDenied:
            return self.skipped_for_access_denied(context, "servicequotas:ListServiceQuotas")

        match = _match_quota(quotas, _CONCURRENT_CALLS_KEYWORDS)
        if match is None:
            return self.not_applicable(
                context,
                reason=(
                    "Service Quotas published no concurrent-calls limit for Amazon "
                    "Connect in this region, so no headroom percentage can be computed"
                ),
            )
        quota_name, quota_value = match

        try:
            points = _daily_peaks(factory, instance.instance_id, _HEADROOM_LOOKBACK_DAYS)
        except Exception as exc:
            if factory.is_access_denied(exc):
                return self.skipped_for_access_denied(context, "cloudwatch:GetMetricStatistics")
            raise

        if not points:
            return self.not_applicable(
                context,
                reason=(
                    f"no ConcurrentCalls metric data in the last "
                    f"{_HEADROOM_LOOKBACK_DAYS} days, so this instance carried no "
                    "measurable call traffic"
                ),
                evidence={"quota_name": quota_name, "quota_value": quota_value},
            )

        peak = max(value for _, value in points)
        pct = _utilization_pct(peak, quota_value)
        evidence: Dict[str, Any] = {
            "instance_alias": instance.instance_alias,
            "quota_name": quota_name,
            "quota_value": quota_value,
            "peak_concurrent_calls": peak,
            "utilization_pct": pct,
            "lookback_days": _HEADROOM_LOOKBACK_DAYS,
            "warn_threshold_pct": _UTILIZATION_WARN_PCT,
        }

        if pct is None:
            return self.not_applicable(
                context,
                reason="the published concurrent-calls quota was zero or negative",
                evidence=evidence,
            )

        if pct < _UTILIZATION_WARN_PCT:
            return self.create_finding(
                status=CheckStatus.PASS,
                resource_id=instance.instance_id,
                resource_type="ConnectInstance",
                description=(
                    f"Peak concurrent calls on instance {instance.display_name} reached "
                    f"{int(peak)} over the last {_HEADROOM_LOOKBACK_DAYS} days, "
                    f"{pct}% of the {int(quota_value)}-call quota."
                ),
                evidence=evidence,
            )

        return self.create_finding(
            status=CheckStatus.FAIL,
            resource_id=instance.instance_id,
            resource_type="ConnectInstance",
            severity=Severity.CRITICAL if pct >= _UTILIZATION_CRITICAL_PCT else Severity.HIGH,
            description=(
                f"Peak concurrent calls on instance {instance.display_name} reached "
                f"{int(peak)} over the last {_HEADROOM_LOOKBACK_DAYS} days, which is "
                f"{pct}% of the {int(quota_value)}-call quota. Calls beyond this ceiling "
                "are rejected, so a normal seasonal increase would drop live traffic."
            ),
            evidence=evidence,
            structured_remediation=Remediation(
                summary=(
                    "Request a concurrent-calls quota increase, and alarm on the "
                    "utilization metric so the next approach to the ceiling is caught "
                    "before callers are affected."
                ),
                steps=[
                    RemediationStep(
                        order=1,
                        instruction=(
                            "Request an increase to the concurrent active calls quota, "
                            "sized against your forecast peak rather than your current "
                            "peak."
                        ),
                        console_path=(
                            "Service Quotas > AWS services > Amazon Connect > "
                            f"{quota_name} > Request increase at account level"
                        ),
                        command=("aws service-quotas list-service-quotas --service-code connect"),
                    ),
                    RemediationStep(
                        order=2,
                        instruction=(
                            "Create a CloudWatch alarm on the "
                            "ConcurrentCallsPercentage metric so this is detected "
                            "continuously rather than at the next assessment."
                        ),
                        console_path=(
                            "CloudWatch > Alarms > Create alarm > AWS/Connect > "
                            "ConcurrentCallsPercentage"
                        ),
                    ),
                    RemediationStep(
                        order=3,
                        instruction=(
                            "Reduce peak concurrent demand where possible — queued "
                            "callbacks and self-service containment both lower the "
                            "concurrent-call count for the same contact volume."
                        ),
                    ),
                ],
                target_resources=[quota_name],
                references=_quota_references(),
            ),
        )


class CallVolumeGrowthTrendCheck(BaseCheck):
    """Project when call-volume growth will reach the concurrent-calls quota."""

    def __init__(self):
        super().__init__(
            check_id="res-quota-growth-001",
            name="Call Volume Growth Against Quota",
            pillar=Pillar.RESILIENCE,
            severity=Severity.MEDIUM,
            description=(
                "Fits a trend to weekly peak concurrent calls over the last 90 days "
                "and projects how long the current growth rate leaves before the "
                "concurrent-calls quota is reached."
            ),
        )

    def execute(self, context: CheckContext):
        instance = context.instance
        factory = context.aws_client_factory

        try:
            quotas = get_connect_quotas(factory)
        except QuotaLookupDenied:
            return self.skipped_for_access_denied(context, "servicequotas:ListServiceQuotas")

        match = _match_quota(quotas, _CONCURRENT_CALLS_KEYWORDS)
        if match is None:
            return self.not_applicable(
                context,
                reason=(
                    "Service Quotas published no concurrent-calls limit for Amazon "
                    "Connect in this region, so growth cannot be projected against a "
                    "ceiling"
                ),
            )
        quota_name, quota_value = match

        try:
            points = _daily_peaks(factory, instance.instance_id, _GROWTH_LOOKBACK_DAYS)
        except Exception as exc:
            if factory.is_access_denied(exc):
                return self.skipped_for_access_denied(context, "cloudwatch:GetMetricStatistics")
            raise

        weekly = _weekly_peaks(points)
        weekly_evidence = [{"week_index": week, "peak": peak} for week, peak in weekly]
        if len(weekly) < _MIN_GROWTH_DATA_POINTS:
            return self.not_applicable(
                context,
                reason=(
                    f"only {len(weekly)} week(s) of ConcurrentCalls data are available; "
                    f"at least {_MIN_GROWTH_DATA_POINTS} are needed before a trend is "
                    "meaningful"
                ),
                evidence={"weekly_peaks": weekly_evidence},
            )

        slope = round(_linear_slope(weekly), 2)
        latest = weekly[-1][1]
        # Reported alongside the observed count so a reader can see that a trend
        # fitted over four datapoints may span considerably more than four weeks.
        weeks_spanned = weekly[-1][0] - weekly[0][0] + 1
        evidence: Dict[str, Any] = {
            "instance_alias": instance.instance_alias,
            "quota_name": quota_name,
            "quota_value": quota_value,
            "weekly_peaks": weekly_evidence,
            "weeks_observed": len(weekly),
            "weeks_spanned": weeks_spanned,
            "growth_calls_per_week": slope,
            "latest_weekly_peak": latest,
            "horizon_weeks": _GROWTH_HORIZON_WEEKS,
        }

        if slope <= 0:
            return self.create_finding(
                status=CheckStatus.PASS,
                resource_id=instance.instance_id,
                resource_type="ConnectInstance",
                description=(
                    f"Peak concurrent calls on instance {instance.display_name} are flat "
                    f"or declining across the last {weeks_spanned} weeks "
                    f"({len(weekly)} with data, {slope:+.2f} calls per week), so the "
                    f"{int(quota_value)}-call quota is not being approached."
                ),
                evidence=evidence,
            )

        remaining = quota_value - latest
        if remaining <= 0:
            weeks_to_quota = 0.0
        else:
            weeks_to_quota = round(remaining / slope, 1)
        evidence["projected_weeks_to_quota"] = weeks_to_quota

        if weeks_to_quota > _GROWTH_HORIZON_WEEKS:
            return self.create_finding(
                status=CheckStatus.PASS,
                resource_id=instance.instance_id,
                resource_type="ConnectInstance",
                description=(
                    f"Peak concurrent calls on instance {instance.display_name} are "
                    f"growing by {slope:.2f} per week. At that rate the "
                    f"{int(quota_value)}-call quota is about {weeks_to_quota:.0f} weeks "
                    f"away, beyond the {_GROWTH_HORIZON_WEEKS}-week planning horizon."
                ),
                evidence=evidence,
            )

        return self.create_finding(
            status=CheckStatus.FAIL,
            resource_id=instance.instance_id,
            resource_type="ConnectInstance",
            severity=Severity.HIGH if weeks_to_quota <= 13 else Severity.MEDIUM,
            description=(
                f"Peak concurrent calls on instance {instance.display_name} are growing "
                f"by {slope:.2f} per week, from a latest weekly peak of {int(latest)}. "
                f"At that rate the {int(quota_value)}-call quota is reached in about "
                f"{weeks_to_quota:.0f} weeks. Because quota increases are handled as "
                "support cases rather than instantly, this needs to be raised now "
                "rather than when the ceiling is hit."
            ),
            evidence=evidence,
            structured_remediation=Remediation(
                summary=(
                    "Request a concurrent-calls quota increase sized to the projected "
                    "peak, not the current one, and keep the trend under observation."
                ),
                steps=[
                    RemediationStep(
                        order=1,
                        instruction=(
                            f"Request an increase to {quota_name}. Size the request "
                            "against the forecast peak for your next seasonal high, "
                            "which this trend places above the current quota."
                        ),
                        console_path=(
                            "Service Quotas > AWS services > Amazon Connect > "
                            f"{quota_name} > Request increase at account level"
                        ),
                    ),
                    RemediationStep(
                        order=2,
                        instruction=(
                            "Confirm the growth is expected rather than the result of "
                            "repeat contacts. Rising concurrent calls with flat unique "
                            "customers usually indicates a containment or resolution "
                            "problem upstream."
                        ),
                    ),
                    RemediationStep(
                        order=3,
                        instruction=(
                            "Alarm on ConcurrentCallsPercentage so the approach to the "
                            "ceiling is tracked continuously between assessments."
                        ),
                        console_path=(
                            "CloudWatch > Alarms > Create alarm > AWS/Connect > "
                            "ConcurrentCallsPercentage"
                        ),
                    ),
                ],
                target_resources=[quota_name],
                references=_quota_references(),
            ),
        )


def register_capacity_checks(registry) -> None:
    """Register the service-quota and capacity headroom checks."""
    registry.register_check(ConfigurationQuotaUtilizationCheck())
    registry.register_check(ConcurrentCallsHeadroomCheck())
    registry.register_check(CallVolumeGrowthTrendCheck())
