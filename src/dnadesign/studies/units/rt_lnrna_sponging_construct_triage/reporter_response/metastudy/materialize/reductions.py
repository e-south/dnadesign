"""Reduction orchestration for reporter-response materialization."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ....reader_evidence import ReaderDataframeRecordRef, ReaderEvidenceBinding, ReaderEvidenceBindingSet
from ...policy import ReporterResponseObservationPolicy
from ...profile.measurement import EndpointReduction, Reduction
from ..condition_ontology import ReporterResponseConditionOntology
from ..contracts.materialization import MaterializationOmission
from ..contracts.profile import ProfileEvidence
from ..contracts.protocol import MetastudyProtocol
from .identities import _observed_reader_identities, _reader_identity_mask
from .profile_building import _build_profile, _reduction_id


def _materialize_reductions(
    frame: pd.DataFrame,
    *,
    reductions: Iterable[Reduction],
    reference: EndpointReduction,
    record: ReaderDataframeRecordRef,
    bindings: ReaderEvidenceBindingSet,
    binding_by_identity: dict[tuple[str, str | None], ReaderEvidenceBinding],
    ontology: ReporterResponseConditionOntology,
    policy: ReporterResponseObservationPolicy,
    protocol: MetastudyProtocol,
    include_sensitivity_doses: bool,
) -> tuple[tuple[ProfileEvidence, ...], tuple[MaterializationOmission, ...]]:
    """Materialize every requested reduction for each observed Reader identity."""

    evidence: list[ProfileEvidence] = []
    omissions: list[MaterializationOmission] = []
    for reduction in reductions:
        for identity in sorted(_observed_reader_identities(frame), key=lambda value: (value[0], value[1] or "")):
            design_id, assay_subject_id = identity
            binding = binding_by_identity[identity]
            subject_id = binding.subject_id
            design_frame = frame.loc[_reader_identity_mask(frame, identity)]
            built = _build_profile(
                design_frame,
                reduction=reduction,
                reference=reference,
                record=record,
                bindings=bindings,
                design_id=design_id,
                assay_subject_id=assay_subject_id,
                subject_id=subject_id,
                ontology=ontology,
                policy=policy,
                protocol=protocol,
                include_sensitivity_doses=include_sensitivity_doses,
            )
            if isinstance(built, str):
                omissions.append(
                    MaterializationOmission(
                        code=built,
                        subject_id=subject_id,
                        reduction_id=_reduction_id(reduction),
                    )
                )
            else:
                evidence.append(built)
    return tuple(evidence), tuple(omissions)
