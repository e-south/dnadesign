"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/folding/secondary_structure_prediction_v2.py

Backend-neutral secondary-structure prediction contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._viennarna_parameters import validate_viennarna_parameters


class FoldingContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _not_blank(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    return text


class SecondaryStructurePredictionInputV1(FoldingContractModel):
    sequence_id: str
    sequence_sha256: str
    alphabet: Literal["dna"]
    topology: Literal["linear_ssdna"]
    length: int = Field(ge=1)

    @field_validator("sequence_id", "sequence_sha256")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value, label="input field")


class SecondaryStructurePredictionBackendV1(FoldingContractModel):
    name: str
    version: str
    command: list[str] = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "version")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value, label="backend field")

    @field_validator("command")
    @classmethod
    def _command_not_blank(cls, value: list[str]) -> list[str]:
        return [_not_blank(item, label="backend.command item") for item in value]


class SecondaryStructurePredictionDnaPolicyV1(FoldingContractModel):
    mode: Literal["convert_t_to_u_for_rna_backend", "backend_accepts_dna_directly"]
    submitted_alphabet: str
    coordinates_mapped_to: Literal["original_dna_sequence"]

    @field_validator("submitted_alphabet")
    @classmethod
    def _submitted_alphabet_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="dna_policy.submitted_alphabet")


class SecondaryStructurePairV1(FoldingContractModel):
    left: int = Field(ge=0)
    right: int = Field(ge=0)
    pair: str

    @field_validator("pair")
    @classmethod
    def _pair_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="pair")

    @model_validator(mode="after")
    def _validate_pair_order(self) -> "SecondaryStructurePairV1":
        if self.right <= self.left:
            raise ValueError("pair_map right coordinate must be greater than left coordinate.")
        return self


class SecondaryStructurePredictionResultV1(FoldingContractModel):
    dot_bracket: str
    mfe_kcal_mol: float | None = None
    pair_map: list[SecondaryStructurePairV1] = Field(default_factory=list)

    @field_validator("dot_bracket")
    @classmethod
    def _dot_bracket_not_blank(cls, value: str) -> str:
        text = _not_blank(value, label="result.dot_bracket")
        allowed = set(".()[]{}<>")
        if any(char not in allowed for char in text):
            raise ValueError("dot_bracket contains unsupported characters.")
        return text

    @model_validator(mode="after")
    def _validate_brackets(self) -> "SecondaryStructurePredictionResultV1":
        pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
        openers = set(pairs.values())
        stack: list[str] = []
        for char in self.dot_bracket:
            if char in openers:
                stack.append(char)
                continue
            if char in pairs:
                if not stack or stack.pop() != pairs[char]:
                    raise ValueError("dot_bracket has invalid bracket nesting.")
        if stack:
            raise ValueError("dot_bracket has invalid bracket nesting.")
        return self


class SecondaryStructurePairingSummaryV1(FoldingContractModel):
    predicted_pair_count: int = Field(ge=0)
    cross_copy_pair_count: int = Field(ge=0)
    intended_pairing_count: int = Field(ge=0)
    intended_recovered_count: int = Field(ge=0)
    intended_partially_recovered_count: int = Field(ge=0)
    intended_missed_count: int = Field(ge=0)


class SecondaryStructureContiguousWatsonCrickStemRunV1(FoldingContractModel):
    start_offset: int
    end_offset: int
    length_bp: int = Field(ge=1)
    primary_start: int = Field(ge=0)
    primary_end: int = Field(gt=0)
    complement_start: int = Field(ge=0)
    complement_end: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SecondaryStructureContiguousWatsonCrickStemRunV1":
        if self.end_offset <= self.start_offset:
            raise ValueError("stem run end_offset must be > start_offset.")
        if self.length_bp != self.end_offset - self.start_offset:
            raise ValueError("stem run length_bp must equal end_offset - start_offset.")
        if self.primary_end <= self.primary_start:
            raise ValueError("stem run primary span end must be > start.")
        if self.complement_end <= self.complement_start:
            raise ValueError("stem run complement span end must be > start.")
        if self.primary_end - self.primary_start != self.length_bp:
            raise ValueError("stem run primary span length must equal length_bp.")
        if self.complement_end - self.complement_start != self.length_bp:
            raise ValueError("stem run complement span length must equal length_bp.")
        return self


class SecondaryStructureIntendedPairingQaV1(FoldingContractModel):
    pairing_id: str
    primary_start: int = Field(ge=0)
    primary_end: int = Field(gt=0)
    complement_start: int = Field(ge=0)
    complement_end: int = Field(gt=0)
    expected_pair_count: int = Field(ge=0)
    predicted_pair_count: int = Field(ge=0)
    predicted_watson_crick_pair_count: int = Field(default=0, ge=0)
    contiguous_watson_crick_stem_bp: int = Field(default=0, ge=0)
    contiguous_watson_crick_stem_runs: list[SecondaryStructureContiguousWatsonCrickStemRunV1] = Field(
        default_factory=list
    )
    status: Literal["fully_recovered", "partially_recovered", "missed"]

    @field_validator("pairing_id")
    @classmethod
    def _pairing_id_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="intended pairing id")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SecondaryStructureIntendedPairingQaV1":
        if self.primary_end <= self.primary_start:
            raise ValueError("intended pairing primary span end must be > start.")
        if self.complement_end <= self.complement_start:
            raise ValueError("intended pairing complement span end must be > start.")
        return self


class SecondaryStructureQaV1(FoldingContractModel):
    length_matches_input: bool | None = None
    pairing_summary: SecondaryStructurePairingSummaryV1 | None = None
    cross_copy_pairings: list[dict[str, Any]] = Field(default_factory=list)
    intended_pairings: list[SecondaryStructureIntendedPairingQaV1] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SecondaryStructureArtifactsV1(FoldingContractModel):
    stdout: str | None = None
    stderr: str | None = None

    @field_validator("stdout", "stderr")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="artifact ref")


SecondaryStructureFailureKindV2 = Literal[
    "backend_invocation_exception",
    "backend_nonzero_exit",
    "backend_exception",
    "output_parse_exception",
]


class SecondaryStructureFailureV2(FoldingContractModel):
    kind: SecondaryStructureFailureKindV2
    message: str
    returncode: int | None = None
    exception_type: str | None = None

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="failure.message")

    @field_validator("exception_type")
    @classmethod
    def _exception_type_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="failure.exception_type")

    @model_validator(mode="after")
    def _validate_shape(self) -> "SecondaryStructureFailureV2":
        if self.kind == "backend_nonzero_exit":
            if self.returncode in {None, 0} or self.exception_type is not None:
                raise ValueError("backend_nonzero_exit requires one nonzero returncode only.")
            return self
        if self.returncode is not None:
            raise ValueError("returncode is only valid for backend_nonzero_exit.")
        exception_kinds = {
            "backend_invocation_exception",
            "backend_exception",
            "output_parse_exception",
        }
        if (self.kind in exception_kinds) is (self.exception_type is None):
            raise ValueError("failure exception_type does not match its kind.")
        return self


class SecondaryStructurePredictionV2(FoldingContractModel):
    contract: Literal["secondary_structure_prediction_v2"] = "secondary_structure_prediction_v2"
    schema_version: Literal[2] = 2
    prediction_id: str
    status: Literal[
        "ok",
        "not_run",
        "error",
        "warning_optional_missing",
        "blocker_required_missing",
        "blocker_policy_unknown",
        "blocker_output_unwritable",
    ]
    input: SecondaryStructurePredictionInputV1
    backend: SecondaryStructurePredictionBackendV1 | None = None
    dna_policy: SecondaryStructurePredictionDnaPolicyV1 | None = None
    result: SecondaryStructurePredictionResultV1 | None = None
    failure: SecondaryStructureFailureV2 | None = None
    qa: SecondaryStructureQaV1 = Field(default_factory=SecondaryStructureQaV1)
    artifacts: SecondaryStructureArtifactsV1 = Field(default_factory=SecondaryStructureArtifactsV1)

    @field_validator("prediction_id")
    @classmethod
    def _prediction_id_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="prediction_id")

    @model_validator(mode="after")
    def _validate_result(self) -> "SecondaryStructurePredictionV2":
        if self.status == "ok":
            if self.backend is None:
                raise ValueError("backend is required when status='ok'.")
            if self.dna_policy is None:
                raise ValueError("dna_policy is required when status='ok'.")
            if self.result is None:
                raise ValueError("result is required when status='ok'.")
            if self.failure is not None:
                raise ValueError("failure must be absent when status='ok'.")
        elif self.status == "error":
            if self.failure is None:
                raise ValueError("failure is required when status='error'.")
        if self.status != "ok" and self.result is not None:
            raise ValueError("result must be absent when status is not 'ok'.")
        if self.status != "error" and self.failure is not None:
            raise ValueError("failure is only valid when status='error'.")
        if self.result is None:
            return self
        if len(self.result.dot_bracket) != self.input.length:
            raise ValueError("dot_bracket length must equal input length.")
        for pair in self.result.pair_map:
            if pair.right >= self.input.length:
                raise ValueError("pair_map coordinate exceeds input length.")
        return self


class SecondaryStructurePredictionRequestInputV1(SecondaryStructurePredictionInputV1):
    sequence_artifact: str

    @field_validator("sequence_artifact")
    @classmethod
    def _sequence_artifact_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="input.sequence_artifact")


class SecondaryStructurePredictionScopeV1(FoldingContractModel):
    mode: Literal["canonical_component_unit"] = "canonical_component_unit"


class SecondaryStructurePredictionRequestDnaPolicyV1(FoldingContractModel):
    mode: Literal["convert_t_to_u_for_rna_backend", "backend_accepts_dna_directly"]
    output_coordinates: Literal["original_dna_sequence"] = "original_dna_sequence"


class SecondaryStructurePredictionRequestBackendV1(FoldingContractModel):
    name: Literal["ViennaRNA"]
    interface: Literal["cli", "python_api"] = "cli"
    executable: str | None = None
    python_module: str | None = None
    backend_contract: Literal["secondary_structure_prediction_v2"] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    dna_policy: SecondaryStructurePredictionRequestDnaPolicyV1

    @field_validator("executable", "python_module")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="backend field")

    @field_validator("parameters", mode="before")
    @classmethod
    def _supported_parameters(cls, value: object) -> dict[str, Any]:
        return validate_viennarna_parameters(value)

    @model_validator(mode="after")
    def _validate_backend_entrypoint(self) -> "SecondaryStructurePredictionRequestBackendV1":
        if self.interface == "cli" and self.executable is None:
            raise ValueError("backend.executable is required when backend.interface='cli'.")
        if self.interface == "python_api" and self.python_module is None:
            raise ValueError("backend.python_module is required when backend.interface='python_api'.")
        return self


class SecondaryStructurePredictionRequestPolicyV1(FoldingContractModel):
    required: bool = False
    fail_on_malformed_output: bool = True
    fail_on_length_mismatch: bool = True


class SecondaryStructurePredictionRequestV1(FoldingContractModel):
    contract: Literal["secondary_structure_prediction_request_v1"] = "secondary_structure_prediction_request_v1"
    schema_version: Literal[1] = 1
    request_id: str
    input: SecondaryStructurePredictionRequestInputV1
    scope: SecondaryStructurePredictionScopeV1 = Field(default_factory=SecondaryStructurePredictionScopeV1)
    backend: SecondaryStructurePredictionRequestBackendV1
    policy: SecondaryStructurePredictionRequestPolicyV1 = Field(
        default_factory=SecondaryStructurePredictionRequestPolicyV1
    )

    @field_validator("request_id")
    @classmethod
    def _request_id_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="request_id")


__all__ = ["SecondaryStructurePredictionRequestV1", "SecondaryStructurePredictionV2"]
