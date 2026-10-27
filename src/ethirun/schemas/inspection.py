"""Static-scan and inventory schemas added for the inspection stage."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FileKind(StrEnum):
    """Why a repository file matters to inspection."""

    CONFIG = "config"
    GENERATION_CONFIG = "generation_config"
    PREPROCESSOR_CONFIG = "preprocessor_config"
    TOKENIZER = "tokenizer"
    README = "readme"
    REQUIREMENTS = "requirements"
    SAFETENSORS_INDEX = "safetensors_index"
    PYTHON = "python"
    WEIGHTS = "weights"
    NEMO = "nemo"
    ONNX = "onnx"
    OTHER = "other"


class FileRecord(BaseModel):
    """One file seen in the repository inventory.

    Only allowlisted small files are read. Weights are listed with their size
    and ``read=False``; inspection never downloads them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    size: int | None = Field(default=None, ge=0)
    kind: FileKind
    read: bool
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    skip_reason: str | None = None


class ImportKind(StrEnum):
    """Where an imported module comes from."""

    STDLIB = "stdlib"
    LOCAL = "local"
    THIRD_PARTY = "third_party"


class ImportRecord(BaseModel):
    """One import statement, parsed statically and never executed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    module: str = Field(min_length=1)
    pypi_name: str | None = None
    source_file: str = Field(min_length=1)
    lineno: int = Field(ge=1)
    kind: ImportKind
    declared: bool | None = None

    @model_validator(mode="after")
    def _third_party_mapping_is_complete(self) -> Self:
        if self.kind is ImportKind.THIRD_PARTY:
            if not self.pypi_name:
                raise ValueError("third-party imports need a PyPI name")
            if self.declared is None:
                raise ValueError("third-party imports need a declaration verdict")
        elif self.pypi_name is not None or self.declared is not None:
            raise ValueError("stdlib/local imports must not carry PyPI declarations")
        return self


class AdapterChoice(BaseModel):
    """Adapter selection with confidence and support status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    supported: bool
    requires_trust_remote_code: bool = False
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _supported_needs_family(self) -> Self:
        if self.supported and not self.family:
            raise ValueError("a supported adapter choice needs a family")
        return self


class StaticScanResult(BaseModel):
    """Stage 1 output: imports, declarations, remote-code flag, adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python_files: tuple[str, ...] = ()
    remote_code: bool = False
    remote_code_reasons: tuple[str, ...] = ()
    imports: tuple[ImportRecord, ...] = ()
    declared_packages: tuple[str, ...] = ()
    undeclared_imports: tuple[str, ...] = ()
    scan_errors: tuple[str, ...] = ()
    adapter: AdapterChoice
