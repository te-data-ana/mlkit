from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import GenericAlias
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model
from pydantic_core import PydanticCustomError
from sklearn.base import BaseEstimator


def _offset_validation_error(exc: ValidationError, start: int) -> ValidationError:
    """Translate chunk-local error locations to DataFrame row positions."""
    errors = exc.errors(include_url=False)
    for error in errors:
        error["loc"] = (error["loc"][0] + start, *error["loc"][1:])
        error["type"] = PydanticCustomError(
            error["type"], error["msg"], error.get("ctx")
        )
    return ValidationError.from_exception_data(exc.title, errors)


@dataclass(frozen=True)
class UseCase:
    """Data and adapter contract for one classification use case."""

    name: str
    target_column: str
    time_column: str | None
    feature_schema: type[BaseModel]
    feature_builder_factory: Callable[[], BaseEstimator]
    split_strategy: Literal["temporal", "stratified_random"] = "temporal"
    index_column: str | None = None
    metadata_fields: Mapping[str, Any] = field(default_factory=dict)
    target_classes: tuple[int, ...] = (0, 1)
    categorical_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Use-case name must not be empty.")
        if not self.target_column:
            raise ValueError("Use-case target column must not be empty.")
        if self.split_strategy not in {"temporal", "stratified_random"}:
            raise ValueError(
                "Use-case split strategy must be temporal or stratified_random."
            )
        if self.time_column is not None and (
            not isinstance(self.time_column, str) or not self.time_column
        ):
            raise ValueError("Use-case time column must be nonempty text or None.")
        if self.split_strategy == "temporal" and self.time_column is None:
            raise ValueError("Temporal use cases require a nonempty time column.")
        if self.target_column == self.event_time_column:
            raise ValueError("Use-case target and time columns must be distinct.")
        if not self.feature_columns:
            raise ValueError("Use-case feature schema must not be empty.")
        if (
            len(self.target_classes) < 2
            or any(type(value) is not int for value in self.target_classes)
            or any(value < 0 for value in self.target_classes)
            or len(set(self.target_classes)) != len(self.target_classes)
            or (len(self.target_classes) == 2 and self.target_classes != (0, 1))
        ):
            raise ValueError(
                "Target classes must be distinct nonnegative integers; binary classes must be (0, 1)."
            )
        if not set(self.categorical_columns).issubset(self.feature_columns):
            raise ValueError("Categorical columns must be use-case features.")
        if {self.target_column, self.event_time_column}.intersection(
            self.feature_columns
        ):
            raise ValueError(
                "Use-case feature_columns must not include target or time columns."
            )
        reserved = set(self.feature_columns) | {
            self.target_column,
            self.event_time_column,
        }
        if reserved.intersection(self.metadata_fields):
            raise ValueError(
                "Metadata fields must not overlap features, target or time."
            )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the feature names in schema order."""
        return tuple(self.feature_schema.model_fields)

    @property
    def class_selection_strategy(self) -> Literal["max_f1", "argmax"]:
        """Return the class selection strategy implied by the target classes."""
        return "max_f1" if len(self.target_classes) == 2 else "argmax"

    @property
    def event_time_column(self) -> str:
        """Return the timestamp field used for serving and monitoring events."""
        return self.time_column if self.time_column is not None else "timestamp"

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate features and present metadata, filtering unknown columns.

        Pydantic error locations identify the zero-based DataFrame row position.
        The original index and recognized column order are preserved.
        """
        if not df.columns.is_unique:
            raise ValueError("DataFrame columns must be unique.")
        missing = {
            name
            for name, info in self.feature_schema.model_fields.items()
            if info.is_required() and name not in df.columns
        }
        if missing:
            raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}.")

        feature_adapter: TypeAdapter[list[BaseModel]] = TypeAdapter(
            GenericAlias(list, self.feature_schema)
        )

        metadata_schema = self._metadata_schema(df.columns)
        metadata_adapter: TypeAdapter[list[BaseModel]] = TypeAdapter(
            GenericAlias(list, metadata_schema)
        )

        input_columns = [
            name
            for name in df.columns
            if name in self.feature_columns or name in metadata_schema.model_fields
        ]
        columns = input_columns + [
            name for name in self.feature_columns if name not in df.columns
        ]
        if df.empty:
            return df.loc[:, input_columns].reindex(columns=columns).copy()

        chunks = []
        for start in range(0, len(df), 10_000):
            chunk = df.iloc[start : start + 10_000].loc[:, input_columns]
            chunks.append(
                self._validate_chunk(
                    chunk, start, columns, feature_adapter, metadata_adapter
                )
            )
        return pd.concat(chunks)

    def _metadata_schema(self, columns: pd.Index) -> type[BaseModel]:
        """Create a schema for the metadata columns present in a DataFrame."""
        metadata_types = {
            self.target_column: (
                Annotated[int, Field(ge=0, le=1)]
                if self.target_classes == (0, 1)
                else Literal.__getitem__(self.target_classes)
            ),
            self.event_time_column: datetime,
            **self.metadata_fields,
        }
        return create_model(
            "DataMetadata",
            **{
                name: (annotation, ...)
                for name, annotation in metadata_types.items()
                if name in columns
            },
        )

    def _validate_chunk(
        self,
        chunk: pd.DataFrame,
        start: int,
        columns: list[str],
        feature_adapter: TypeAdapter[list[BaseModel]],
        metadata_adapter: TypeAdapter[list[BaseModel]],
    ) -> pd.DataFrame:
        """Validate one chunk and reconstruct it with its original index."""
        normalized = chunk.astype(object).where(chunk.notna(), None)
        records = normalized.to_dict("records")
        feature_columns = [
            name for name in self.feature_columns if name in chunk.columns
        ]
        feature_records = (
            normalized[feature_columns].to_dict("records")
            if feature_columns
            else [{} for _ in records]
        )
        try:
            features = feature_adapter.validate_python(feature_records)
            metadata = metadata_adapter.validate_python(records)
        except ValidationError as exc:
            raise _offset_validation_error(exc, start) from exc

        return pd.DataFrame(
            [
                feature.model_dump() | meta.model_dump()
                for feature, meta in zip(features, metadata)
            ],
            columns=columns,
            index=chunk.index,
        )

    def build_feature_builder(self) -> BaseEstimator:
        """Create a fresh scikit-learn-compatible feature transformer."""
        return self.feature_builder_factory()
