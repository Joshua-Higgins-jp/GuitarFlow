from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict

from globals import ClassLabels


class ClassProbabilities(BaseModel):
    """
    Softmax output probabilities for each guitar classification class.

    All three fields should sum to approximately 1.0. Each value is the
    raw probability assigned by the model to that class.
    """
    model_config = ConfigDict(populate_by_name=True)

    acoustic: float = Field(ge=0.0, le=1.0)
    electric: float = Field(ge=0.0, le=1.0)
    not_guitar: float = Field(default=0.0, ge=0.0, le=1.0)  # TODO: not implemented


class InferenceEvent(BaseModel):
    """
    A single end-to-end record of one guitar classification inference.

    Captures everything needed to reconstruct, audit, or monitor a
    prediction — from the input image metadata through to the model
    output and latency. Intended to be serialised and shipped to the
    Datadog Logs API via send_inference_event().

    One InferenceEvent is created per user-submitted image. Events are
    grouped by session_id to correlate multiple predictions within the
    same user session.

    Notes:
        hash="SHA-256 hex digest of the raw image bytes. Used for deduplication."
    """
    event_id: UUID = Field(default_factory=uuid4)
    image_capture_timestamp: datetime
    session_id: UUID
    model_version: str

    # input
    image_filename: str
    image_hash: str
    image_width_px: int = Field(ge=1)
    image_height_px: int = Field(ge=1)
    image_format: str
    image_file_size_bytes: int = Field(ge=0)
    num_channels: int = Field(default=3, ge=1, le=4)

    # output
    predicted_class: ClassLabels
    confidence_score: float = Field(ge=0.0, le=1.0)
    all_class_probabilities: ClassProbabilities
    inference_latency_ms: float = Field(ge=0.0)

    @field_validator("image_capture_timestamp")
    @classmethod
    def must_be_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (UTC)")
        return v.astimezone(timezone.utc)
