from typing import Optional

from attr import dataclass


@dataclass
class AudioRecordMetadata:
    text: str
    zip_entry_name: str
    sample_rate: int
    sample_count: int
    id: str
    group: str
    batch: int
    seq_no: int
    bitrate: Optional[str] = None
    format: Optional[str] = None

    @property
    def duration(self):
        return float(self.sample_count) / self.sample_rate
