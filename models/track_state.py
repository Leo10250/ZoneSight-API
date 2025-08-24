from dataclasses import dataclass


@dataclass
class TrackState:
    id: int
    first_seen: float
    last_seen: float
    zone: str | None = None
    entered_at: float | None = None
    cls: str | None = None
    
    def dwell(self, now: float) -> float:
        return 0.0 if self.entered_at is None else max(0.0, now - self.entered_at)
