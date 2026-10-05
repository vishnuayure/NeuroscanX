import numpy as np


MODALITIES = ("T1", "T1ce", "T2", "FLAIR")


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)

    minimum = np.nanmin(volume)
    maximum = np.nanmax(volume)

    if maximum <= minimum:
        return np.zeros_like(volume, dtype=np.float32)

    return ((volume - minimum) / (maximum - minimum)).astype(np.float32)


def modality_contribution(
    volumes: dict[str, np.ndarray],
    segmentation: np.ndarray | None = None
) -> dict[str, float]:
    contributions = {}

    for modality in MODALITIES:
        if modality not in volumes:
            raise ValueError(f"Missing modality: {modality}")

        volume = normalize_volume(volumes[modality])

        if segmentation is not None:
            mask = np.asarray(segmentation) > 0

            if volume.shape != mask.shape:
                raise ValueError(
                    f"{modality} and segmentation must have the same shape"
                )

            score = float(np.mean(volume[mask])) if np.any(mask) else 0.0
        else:
            score = float(np.mean(volume))

        contributions[modality] = score

    total = sum(contributions.values())

    if total > 0:
        contributions = {
            modality: score / total
            for modality, score in contributions.items()
        }

    return contributions


def modality_contribution_map(
    volumes: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    maps = {}

    for modality in MODALITIES:
        if modality not in volumes:
            raise ValueError(f"Missing modality: {modality}")

        maps[modality] = normalize_volume(
            volumes[modality]
        )

    return maps


def dominant_modality(
    contributions: dict[str, float]
) -> str:
    if not contributions:
        raise ValueError("Contributions cannot be empty")

    return max(
        contributions,
        key=contributions.get
    )


__all__ = [
    "MODALITIES",
    "normalize_volume",
    "modality_contribution",
    "modality_contribution_map",
    "dominant_modality"
]