# wavefront

The official Python client for **[Finwave](https://finwave.io)** datasets.

Finwave serves frozen, versioned wildlife photo-identification and detector
datasets behind a small handshake API. `wavefront` turns that into one call.

```bash
pip install wavefront
```

## Quick start

```python
import wavefront

# the API key is read from $FW_API_TOKEN (or passed as api_key=...)
ds = wavefront.fetch("a7673931-9810-4c52-9654-1c9b1fafb63d", format="yolo")

print(ds.path)          # extracted, ready to train on
print(ds.classes)       # ['fluke']
print(ds.num_images)    # 497
print(ds.fingerprint)   # content hash — record it next to any model you train
```

`ds` is path-like, so it drops straight into a trainer:

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").train(data=f"{ds.path}/data.yaml")
```

### Pre-flight without downloading

```python
m = wavefront.manifest("a7673931-9810-4c52-9654-1c9b1fafb63d")
print(m.name, m.sample_count, m.available_formats)   # Flukes v1 497 ['Yolo']
```

### A reusable client

```python
from wavefront import Client
client = Client(api_key="...", base_url="https://finwave.io")
ds = client.fetch(dataset_id, format="yolo", dest="./data/flukes")
```

### Command line

```bash
export FW_API_TOKEN=...
wavefront manifest a7673931-9810-4c52-9654-1c9b1fafb63d
wavefront fetch    a7673931-9810-4c52-9654-1c9b1fafb63d --format yolo --dest ./data/flukes
```

## How it works

1. `GET /manifest` — cheap metadata + which export formats are ready.
2. `GET ?format=…` — a **handshake** that mints a short-lived signed download URL.
3. Download that URL → a zip → extract → a `Dataset`.

Downloads are **cached by content fingerprint**, so re-fetching a frozen
version is a no-op. The key needs the dataset-download scope.

## Authentication

Provide the key explicitly (`fetch(..., api_key=...)`) or set **`FW_API_TOKEN`**.
For compatibility, `WAVEFRONT_API_KEY`, `FINWAVE_DATASET_API_KEY` and
`DATASET_API_KEY` are also accepted (in that order).

## Errors

All errors subclass `wavefront.WavefrontError`:

| Exception | When |
|---|---|
| `AuthError` | key missing / rejected (401/403) |
| `DatasetNotFoundError` | no such version, or not visible to the key (404) |
| `FormatNotAvailableError` | the version exists but that export hasn't been generated yet (`.available` lists what is) |
| `APIError` | any other non-success response |

## License

MIT © Alexander Barnhill / [Operational Ecology](https://operationalecology.io).
A partnership artifact between finwave and Operational Ecology.
