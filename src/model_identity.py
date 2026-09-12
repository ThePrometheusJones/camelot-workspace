"""Model identity gate — refuse requests when the wrong model is loaded."""
import logging
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 5


def check_model_identity(endpoint_url: str, expected_model: str):
    """Check /v1/models on the endpoint against expected_model.

    Returns (ok, actual_id, message).
    - ok=True: model matches or check couldn't run (fail-open).
    - ok=False: mismatch — message explains what's loaded vs expected.
    """
    if not expected_model:
        return True, "", ""

    # Derive the /v1/models URL from the endpoint
    base = endpoint_url.rsplit("/v1/", 1)[0] if "/v1/" in endpoint_url else endpoint_url.rstrip("/")
    models_url = f"{base}/v1/models"

    try:
        r = requests.get(models_url, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        entries = data.get("data") or []
        actual = entries[0].get("id", "") if entries else ""
    except Exception as e:
        logger.warning("Model identity check failed (endpoint unreachable): %s", e)
        return True, "", ""  # fail-open: don't block if endpoint is down

    if actual == expected_model:
        return True, actual, ""

    msg = (
        f"Guinevere is offline \u2014 :8080 is serving {actual!r}, "
        f"expected {expected_model!r}"
    )
    logger.warning("Model identity mismatch: %s", msg)
    return False, actual, msg
