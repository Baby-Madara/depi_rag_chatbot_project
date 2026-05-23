"""
CityFoam — Secret Management
==============================
Priority chain for every secret:
  1. Azure Key Vault  (when AZURE_KEYVAULT_URL is set)
  2. Environment variable / .env file  (local dev fallback)

Usage anywhere in the codebase:
    from app_secrets import get_secret
    api_key = get_secret("AZURE-API-KEY")   # KV name  (hyphens)
    api_key = get_secret("AZURE_API_KEY")   # env name (underscores) — auto-mapped

The function maps underscores → hyphens when querying Key Vault, because Azure
Key Vault secret names cannot contain underscores.

Install extras for Key Vault support:
    pip install azure-identity azure-keyvault-secrets
"""

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("cityfoam.secrets")

# ---------------------------------------------------------------------------
# Lazy Key Vault client — only created if AZURE_KEYVAULT_URL is set
# ---------------------------------------------------------------------------
_kv_client = None
_kv_available: Optional[bool] = None   # None = not yet checked


def _get_kv_client():
    global _kv_client, _kv_available

    if _kv_available is False:
        return None
    if _kv_client is not None:
        return _kv_client

    vault_url = os.environ.get("AZURE_KEYVAULT_URL", "").strip()
    if not vault_url:
        _kv_available = False
        logger.info("AZURE_KEYVAULT_URL not set — using environment variables only.")
        return None

    try:
        from azure.identity import DefaultAzureCredential          # noqa: PLC0415
        from azure.keyvault.secrets import SecretClient            # noqa: PLC0415

        credential = DefaultAzureCredential()
        _kv_client = SecretClient(vault_url=vault_url, credential=credential)
        _kv_available = True
        logger.info("Azure Key Vault client initialised (%s).", vault_url)
        return _kv_client

    except ImportError:
        _kv_available = False
        logger.warning(
            "azure-identity / azure-keyvault-secrets not installed. "
            "Run: pip install azure-identity azure-keyvault-secrets"
        )
        return None

    except Exception as exc:
        _kv_available = False
        logger.warning("Key Vault init failed (%s) — falling back to env vars.", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def get_secret(name: str, default: str = "") -> str:
    """
    Retrieve a secret by name.

    Args:
        name:    Secret name using either underscores (env style) or
                 hyphens (Key Vault style). Both are tried automatically.
        default: Value returned when the secret is not found anywhere.

    Returns:
        The secret value string, or `default` if not found.
    """
    # Try Key Vault first
    client = _get_kv_client()
    if client is not None:
        kv_name = name.replace("_", "-")   # KV doesn't allow underscores
        try:
            secret = client.get_secret(kv_name)
            logger.debug("Secret '%s' loaded from Key Vault.", kv_name)
            return secret.value or default
        except Exception as exc:
            logger.debug("Key Vault miss for '%s': %s — trying env.", kv_name, exc)

    # Fall back to environment variable
    value = os.environ.get(name, os.environ.get(name.replace("-", "_"), default))
    if value and value != default:
        logger.debug("Secret '%s' loaded from environment.", name)
    return value


def invalidate_cache() -> None:
    """Clear the LRU cache so secrets are re-fetched (e.g. after rotation)."""
    get_secret.cache_clear()
    logger.info("Secret cache invalidated.")