"""training_data_provider/provider_factory.py

Factory for creating training data provider instances.

This module provides a factory function to create training data providers.
Currently supports SimpleTextProvider, but the factory pattern allows for
easy addition of other provider types (e.g., local file provider, database
provider, streaming provider) in the future.
"""

from typing import Any
from training_data_provider.simple_text_provider import SimpleTextProvider


def get_provider(
    provider_type: str = "simple_text",
    url: str | None = None,
    cache_path: str | None = None,
    timeout: int = 30,
    use_cache: bool = True,
    **kwargs: Any,
) -> Any:
    """Factory function to create a training data provider.

    Parameters:
    - provider_type: Type of provider to create (default: "simple_text")
    - url: URL to fetch training data from (required for simple_text provider)
    - cache_path: Optional path to cache the data
    - timeout: HTTP request timeout in seconds (default: 30)
    - use_cache: Whether to use cached data if available (default: True)
    - **kwargs: Additional provider-specific parameters

    Returns:
    - A training data provider instance with a get_text() method

    Raises:
    - ValueError: If provider_type is unknown or required params are missing
    """
    if provider_type == "simple_text":
        if url is None:
            raise ValueError("url parameter is required for simple_text provider")
        return SimpleTextProvider(
            url=url,
            cache_path=cache_path,
            timeout=timeout,
            use_cache=use_cache,
        )
    else:
        raise ValueError(
            f"Unknown provider_type '{provider_type}'. "
            f"Supported types: simple_text"
        )


__all__ = ["get_provider"]
