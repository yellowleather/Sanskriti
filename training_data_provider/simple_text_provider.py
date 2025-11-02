"""training_data_provider/simple_text_provider.py

Provides training data by downloading text from a URL, optionally caching it to disk.
"""

import os
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


class SimpleTextProvider:
    """Fetches text from a URL and optionally caches it to disk.

    This provider handles:
    - Downloading text from a remote URL
    - Caching the downloaded text to a local file (optional)
    - Character encoding detection and handling
    """

    def __init__(
        self,
        url: str,
        cache_path: str | None = None,
        timeout: int = 30,
        use_cache: bool = True,
    ):
        """Initialize the SimpleTextProvider.

        Parameters:
        - url: URL to download text from
        - cache_path: Optional path to cache the downloaded text (default: None)
        - timeout: HTTP request timeout in seconds (default: 30)
        - use_cache: Whether to read from cache if it exists (default: True)
        """
        self.url = url
        self.cache_path = cache_path
        self.timeout = timeout
        self.use_cache = use_cache

    def get_text(self) -> str:
        """Fetch and return the text content.

        Returns the cached version if available and use_cache is True,
        otherwise downloads from the URL and optionally saves to cache.

        Returns:
        - str: The text content

        Raises:
        - HTTPError: If the HTTP request fails
        - URLError: If there's a network/URL error
        - IOError: If file operations fail
        """
        # Try to read from cache if enabled and file exists
        if self.use_cache and self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return f.read()

        # Download from URL
        text = self._download_text()

        # Save to cache if path is specified
        if self.cache_path:
            self._save_to_cache(text)

        return text

    def _download_text(self) -> str:
        """Download text from the URL.

        Tries to use the response Content-Type charset when available,
        falls back to utf-8.
        """
        req = Request(self.url, headers={"User-Agent": "myllm-training/1.0"})
        with urlopen(req, timeout=self.timeout) as resp:
            # Try to get charset from headers
            try:
                charset = resp.headers.get_content_charset()
            except Exception:
                # Fallback if headers object doesn't have helper
                content_type = resp.headers.get("Content-Type")
                charset = None
                if content_type and "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()

            raw = resp.read()
            encoding = charset or "utf-8"
            try:
                text = raw.decode(encoding, errors="replace")
            except Exception:
                # Last resort: decode as utf-8 replacing invalid bytes
                text = raw.decode("utf-8", errors="replace")
            return text

    def _save_to_cache(self, text: str) -> None:
        """Save text to the cache file.

        Creates parent directories if they don't exist.
        """
        if not self.cache_path:
            return

        # Ensure parent directory exists
        parent = os.path.dirname(self.cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write(text)


__all__ = ["SimpleTextProvider"]
