"""training_data_provider package

Provides abstractions for fetching and managing training data.
"""

from training_data_provider.training_data_provider_factory import get_provider

__all__ = ["get_provider"]
