from remem.utils.config_utils import resolve_embedding_api_key, resolve_embedding_base_url

from .base import BaseEmbeddingModel, EmbeddingConfig


def _get_embedding_client(global_config, embedding_model_name: str = "nvidia/NV-Embed-v2", openai_style_server=True):
    if "text-embedding" in embedding_model_name or openai_style_server:
        from .openai_embedding_client import CacheOpenAIEmbeddingModel

        base_url = resolve_embedding_base_url(
            config=global_config,
            embedding_model_name=embedding_model_name,
            openai_style_server=openai_style_server,
        )
        embedding_client = CacheOpenAIEmbeddingModel(
            None,
            global_config,
            embedding_model_name,
            api_key=resolve_embedding_api_key(global_config, embedding_model_name=embedding_model_name),
            base_url=base_url,
        )
    elif "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel

        embedding_client = GritLMEmbeddingModel(global_config, embedding_model_name)
    elif "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel

        embedding_client = NVEmbedV2EmbeddingModel(global_config, embedding_model_name)
    else:
        assert False, f"Unknown embedding model name: {embedding_model_name}"
    return embedding_client
