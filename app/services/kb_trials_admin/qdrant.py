"""Qdrant collections health and metadata."""


async def get_qdrant_collections(host: str, port: int) -> dict:
    """
    Get list of KB collections in Qdrant with health info.

    Returns collection metadata, vector config, and optimizer status.
    """
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(host=host, port=port)

    try:
        collections_response = await client.get_collections()

        result = []
        for coll in collections_response.collections:
            try:
                info = await client.get_collection(coll.name)

                # Parse embedding model from collection name if encoded
                # Format: trading_kb_trials__{model}__{dim}
                parts = coll.name.split("__")
                embedding_model = parts[1] if len(parts) >= 2 else None
                embedding_dim = int(parts[2]) if len(parts) >= 3 else None

                # Vector config - handle both single and named vectors
                vec_cfg = info.config.params.vectors
                vector_size = None
                distance = None
                if vec_cfg:
                    # Check if it's a dict (named vectors) or single VectorParams
                    if isinstance(vec_cfg, dict):
                        # Get first vector config from dict
                        first_vec = next(iter(vec_cfg.values()), None)
                        if first_vec:
                            vector_size = getattr(first_vec, "size", None)
                            dist_attr = getattr(first_vec, "distance", None)
                            distance = dist_attr.value if dist_attr else None
                    else:
                        vector_size = getattr(vec_cfg, "size", None)
                        dist_attr = getattr(vec_cfg, "distance", None)
                        distance = dist_attr.value if dist_attr else None

                # Payload indexes count
                payload_indexes = 0
                if info.payload_schema:
                    payload_indexes = len(info.payload_schema)

                # Optimizer status
                optimizer_status = "unknown"
                if info.optimizer_status:
                    optimizer_status = (
                        info.optimizer_status.status.value
                        if hasattr(info.optimizer_status, "status")
                        else str(info.optimizer_status)
                    )

                result.append(
                    {
                        "name": coll.name,
                        "points_count": info.points_count,
                        "vectors_count": info.vectors_count,
                        "status": info.status.value if info.status else "unknown",
                        "vector_size": vector_size or embedding_dim,
                        "distance": distance,
                        "embedding_model_id": embedding_model,
                        "payload_indexes_count": payload_indexes,
                        "optimizer_status": optimizer_status,
                        "segments_count": (
                            len(info.segments or [])
                            if hasattr(info, "segments")
                            else None
                        ),
                    }
                )
            except Exception as e:
                result.append(
                    {
                        "name": coll.name,
                        "error": str(e),
                    }
                )

        return {
            "collections": result,
            "qdrant_host": host,
            "qdrant_port": port,
            "total_collections": len(result),
        }

    finally:
        await client.close()
