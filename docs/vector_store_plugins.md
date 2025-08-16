# Vector store plugins

The memory system exposes a small plugin architecture for vector storage
backends.  Third‑party packages can provide their own implementation by
conforming to the :class:`~memory_system.core.vector_store.VectorStoreProtocol`
interface and registering the class with ``register_vector_store``.

## Creating a plugin

1. Implement the required coroutine methods ``add``, ``search``, ``delete``,
   ``flush`` and ``close``.
2. Register the class using the decorator::

    from memory_system.core.vector_store import register_vector_store

    @register_vector_store("mybackend")
    class MyVectorStore:
        ...

3. Instantiate the backend via :func:`create_vector_store` or through the
   :class:`VectorStoreFactory` which allows runtime swapping between registered
   backends.

See :mod:`memory_system.core.faiss_vector_store`,
:mod:`memory_system.core.qdrant_store` and
:mod:`memory_system.core.duckdb_store` for reference implementations.
