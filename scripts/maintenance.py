from __future__ import annotations

import asyncio
import logging

from memory_system.core.maintenance import maintenance_loop

if __name__ == "__main__":  # pragma: no cover - manual execution
    logging.basicConfig(level=logging.INFO)
    asyncio.run(maintenance_loop())
