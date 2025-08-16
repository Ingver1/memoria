# aiosqlite fork

This project includes a minimal async wrapper around Python's built-in
`sqlite3` module rather than depending on the official `aiosqlite` package.
The implementation lives in [`memory_system/_vendor/aiosqlite/__init__.py`](../memory_system/_vendor/aiosqlite/__init__.py)
and intentionally covers only a small subset of the upstream API.

## Rationale
- Keep the dependency footprint small while exposing an async interface
  used by the memory store and tests.

## Key differences from upstream
- **Execution model:** each database call is dispatched with
  `asyncio.to_thread`, whereas the real `aiosqlite` maintains its own worker
  thread.
- **Surface area:** only `Row`, `Cursor.fetchone`, `Cursor.fetchall`, and
  connection methods `execute`, `executemany`, `executescript`, `commit` and
  `close` are provided. Features such as `cursor()`, transaction helpers,
  `rollback`, `iterdump`, custom functions and more are absent.
- **Context management:** connections returned by `connect()` are not async
  context managers; callers must explicitly `close()` them.
- **API options:** the `connect()` helper accepts `dsn`, `uri`, `timeout` and
  `check_same_thread` arguments. Other parameters from the official package
  are omitted.

Because of these limitations, code written against the real `aiosqlite` may
require adjustments when used with this fork
