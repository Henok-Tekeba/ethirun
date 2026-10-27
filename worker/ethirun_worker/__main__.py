"""Worker entry point: claim stdout first, then load the runtime."""

from __future__ import annotations


def _run() -> int:
    from ethirun_worker.streams import claim_stdout

    writer = claim_stdout()
    try:
        from ethirun_worker.server import main

        return main(writer)
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(_run())
