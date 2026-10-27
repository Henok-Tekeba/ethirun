"""Protocol stream setup, kept free of adapter imports.

The protocol writer owns a private duplicate of the original stdout descriptor.
Claiming stdout happens before adapter imports, so import-time chatter from
libraries cannot corrupt the first frame.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from ethirun_worker.protocol import encode


@dataclass
class ProtocolWriter:
    """Writes protocol frames to the private copy of the original stdout."""

    stream: TextIO

    def send(self, message: dict[str, Any]) -> None:
        self.stream.write(encode(message))
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def claim_stdout() -> ProtocolWriter:
    """Reserve original stdout for protocol and redirect everything else.

    A private descriptor is duplicated before file descriptor 1 is pointed at
    stderr. Both ``sys.stdout`` writes and raw ``os.write(1, ...)`` calls from
    libraries therefore land on stderr, where the host drains them as logs.
    """
    sys.stdout.flush()
    private_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    private_stream = os.fdopen(
        private_fd,
        "w",
        encoding="utf-8",
        buffering=1,
        newline="\n",
    )
    return ProtocolWriter(private_stream)
