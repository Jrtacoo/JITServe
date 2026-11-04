import pickle
import socket
import struct
import threading

import pytest

from jitserve.request_analyzer.prediction_protocol import receive_message, send_message


def test_long_prompt_round_trip():
    message = {"prompt": ["A long research prompt. " * 10000]}
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.settimeout(5)
        writer.settimeout(5)
        thread = threading.Thread(target=send_message, args=(writer, message))
        thread.start()
        assert receive_message(reader) == message
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_fragmented_header_and_body():
    message = {"output_len": [452]}
    data = pickle.dumps(message)
    frame = struct.pack("!I", len(data)) + data
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.settimeout(5)

        def write_fragments():
            for byte in frame:
                writer.sendall(bytes([byte]))

        thread = threading.Thread(target=write_fragments)
        thread.start()
        assert receive_message(reader) == message
        thread.join(timeout=5)


def test_truncated_message_raises_eof():
    reader, writer = socket.socketpair()
    with reader:
        writer.sendall(struct.pack("!I", 100) + b"partial")
        writer.close()
        with pytest.raises(EOFError, match="complete message"):
            receive_message(reader)
