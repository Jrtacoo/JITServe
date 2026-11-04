"""Frame QRF messages so prompts larger than a single TCP read are supported."""
import pickle
import struct


def _read_exact(connection, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("QRF connection closed before a complete message arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection, message):
    data = pickle.dumps(message)
    connection.sendall(struct.pack("!I", len(data)) + data)


def receive_message(connection):
    size, = struct.unpack("!I", _read_exact(connection, 4))
    return pickle.loads(_read_exact(connection, size))
