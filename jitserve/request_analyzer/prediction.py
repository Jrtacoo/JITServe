import asyncio
import argparse
import socket
import numpy as np
from joblib import load
from typing import Tuple
from jitserve.request_info import RequestInfo
from jitserve.paths import QRF_MODEL, QRF_VECTORIZER
from jitserve.request_analyzer.prediction_protocol import send_message, receive_message

# Load the model and tokenizer
def load_model(model_path: str, tokenizer_path: str) -> Tuple:
    try:
        model = load(model_path)
        tokenizer = load(tokenizer_path)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load QRF model {model_path} and vectorizer {tokenizer_path}: {exc}"
        ) from exc
    return model, tokenizer

# Perform prediction on the input using the loaded model and vectorizer
def predict(vectorizer, model, input):
    X_test_vec = vectorizer.transform(input)
    y_pred_i: np.array = model.predict(X_test_vec.toarray(), quantiles=[0.95])
    return [int(element) for element in y_pred_i]

# Asynchronous function to handle prediction and return result
async def async_predict(vectorizer, model, input, request_info: RequestInfo) -> None:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, predict, vectorizer, model, input)
    request_info.output_len = result[0]


# Handle incoming connections and process requests
def handle_request(connection: socket.socket, model, vectorizer):
    try: 
        with connection:
            connection.settimeout(30)
            request_info = receive_message(connection)
            result = predict(vectorizer, model, request_info['prompt'])
            send_message(connection, {"output_len": result})
    except Exception as e:
        print(f"Error during request handling: {e}")
    finally:
        connection.close()

def start_client(model_path: str, tokenizer_path: str):
    model, tokenizer = load_model(model_path, tokenizer_path)

    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 65433))
    server_socket.listen(5)

    print("QRF server is listening on localhost:65433", flush=True)

    while True:
        connection, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")
        
        # Handle each request
        handle_request(connection, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the QRF length prediction service")
    parser.add_argument("--model-path", default=str(QRF_MODEL))
    parser.add_argument("--tokenizer-path", default=str(QRF_VECTORIZER))
    args = parser.parse_args()
    start_client(args.model_path, args.tokenizer_path)
