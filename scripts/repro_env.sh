#!/usr/bin/env bash
# Source this file after optionally exporting JITSERVE_PROXY.
if [[ -n "${JITSERVE_PROXY:-}" ]]; then
    export http_proxy="$JITSERVE_PROXY" https_proxy="$JITSERVE_PROXY"
    export HTTP_PROXY="$JITSERVE_PROXY" HTTPS_PROXY="$JITSERVE_PROXY"
fi
if [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
    export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
    export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
    export PIP_CERT="${PIP_CERT:-$SSL_CERT_FILE}"
    export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$SSL_CERT_FILE}"
    export GIT_SSL_CAINFO="${GIT_SSL_CAINFO:-$SSL_CERT_FILE}"
fi
export NO_PROXY="${NO_PROXY:-${no_proxy:-}},localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
# Use requests, which honors the CA bundle above, for Hugging Face downloads.
export HF_HUB_DISABLE_XET=1
