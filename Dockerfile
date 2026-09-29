# SECURITY: pinned to a digest, not floating on :latest (the only tag this
# registry actually publishes — ghcr.io/v2/lostruins/koboldcpp/tags/list
# returns just ["latest"], so a tag-only pin wasn't possible). Floating on
# :latest meant every rebuild could silently pull a different — or a
# compromised, or simply unavailable — image with no record of what
# actually changed. This is the digest latest resolved to as of this fix;
# refresh it deliberately (not as a side effect of an unrelated rebuild) by
# re-running:
#   TOKEN=$(curl -sS "https://ghcr.io/token?scope=repository:lostruins/koboldcpp:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
#   curl -sS -D - -o /dev/null -H "Authorization: Bearer $TOKEN" \
#     -H "Accept: application/vnd.oci.image.index.v1+json" \
#     https://ghcr.io/v2/lostruins/koboldcpp/manifests/latest | grep -i docker-content-digest
FROM ghcr.io/lostruins/koboldcpp@sha256:fc7e2b540eea67cf973b5b56f3fa5822215d7cf0fe64dbe326d0752126c7df88

WORKDIR /opt/koboldcpp

RUN apt update && apt install -y python3-pip python3-venv curl ffmpeg

# The official frozen koboldcpp binary is no longer downloaded at build
# time here — handler.py's ensure_koboldcpp_engine() fetches it once at
# runtime (cached on the persistent RunPod volume) purely to extract its
# compiled CUDA backend (koboldcpp_cublas.so). We run our own patched
# koboldcpp.py (below) instead of that frozen binary, so it adds
# /api/extra/abort_image — a genuine on-demand cancel for image/video
# generation. See koboldcpp_engine/UPDATING.md for what changed and why.
COPY koboldcpp_engine ./koboldcpp_engine
COPY handler.py .
COPY test_input.json .

RUN pip install runpod requests boto3 psutil --break-system-packages

ENTRYPOINT []
CMD ["python3", "-u", "handler.py"]
