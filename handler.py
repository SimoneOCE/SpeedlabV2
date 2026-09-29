import runpod
import subprocess
import threading
import time
import requests
import os
import shutil
import base64
import uuid
import boto3
from botocore.client import Config

KOBOLD_URL = "http://127.0.0.1:5001"
VOLUME_DIR = "/runpod-volume"
# Overridable per-endpoint so a test endpoint (e.g. one trying a different
# checkpoint/quantization) can point at its own .kcppt without touching
# what every other endpoint, including production, loads by default.
KCPPT_CONFIG_URL = os.environ.get(
    "KCPPT_CONFIG_URL",
    "https://huggingface.co/koboldcpp/kcppt/resolve/main/DasiwaMinimaxH3.kcppt?download=true",
)
# Where our fork of koboldcpp actually runs from. Not the official frozen
# single-file binary anymore — see ensure_koboldcpp_engine()'s docstring —
# but a plain koboldcpp.py (patched, vendored in koboldcpp_engine/ below)
# plus the same official compiled koboldcpp_cublas.so backend, extracted
# once from that same frozen binary and cached here alongside it.
KOBOLD_DIR = os.path.join(VOLUME_DIR, "koboldcpp_engine")
KOBOLD_PY = os.path.join(KOBOLD_DIR, "koboldcpp.py")
# Written only once every .so from the extraction (koboldcpp_cublas.so AND
# the CUDA runtime libraries it itself depends on to even load, e.g.
# libcublas.so.12) has actually been copied to KOBOLD_DIR. Checking for
# this instead of just koboldcpp_cublas.so's own existence is what lets a
# volume that already has a half-done extraction (koboldcpp_cublas.so
# present, its dependencies missing — exactly the state a real one hit,
# see libcublas.so.12: cannot open shared object file) self-heal on its
# next worker start, instead of the "already there" check skipping past
# the fix forever because the one file it used to check for is already
# there.
ENGINE_READY_MARKER = os.path.join(KOBOLD_DIR, ".engine_extraction_complete")
# The files this worker itself ships with (COPYd into the image by the
# Dockerfile) — the source we copy from into KOBOLD_DIR on the volume.
ENGINE_SRC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "koboldcpp_engine")
OUTPUT_DIR = os.path.join(VOLUME_DIR, "outputs")

LORA_DIR = os.path.join(VOLUME_DIR, "loras")

# Holds the Popen handle for the currently running koboldcpp process, if
# any -- set in start_kobold_if_needed(), read by force_kill_kobold().
# Needed because Force Cancellation has to reach across threads (the
# cancel-watch thread inside run_generation, started fresh for every job)
# to kill a process that was launched once, at session start, somewhere
# else entirely.
kobold_process = None
kobold_process_lock = threading.Lock()


def force_kill_kobold():
    """Used by Force Cancellation: kills koboldcpp outright, at any point
    in a generation, not just at a step boundary. Unlike the graceful
    /api/extra/abort_image path, this destroys the whole loaded model in
    VRAM, so the next generation pays the full startup+warmup cost again
    -- see run_session()'s own handling of a force-killed result, which
    restarts and re-warms the engine right away rather than waiting for
    the next job to discover koboldcpp is down.

    The blocked request in run_generation's main thread errors out once
    the process actually dies (connection reset), and its existing
    should_cancel() check in that except branch already reports the
    result as cancelled rather than failed -- this function only has to
    handle the killing itself.
    """
    global kobold_process
    with kobold_process_lock:
        proc = kobold_process
        kobold_process = None
    if proc is None:
        print("Force kill requested but no koboldcpp process handle on record.")
        return
    try:
        proc.kill()
        proc.wait(timeout=10)
        print("koboldcpp process force-killed.")
    except Exception as e:
        print(f"Error force-killing koboldcpp process: {e}")

# Every LoRA below is downloaded once to LORA_DIR on cold start and kept
# there on the persistent volume. koboldcpp's --sdlora flag is pointed at
# the whole directory (not a single file) in start_kobold_if_needed(),
# which makes every LoRA inside it loadable — but only if you type its
# <lora:filename:multiplier> tag into the prompt yourself (see
# run_generation()). Nothing here auto-selects one for you right now, by
# design.
# (Source: LostRuins/koboldcpp wiki, Image Generation page — "--sdlora
# now supports specifying directories as well. All the image LoRAs there
# will be loadable at runtime by using the LoRA syntax in your image
# generation prompt." Worth a quick smoke test after deploying, since this
# wasn't verifiable against the exact source line.)
#
# "multiplier"/"default_steps"/"default_sampler" below aren't read by
# run_generation() right now — they're notes for when this gets wired to a
# "lora" field + Framer button later (see chat).
LORA_CHOICES = {
    "quality": {
        # EMA-ckpt500, 8 steps. Better detail retention and better
        # motion/audio consistency than plain Turbo, still dramatically
        # faster than base H3. Simon's pick to deploy first.
        "filename": "minimax_h3_turbo_ema_ckpt500.safetensors",
        "url": "https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_4step_ema_ckpt500.safetensors",
        "multiplier": 1.0,
        "default_steps": 8,
        "default_sampler": "Euler",
    },
    "fast": {
        # LightX2V, 4 steps @ 0.75 strength. Minimum generation time, more
        # aggressive quality compromise. "er_sde" is the sampler the
        # LightX2V authors recommend for this LoRA (sa_solver is their
        # other suggestion) — not confirmed against koboldcpp's exact
        # sampler list, so test this one and fall back to "Euler" if
        # koboldcpp rejects the sampler name.
        "filename": "minimax_h3_lightx2v_turbo.safetensors",
        "url": "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
        "multiplier": 0.75,
        "default_steps": 4,
        "default_sampler": "er_sde",
    },
}

# Mirrors server.js's own GENERATION_LIMITS — applied here too since the
# session-mode queue (gpu_session_jobs) is inserted straight from the
# browser to Supabase, never passing through server.js at all, so this
# is the one point both the classic and session paths actually share.
# Clamped silently (not rejected) so nothing calling this needs a new
# error case to handle.
GENERATION_LIMITS = {
    "frames": (1, 200),
    "fps": (1, 30),
    "steps": (1, 40),
    "width": (64, 1920),
    "height": (64, 1920),
}


def clamp_generation_params(job_input):
    clamped = dict(job_input)
    for key, (lo, hi) in GENERATION_LIMITS.items():
        value = clamped.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            clamped[key] = max(lo, min(hi, value))
    return clamped


S3_ACCESS_KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("RUNPOD_S3_SECRET_KEY")
S3_ENDPOINT = os.environ.get("RUNPOD_S3_ENDPOINT")
S3_VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID")

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="eur-no-1",
)

# --- Supabase (session mode only) ---------------------------------------
# Used exclusively by the held-open-worker path below: polling for a
# session's queued work, writing per-job progress/results, and reading/
# clearing the session's own control rows. The classic one-shot path
# above never touches any of this — talks to koboldcpp and S3 the same
# way it always has.
#
# Plain REST (PostgREST) over `requests`, not the supabase-py client —
# this runs as a long-lived poll loop inside a GPU container for
# potentially many hours, and a dropped realtime websocket here is a
# worse failure mode than a 3-second HTTP poll. The service-role key
# bypasses RLS entirely, which is exactly why it only ever lives here as
# a RunPod endpoint secret, never anywhere client-facing.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

SESSION_POLL_INTERVAL_SECONDS = 3
# How long a session can sit with no queued work before this worker ends
# it itself and returns, freeing the GPU back to the pool. server.js's
# /api/internal/reap-sessions (pg_cron, every 30s) is the external
# backstop for this — this is the primary, fast path. Applies uniformly
# whether or not the session has ever had a job — a shorter, separate
# threshold just for "never used" sessions was tried and reverted: it
# fired purely on elapsed time with no way to tell "nobody's here" apart
# from "still writing the first prompt," which isn't a distinction
# elapsed time alone can make. 15 minutes covers the same real-money risk
# (an abandoned session left running) without needing that distinction.
SESSION_IDLE_TIMEOUT_SECONDS = 15 * 60
# A real user report exists of RunPod killing jobs at exactly 24 hours
# regardless of a higher configured executionTimeout — unconfirmed, not
# yet tested empirically. Self-returning well before that suspected
# ceiling means a session ends cleanly on our own terms instead of
# finding out the hard way against a live paying session.
SESSION_SAFETY_MAX_SECONDS = 23 * 60 * 60
# How often the loop refreshes its own heartbeat, independent of whether
# there's any real work — see touch_session_heartbeat below.
HEARTBEAT_INTERVAL_SECONDS = 20
# How long the GPU can sit with no real inference before the next
# generation pays a real, measurable slowdown again — see the keep-warm
# ping in run_session()'s idle branch and warmup_kobold()'s own docstring
# for the full explanation. Chosen conservatively short (well under any
# plausible GPU idle-clock-drop window) rather than tuned against hard
# data we don't have — a false-positive ping costs a couple seconds of
# harmless GPU work; a missed one costs the user a slow "fast" generation.
KEEPWARM_INTERVAL_SECONDS = 30


def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def sb_get(table, params, timeout=10):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_sb_headers(),
        params=params,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def sb_patch(table, params, body, timeout=10):
    headers = _sb_headers()
    headers["Prefer"] = "return=representation"
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=headers,
        params=params,
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def sb_delete(table, params, timeout=10):
    r = requests.delete(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_sb_headers(),
        params=params,
        timeout=timeout,
    )
    r.raise_for_status()


class WorkerDeadError(Exception):
    """Raised when koboldcpp can't be brought back up within a reasonable
    wait. Distinct from an ordinary generation failure (a bad prompt,
    koboldcpp rejecting a request, etc.) on purpose: callers need to be
    able to tell "this one generation failed" apart from "this whole
    worker is broken and nothing else should be routed to it" — see
    handler() and run_session()'s own handling of this below."""
    pass


def terminate_worker_soon(delay_seconds=2):
    """Kills this worker process shortly after it returns its result, so
    RunPod can't route another job to a container already known to be
    broken. Relaunching koboldcpp happens inside the SAME container —
    it's not a new GPU or a new RunPod job — so if the underlying crash
    was actually a GPU/driver-level problem (a corrupted CUDA context,
    VRAM fragmentation, an ECC error), restarting the process in place
    may never actually fix it. Rather than let that possibly-broken
    container keep being offered jobs, this ends the process outright
    once a restart has been judged to have failed, forcing RunPod to
    provision a genuinely fresh worker for whatever comes next. The
    short delay gives RunPod's own SDK a moment to actually send the
    response back before the process disappears out from under it."""
    def _die():
        time.sleep(delay_seconds)
        os._exit(1)
    threading.Thread(target=_die, daemon=True).start()


def is_kobold_ready():
    try:
        r = requests.get(f"{KOBOLD_URL}/api/v1/model", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def ensure_koboldcpp_engine():
    """Ensures our fork of koboldcpp.py — and the compiled CUDA backend it
    needs — are both present on the volume, then start_kobold_if_needed()
    runs it as `python3 koboldcpp.py` instead of the official single-file
    frozen binary.

    Why: our fork's only change is one new HTTP route,
    /api/extra/abort_image (see koboldcpp_engine/koboldcpp.py's own header
    comment), added so a generation can be genuinely cancelled on demand —
    upstream koboldcpp already has the underlying native abort
    (handle.sd_abort_generation, safe to call mid-diffusion-step, see
    otherarch/sdcpp/sdtype_adapter.cpp upstream) but only ever fires it
    automatically on client TCP disconnect, never on request. The official
    frozen binary is a PyInstaller bundle with koboldcpp.py's bytecode
    baked in — there's no way to hand it a patched .py file — so running
    the plain, patched script ourselves is the only way to expose that
    route without recompiling anything.

    The CUDA backend (koboldcpp_cublas.so) is NOT rebuilt — that would mean
    a full C++/CUDA toolchain, hours of build time, and real risk of
    getting a subtly different binary than the one this whole pipeline has
    been validated against. Instead it's extracted, byte-for-byte, from
    the official frozen binary: PyInstaller "--onefile" binaries are just
    a self-extracting archive of the exact same files a normal (non-onefile)
    install would have on disk, koboldcpp_cublas.so among them — verified
    locally before writing this by building a throwaway PyInstaller onefile
    binary using the identical `--add-data './koboldcpp_cublas.so:.'`
    pattern koboldcpp's own release build script (koboldcpp.sh) uses, then
    extracting it with pyinstxtractor.py and confirming the extracted file
    is byte-identical (sha256) to the original. Same mechanism, real file,
    just parsed off disk instead of by actually running the binary.
    """
    os.makedirs(KOBOLD_DIR, exist_ok=True)

    # Our patch lives in these small text files, not in the .so — always
    # refresh them from what this worker's own image ships with, so a
    # redeploy with an updated patch takes effect without wiping the
    # volume (the expensive part, the .so, is left alone below if present).
    for fname in ("koboldcpp.py", "json_to_gbnf.py"):
        src = os.path.join(ENGINE_SRC_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(KOBOLD_DIR, fname))
    adapters_src = os.path.join(ENGINE_SRC_DIR, "kcpp_adapters")
    if os.path.isdir(adapters_src):
        shutil.copytree(adapters_src, os.path.join(KOBOLD_DIR, "kcpp_adapters"), dirs_exist_ok=True)

    # embd_res/ isn't just the optional web UI it looked like from reading
    # koboldcpp.py alone (klite/docs/musicui, all wrapped in try/except
    # with a graceful "could not find" fallback) — it also holds tokenizer
    # vocab files the C++ engine itself reads directly, uncaught, for
    # specific model families. Confirmed by a real crash:
    # `std::runtime_error: Failed to open file:
    # .../embd_res/qwen2_merges_utf8_c_str.embd` — needed for the Qwen2
    # text encoder MiniMax H3's CLIP model uses. Rather than cherry-pick
    # which of the ~25 files in here are truly required (this one wasn't
    # obviously guessable from the name alone), just vendor the whole
    # folder — it's only ~70MB total, trivial next to everything else on
    # this volume.
    embd_res_src = os.path.join(ENGINE_SRC_DIR, "embd_res")
    if os.path.isdir(embd_res_src):
        shutil.copytree(embd_res_src, os.path.join(KOBOLD_DIR, "embd_res"), dirs_exist_ok=True)

    if os.path.exists(ENGINE_READY_MARKER):
        print("koboldcpp CUDA backend already present on volume.")
        return

    print("Downloading official koboldcpp binary to extract its CUDA backend from...")
    frozen_path = os.path.join(VOLUME_DIR, "_koboldcpp_frozen_tmp")
    r = requests.get(
        "https://koboldai.org/cpplinuxcu12",
        stream=True, timeout=300
    )
    r.raise_for_status()
    with open(frozen_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    os.chmod(frozen_path, 0o755)
    print("Downloaded. Extracting koboldcpp_cublas.so with pyinstxtractor...")

    extractor = os.path.join(ENGINE_SRC_DIR, "pyinstxtractor.py")
    result = subprocess.run(
        ["python3", extractor, frozen_path],
        cwd=VOLUME_DIR, capture_output=True, text=True, timeout=180,
    )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"pyinstxtractor failed: {result.stderr[-2000:]}")

    extracted_dir = os.path.join(VOLUME_DIR, os.path.basename(frozen_path) + "_extracted")
    extracted_so = os.path.join(extracted_dir, "koboldcpp_cublas.so")
    if not os.path.exists(extracted_so):
        raise RuntimeError(
            f"koboldcpp_cublas.so not found after extraction (looked in {extracted_dir})"
        )

    # koboldcpp_cublas.so alone isn't enough to load — it needs the CUDA
    # runtime libraries it was bundled with (e.g. libcublas.so.12), which
    # sit right alongside it in the same extraction. Copy every .so file
    # found there, not just the one we went looking for — this is what
    # was missing before (extraction only ever kept koboldcpp_cublas.so
    # itself, discarding everything else pyinstxtractor pulled out,
    # which is why loading it crashed with "libcublas.so.12: cannot open
    # shared object file: No such file or directory" on a real worker).
    copied = []
    for fname in os.listdir(extracted_dir):
        full_path = os.path.join(extracted_dir, fname)
        if ".so" in fname and os.path.isfile(full_path):
            shutil.copy(full_path, os.path.join(KOBOLD_DIR, fname))
            os.chmod(os.path.join(KOBOLD_DIR, fname), 0o755)
            copied.append(fname)
    print(f"Copied {len(copied)} shared library file(s) to volume: {copied}")

    with open(ENGINE_READY_MARKER, "w") as f:
        f.write("ok")
    print("koboldcpp CUDA backend extracted and cached on volume.")

    # Only the shared libraries (~hundreds of MB total) are worth keeping —
    # not the multi-GB frozen binary they came from, and not the
    # extraction scratch dir (756 files, mostly Python bytecode we never
    # need since we run our own koboldcpp.py directly).
    for path in (frozen_path, extracted_dir):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Cleanup of {path} failed (non-fatal): {e}")

def ensure_loras():
    os.makedirs(LORA_DIR, exist_ok=True)
    for key, cfg in LORA_CHOICES.items():
        path = os.path.join(LORA_DIR, cfg["filename"])
        if os.path.exists(path):
            print(f"LoRA '{key}' already present on volume.")
            continue
        print(f"Downloading LoRA '{key}' ({cfg['filename']}) to volume...")
        r = requests.get(cfg["url"], stream=True, timeout=300)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"LoRA '{key}' downloaded to volume.")

# Genuinely fresh worker, downloading the multi-GB binary/LoRAs to the
# persistent volume for the very first time ever — this only happens
# once per volume, not once per worker, since every future worker
# (including a brand new one RunPod just allocated) finds them already
# cached. Generous, since a real download-plus-model-load can
# legitimately take a while.
COLD_START_MAX_WAIT_SECONDS = 15 * 60
# The binary/LoRAs are already on the volume — every worker after the
# very first one ever, AND every mid-session crash recovery (the files
# were already there before the crash) — so this is just process start
# plus loading the model into VRAM, no download. Kept short on purpose:
# past this, waiting longer doesn't meaningfully improve the odds of
# recovery, it just quietly burns GPU time on a worker that's likely
# broken — see terminate_worker_soon, called once this is exhausted.
# Was 90s, then 180s, then 300s — raised again because this constant
# conflates two different things. needs_download above only checks
# whether the koboldcpp CUDA engine binary was extracted, not whether
# the SD checkpoint --config points at (the actual multi-GB model file)
# is present — that download happens inside koboldcpp.py itself, via
# its own --config kcppt fetch, invisible to this file. On this
# endpoint specifically, the engine marker is essentially always
# already there (shared volume), but the checkpoint can still be a
# fresh, large, from-scratch download — 300s isn't enough headroom for
# that on top of the network-volume-read slowness already noted below,
# so this is set to match this endpoint's own 6000s execution timeout
# rather than trying to actually distinguish the two cases from here —
# if a job runs this long, RunPod's own timeout is the thing that ends
# it, not this internal wait loop.
#
# Was 90s, then 180s — raised again after real logs showed the
# "already on disk" case is NOT reliably fast: reading
# qwen2_merges_utf8_c_str.embd off the network volume took over 2
# minutes on its own during a real restart (throughput as low as
# 19MB/s at the start of that read, apparently genuine network-volume
# contention rather than anything on our end), blowing straight through
# 180s and crashing the whole worker before it ever got a chance to
# finish. "Files are on disk" turns out not to mean "loads quickly" —
# it can still mean a slow network read away from actually being ready.
WARM_RESTART_MAX_WAIT_SECONDS = 6000
POLL_STEP_SECONDS = 5


def start_kobold_if_needed():
    if is_kobold_ready():
        print("koboldcpp is already running.")
        return

    # Checked BEFORE ensure_koboldcpp_engine()/ensure_loras() run (both
    # are no-ops if the files are already there) so this reflects whether
    # a real download+extraction is actually about to happen, not whether
    # one technically could. The extraction is the expensive one-time part
    # (the .py/.kcpp_adapters copies are cheap and always refreshed).
    needs_download = not os.path.exists(ENGINE_READY_MARKER)

    os.makedirs(VOLUME_DIR, exist_ok=True)
    ensure_koboldcpp_engine()
    ensure_loras()

    print("Starting koboldcpp...")
    # cwd is VOLUME_DIR (the volume's root), NOT KOBOLD_DIR — deliberately.
    # koboldcpp saves whatever --config tells it to download relative to
    # its own working directory, with no separate "models" folder concept.
    # The volume already has a complete, working set of these models
    # sitting at its root from before this fork existed; launching from
    # KOBOLD_DIR instead would make koboldcpp blind to them and re-download
    # everything (12.9GB + 17GB + 4.9GB + 0.6GB) a second time into the new
    # subfolder — which is exactly what filled a real volume's disk quota.
    # getdirpath()/init_library() in koboldcpp.py resolve relative to the
    # script's own file location, not cwd, so this has no effect on finding
    # koboldcpp_cublas.so — only on where downloaded models land.
    # The official frozen binary's PyInstaller bootloader sets up library
    # search paths (so koboldcpp_cublas.so's own dependencies, like
    # libcublas.so.12, get found next to it) before Python code ever runs.
    # Running koboldcpp.py directly with plain `python3` skips that
    # bootloader entirely, so nothing tells the system to look in
    # KOBOLD_DIR for those dependencies — confirmed by a real worker
    # crashing with "libcublas.so.12: cannot open shared object file" even
    # though that exact file was sitting right there, next to
    # koboldcpp_cublas.so, the whole time. LD_LIBRARY_PATH is the standard
    # way to tell the dynamic linker to also search a directory.
    kobold_env = os.environ.copy()
    existing_ld_path = kobold_env.get("LD_LIBRARY_PATH", "")
    kobold_env["LD_LIBRARY_PATH"] = (
        f"{KOBOLD_DIR}:{existing_ld_path}" if existing_ld_path else KOBOLD_DIR
    )
    proc = subprocess.Popen([
        "python3", KOBOLD_PY,
        "--quiet",
        # SECURITY: koboldcpp's own --host default ("") binds every
        # routable interface, not just loopback — despite KOBOLD_URL
        # above implying this server is localhost-only, without this
        # flag it genuinely was not. Nothing here ever configures
        # --password/--adminpassword either, and koboldcpp's own
        # secure_endpoint() only enforces auth at all when one is set,
        # so every route (generation, the abort_image route this fork
        # adds, etc.) was reachable with zero credentials from anywhere
        # that could reach this container's network. handler.py only
        # ever needs to reach koboldcpp from within this same process's
        # own container (KOBOLD_URL is 127.0.0.1), so binding strictly
        # to loopback costs nothing and closes that off regardless of
        # whatever the surrounding network/firewall assumptions are.
        "--host", "127.0.0.1",
        "--config",
        KCPPT_CONFIG_URL,
        "--sdlora", LORA_DIR,
    ], cwd=VOLUME_DIR, env=kobold_env)
    global kobold_process
    with kobold_process_lock:
        kobold_process = proc

    max_wait = COLD_START_MAX_WAIT_SECONDS if needs_download else WARM_RESTART_MAX_WAIT_SECONDS
    attempts = max_wait // POLL_STEP_SECONDS
    for _ in range(attempts):
        if is_kobold_ready():
            print("koboldcpp is ready.")
            return
        time.sleep(POLL_STEP_SECONDS)

    raise RuntimeError(f"koboldcpp did not become ready within {max_wait}s")


def warmup_kobold(session_id, reason="startup", size=512):
    """Fires one tiny throwaway generation, discarded and never uploaded —
    used three ways: once right after koboldcpp reports ready, before the
    session is marked ready for real use (see run_session()); again if
    koboldcpp has to be restarted mid-session (e.g. after a Force
    Cancellation kill); and periodically during idle stretches inside
    run_session()'s poll loop (see KEEPWARM_INTERVAL_SECONDS) to stop the
    same slowdown from coming back mid-session.

    Found by comparing two back-to-back real generations in the same
    session with identical prompt/settings: the first took 3m41s, the
    second 2m21s — a ~80s gap with no koboldcpp restart involved (checked
    gpu_session_jobs.output directly, no kobold_restarted flag). The most
    likely explanation is one-time GPU/CUDA warm-up on the very first
    real inference through a freshly-loaded model — CUDA kernel JIT
    compilation, VRAM allocator growth, flash-attention kernel selection
    — all things that happen inside koboldcpp's own compiled binary, not
    in anything this file controls, and none of which show up as an
    error since nothing is actually wrong.

    A later report showed the SAME slowdown coming back mid-session, with
    the worker held open the whole time (no restart, no re-warm needed by
    any check this file already does) — the one thing that had changed
    was a long idle gap with no real generation in between. That points at
    something that decays with idle TIME specifically, not something a
    one-off warmup at session start can fully cover: most likely the GPU
    driver dropping to a low-power idle clock state after a stretch with
    no active kernels, then ramping back up on the next one — separate
    from, and in addition to, the one-time JIT/allocator cost above. The
    periodic keep-warm ping exists to stop the GPU from ever sitting idle
    long enough for that to matter, by giving it a trivial kernel to run
    every KEEPWARM_INTERVAL_SECONDS instead — and since all it needs to do
    is prove recent GPU activity, not exercise the same kernel/shape
    selection a real generation does, its call site passes a much smaller
    size than the startup call's default. size must stay a multiple of
    32 either way — the model's own dimension requirement (same
    constraint the site's own resolution picker snaps every real quality
    preset to); 1x1 was considered and ruled out for exactly this reason.

    Rather than let the user's first (or next, after a lull) real prompt
    silently pay that cost, this pays it here instead — the whole point of
    pre-warming a session is defeated if "ready" doesn't actually mean the
    next generation will be fast. Best-effort in all three uses: koboldcpp
    already proved itself responsive via is_kobold_ready() before the
    startup call, and the other two calls only ever run once the session
    is already confirmed working, so a failure here doesn't mean the
    worker is broken — it just means the next real generation pays the
    warm-up cost instead of it being hidden here.
    """
    payload = {
        "prompt": "",
        "negative_prompt": "",
        "width": size,
        "height": size,
        "steps": 1,
        "cfg_scale": 1,
        "sampler_name": "Euler",
        "seed": -1,
        "frames": 1,
        "fps": 24,
        "video_output_type": 1,
        "denoising_strength": 0.6,
        "enable_hr": False,
        "init_images": [],
        "extra_images": [],
        "send_as_refimg": False,
        "inpainting_fill": None,
        "inpainting_mask_invert": None,
        "genkey": f"warmup_{session_id}",
        "keepalive": True,
        "kcpp_extra_args": {"keep_image_gen_on_disconnect": True},
    }
    started = time.time()
    try:
        requests.post(f"{KOBOLD_URL}/sdapi/v1/txt2img", json=payload, timeout=300)
        elapsed = round(time.time() - started, 1)
        print(f"Session {session_id}: warm-up generation done ({reason}, {elapsed}s).")
        return elapsed
    except Exception as e:
        print(f"Session {session_id}: warm-up generation failed, continuing anyway ({reason}, {e}).")
        return None


# NOT called eagerly here anymore. It used to run unprotected at module
# import time — meaning if it ever raised (which tonight it genuinely
# did: a slow network-volume read pushed a "files already on disk"
# restart past even the 180s warm ceiling), NOTHING caught it. The whole
# Python process crashed with exit code 1 before run_session() or
# handler() ever ran, before session_id was even known, so none of our
# careful graceful-shutdown code (mark_session_ended, terminate_worker_
# soon, a clean Supabase write) ever got a chance to run — RunPod's own
# infrastructure just silently restarted a fresh container, invisible to
# everything we built to handle exactly this. Moved into run_session()
# (session path) and left to run_generation()'s own existing self-heal
# (classic path) instead, where session_id is known and a failure can be
# handled the same deliberate way every OTHER worker-death case already
# is. See run_session()'s own startup for where this moved to.

# TEMPORARY diagnostic instrumentation — investigating whether sdoffloadcpu
# (the .kcppt config swaps the ~32B qwen3vl sdclip1 prompt encoder between
# CPU and GPU instead of keeping it resident in VRAM) is costing real time
# on a generation, by sampling nvidia-smi alongside koboldcpp's own
# reported sampling step. Both land in gpu_session_jobs.output (see
# run_generation's timing()) so they can be queried and correlated
# straight from Supabase — no RunPod container-log archaeology, which the
# comment on kobold_call_started above already found unreliable (buffered,
# non-monotonic timestamps). Safe to delete once this investigation is
# done: it adds two extra fields to the job output and nothing else reads
# them.
def sample_gpu_stats(stop_event, samples, interval=1.0):
    """Best-effort periodic `nvidia-smi` snapshot for the duration of one
    generation. Each sample is [unix_timestamp, gpu_util_pct, vram_used_mb].
    A GPU utilization that drops to (or near) 0% for a sustained stretch
    while VRAM usage stays flat or shifts is the signature of compute
    actually happening on the CPU (or of a CPU<->GPU weight transfer)
    instead of on the GPU — exactly what sdoffloadcpu would produce for
    whichever component it's offloading. Never raises: a missing/failing
    nvidia-smi must never break or slow down a real generation, only
    silently skip that one sample."""
    while not stop_event.wait(interval):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            util_str, mem_str = out.stdout.strip().split(",")
            samples.append([round(time.time(), 1), int(util_str.strip()), int(mem_str.strip())])
        except Exception:
            pass


def poll_progress(genkey, stop_event, on_progress, history=None):
    """Polls koboldcpp's own progress endpoint and hands each update to
    on_progress — the classic path forwards these straight to RunPod's
    own progress channel; session mode writes them into the job's own
    Supabase row instead, since there's no per-generation RunPod job to
    report progress on anymore.

    history: optional list this also appends [unix_timestamp,
    sampling_step, sampling_steps] to, independent of on_progress and its
    overwrite-only Supabase write — see sample_gpu_stats' own docstring
    for why: on_progress alone only ever leaves the FINAL progress state
    behind, losing the timeline needed to correlate against gpu_util_
    samples. TEMPORARY, same investigation as sample_gpu_stats above.
    """
    while not stop_event.is_set():
        try:
            r = requests.get(
                f"{KOBOLD_URL}/sdapi/v1/progress",
                params={"skip_current_image": "true", "genkey": genkey},
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                on_progress(data)
                if history is not None:
                    history.append([round(time.time(), 1), data.get("sampling_step"), data.get("sampling_steps")])
        except Exception as e:
            print(f"Progress poll error: {e}")
        time.sleep(2)

def detect_extension_and_content_type(raw_bytes):
    """Sniff the real file type from its magic bytes, regardless of what koboldcpp claims."""
    if raw_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", "image/gif"
    if raw_bytes[4:8] == b"ftyp":
        return "mp4", "video/mp4"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "webp", "image/webp"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"AVI ":
        return "avi", "video/x-msvideo"
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png", "image/png"
    if raw_bytes[:3] == b"\xff\xd8\xff":
        return "jpg", "image/jpeg"
    return "bin", "application/octet-stream"

def convert_avi_to_mp4(avi_path):
    """
    Converts an AVI file to MP4 locally using ffmpeg's software encoder.
    (NVENC hardware encoding was tried and removed — RunPod's multi-GPU host
    machines hit a known NVIDIA driver bug that breaks NVENC/NVDEC inside
    containers, so attempting it just wastes time before failing anyway.)
    Returns the path to the new MP4 file, or None if conversion failed.
    """
    mp4_path = avi_path.rsplit(".", 1)[0] + ".mp4"

    software_cmd = [
        "ffmpeg", "-y",
        "-i", avi_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        mp4_path,
    ]

    try:
        print("Converting AVI to MP4 (libx264)...")
        result = subprocess.run(software_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(mp4_path):
            print("Conversion succeeded.")
            return mp4_path
        print(f"Conversion failed (code {result.returncode}): {result.stderr[-800:]}")
    except Exception as e:
        print(f"Conversion raised an exception: {e}")

    return None

def upload_result_and_get_key(base64_data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_bytes = base64.b64decode(base64_data)

    ext, content_type = detect_extension_and_content_type(raw_bytes)
    print(f"Detected output format: {ext} ({content_type}) — magic bytes: {raw_bytes[:12]}")

    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(raw_bytes)

    # If this is an AVI (video with audio), convert it to MP4 locally before
    # uploading, so the browser can actually play it natively.
    if ext == "avi":
        mp4_path = convert_avi_to_mp4(filepath)
        if mp4_path:
            filepath = mp4_path
            filename = os.path.basename(mp4_path)
            content_type = "video/mp4"
        else:
            print("AVI to MP4 conversion failed entirely — uploading original AVI instead.")

    key = f"outputs/{filename}"
    s3_client.upload_file(
        filepath, S3_VOLUME_ID, key,
        ExtraArgs={"ContentType": content_type}
    )
    return filename


def run_generation(job_input, genkey, on_progress, should_cancel=None, should_force_kill=None):
    """The actual generation, shared by both the classic one-shot path and
    the session loop below. Builds the koboldcpp payload, runs it, uploads
    the result. Returns {"videoKey": ...} or {"error": ...} — exactly the
    same shape run_generation's caller already returned before this was
    split out, so the classic path's behavior is unchanged byte-for-byte.

    should_cancel: optional zero-arg callable, polled every ~0.75s on a
    background thread for the whole time this function is blocked inside
    its synchronous call to koboldcpp. The instant it returns True, that
    thread fires our koboldcpp fork's /api/extra/abort_image route — a
    genuine on-demand abort (see koboldcpp_engine/koboldcpp.py and
    ensure_koboldcpp_engine()'s docstring for why that route exists at
    all). Classic (non-session) jobs pass nothing here: there's no
    concept of "cancel this specific queued job" outside session mode,
    so the classic path's behavior is exactly what it always was.

    should_force_kill: optional zero-arg callable, checked first (ahead
    of should_cancel) on that same background thread. When it returns
    True, the thread kills the koboldcpp process outright instead of
    asking it nicely — see force_kill_kobold()'s own docstring for why
    that's a real, different capability (stops at any point, not just a
    step boundary) and what it costs (the whole engine has to reload).
    The returned dict's "force_killed" key tells run_session() whether
    this happened, so it knows to proactively restart the engine.

    No field-based LoRA selection, on purpose: --sdlora points at the
    whole LORA_DIR (see start_kobold_if_needed), so every file in
    LORA_CHOICES is loadable, but nothing gets auto-injected here. The
    prompt is sent exactly as given — you activate a LoRA by typing its
    tag straight into the prompt yourself, e.g.
      <lora:minimax_h3_turbo_ema_ckpt500:1.0> an animated horse wrestles a bear
      <lora:minimax_h3_lightx2v_turbo:0.75> ...
    (filenames without the .safetensors extension, per LORA_CHOICES
    above). Type no tag and no LoRA applies at all.
    """
    # If koboldcpp has crashed (e.g. a CUDA OOM from an oversized request
    # on an earlier job) since this worker's initial cold start, nothing
    # else ever notices or restarts it — start_kobold_if_needed() only
    # ever ran once, at import time. Left alone, every subsequent job on
    # this worker fails immediately and, in session mode, silently for
    # the rest of the session (the heartbeat only proves this Python
    # process is alive, not koboldcpp — see touch_session_heartbeat).
    # start_kobold_if_needed() is already safe to call again: it checks
    # readiness first and only restarts if actually needed.
    # Merged into whatever dict this function ends up returning (every
    # return point below) — previously this only ever showed up as one
    # easy-to-miss print line in raw container logs. Now it rides along
    # in gpu_session_jobs.output (already a jsonb column, no migration
    # needed) so "was this specific generation slow because of a restart"
    # is a normal query instead of log archaeology, and run_session()
    # below rolls it into the session's own final summary too.
    generation_started = time.time()
    restart_info = {}
    if not is_kobold_ready():
        print("koboldcpp not responding — attempting restart before this generation.")
        restart_started = time.time()
        try:
            start_kobold_if_needed()
        except Exception as e:
            # Raised, not returned — a plain {"error": ...} return here
            # would look identical to an ordinary generation failure (a
            # bad prompt, koboldcpp rejecting a request) to both callers
            # below, and neither should react to those two cases the
            # same way. See WorkerDeadError's own docstring.
            raise WorkerDeadError(str(e)) from e
        restart_info = {
            "kobold_restarted": True,
            "restart_seconds": round(time.time() - restart_started, 1),
        }

    job_input = clamp_generation_params(job_input)

    payload = {
        "prompt": job_input.get("prompt", ""),
        "negative_prompt": job_input.get("negative_prompt", ""),
        "width": job_input.get("width", 512),
        "height": job_input.get("height", 512),
        "steps": job_input.get("steps", 20),
        "cfg_scale": job_input.get("cfg_scale", 1),
        "sampler_name": job_input.get("sampler_name", "Euler"),
        "seed": job_input.get("seed", -1),
        "frames": job_input.get("frames", 30),
        "fps": job_input.get("fps", 16),
        # 1, not 2: matches what the real site frontend always sends
        # (GeneratorPanel.tsx) and what warmup_kobold above hardcodes — 2
        # only ever gets hit by a job fired directly at this worker without
        # the field set (e.g. a manual speed/quality-lab test), and it
        # produces a GIF instead of the MP4 every real generation gets.
        "video_output_type": job_input.get("video_output_type", 1),
        "denoising_strength": job_input.get("denoising_strength", 0.6),
        "enable_hr": job_input.get("enable_hr", False),
        "init_images": job_input.get("init_images", []),
        "extra_images": job_input.get("extra_images", []),
        "send_as_refimg": job_input.get("send_as_refimg", False),
        "inpainting_fill": job_input.get("inpainting_fill"),
        "inpainting_mask_invert": job_input.get("inpainting_mask_invert"),
        "genkey": genkey,
        "keepalive": True,
        "kcpp_extra_args": {
            "keep_image_gen_on_disconnect": True
        }
    }

    # Only include scheduler if explicitly provided — SDUI's own confirmed
    # working request never sends this field at all when left at default,
    # so we match that instead of always forcing "discrete" in.
    if "scheduler" in job_input:
        payload["scheduler"] = job_input["scheduler"]

    if "video_start_frame" in job_input:
        payload["video_start_frame"] = job_input["video_start_frame"]
    if "video_end_frame" in job_input:
        payload["video_end_frame"] = job_input["video_end_frame"]

    debug_payload = {}
    for k, v in payload.items():
        if k in ("extra_images", "init_images"):
            debug_payload[k] = [f"<{len(x)} chars>" for x in v]
        elif k in ("video_start_frame", "video_end_frame"):
            debug_payload[k] = f"<{len(v)} chars>"
        else:
            debug_payload[k] = v
    print(f"Final payload to koboldcpp: {debug_payload}")

    stop_event = threading.Event()
    # TEMPORARY: see sample_gpu_stats'/poll_progress's own docstrings —
    # investigating whether sdoffloadcpu is costing real time on a
    # generation. Both lists are folded into timing()'s returned dict
    # below, so they land in gpu_session_jobs.output for free.
    progress_history = []
    gpu_samples = []
    gpu_monitor_stop = threading.Event()
    gpu_monitor_thread = threading.Thread(target=sample_gpu_stats, args=(gpu_monitor_stop, gpu_samples))
    gpu_monitor_thread.start()
    progress_thread = threading.Thread(
        target=poll_progress, args=(genkey, stop_event, on_progress), kwargs={"history": progress_history}
    )
    progress_thread.start()

    # Watches should_cancel() while the main thread is blocked inside the
    # synchronous txt2img POST below — that POST won't return until
    # koboldcpp itself stops generating, so cancellation has to happen on
    # a second connection, concurrently. koboldcpp's HTTP server already
    # proves this pattern works today, for text: /api/extra/abort accepts
    # a request while a completely separate generate request is still in
    # flight on the same running server.
    cancel_watch_stop = threading.Event()
    cancel_thread = None
    force_killed_flag = {"value": False}
    if should_cancel is not None:
        def watch_for_cancel():
            # Deliberately does NOT return once the graceful abort has been
            # sent — koboldcpp only actually stops at the next step
            # boundary, which can be a real wait on a large generation, and
            # the user can escalate to Force Cancellation at any point
            # during that wait. This loop has to stay alive the whole time
            # to see that happen. Returning right after the graceful abort
            # was the bug that made Force Cancellation silently do nothing:
            # this thread would already be gone by the time
            # force_cancel_requested got set a few seconds later, so nothing
            # was left polling for it.
            graceful_sent = False
            while not cancel_watch_stop.wait(0.75):
                try:
                    if should_force_kill is not None and should_force_kill():
                        print(f"Force cancel requested for genkey {genkey} — killing koboldcpp process.")
                        force_kill_kobold()
                        force_killed_flag["value"] = True
                        return
                    if not graceful_sent and should_cancel():
                        print(f"Cancel requested for genkey {genkey} — aborting in-flight generation.")
                        # Scoped to this exact generation now that
                        # koboldcpp.py's abort_image route checks genkey
                        # against currgenimgkey (see that route's own
                        # comment) — matters if this worker ever processes
                        # more than one generation concurrently through the
                        # same koboldcpp process; harmless today.
                        requests.post(f"{KOBOLD_URL}/api/extra/abort_image", json={"genkey": genkey}, timeout=10)
                        graceful_sent = True
                except Exception as e:
                    print(f"Cancel watch error for genkey {genkey} (non-fatal): {e}")
        cancel_thread = threading.Thread(target=watch_for_cancel, daemon=True)
        cancel_thread.start()

    # Real wall-clock timing, independent of container log timestamps —
    # RunPod's log capture block-buffers this process's stdout (it's
    # piped, not a terminal), so print() calls that are genuinely minutes
    # apart in real time can show up flushed together with an identical
    # timestamp. That made an earlier "why was this generation slower"
    # investigation impossible to settle from logs alone. This rides
    # along in the same dict every return point below already returns,
    # so it lands in gpu_session_jobs.output for free — a real, trustworthy
    # number instead of log archaeology.
    kobold_call_started = time.time()
    kobold_seconds = None  # set the instant a response (good or bad) actually arrives

    def timing():
        return {
            # None until a response arrives at all (e.g. the request itself
            # timed out) — kept distinct from a real, small number.
            "kobold_seconds": kobold_seconds,
            "total_seconds": round(time.time() - generation_started, 1),
            # TEMPORARY diagnostic fields -- see sample_gpu_stats'/
            # poll_progress's own docstrings. [unix_timestamp, ...] rows,
            # queryable straight from gpu_session_jobs.output.
            "gpu_util_samples": gpu_samples,
            "progress_history": progress_history,
        }

    try:
        response = requests.post(
            f"{KOBOLD_URL}/sdapi/v1/txt2img",
            json=payload,
            timeout=2500
        )
        kobold_seconds = round(time.time() - kobold_call_started, 1)

        # Checked before interpreting the response at all: an aborted
        # generation's response (whatever koboldcpp happens to have
        # returned — an error, empty images, or stale partial data from
        # the exact instant it was cut off) is meaningless either way, and
        # must never be uploaded or shown as a real result. The flag is
        # the one thing we actually trust here.
        if should_cancel is not None and should_cancel():
            stop_event.set()
            progress_thread.join(timeout=5)
            return {**restart_info, **timing(), "cancelled": True, "force_killed": force_killed_flag["value"]}

        if not response.ok:
            try:
                kobold_error = response.json()
            except Exception:
                kobold_error = response.text[:2000]
            stop_event.set()
            progress_thread.join(timeout=5)
            return {
                **restart_info,
                **timing(),
                "error": f"koboldcpp returned HTTP {response.status_code}",
                "kobold_error": kobold_error,
            }

        result = response.json()

        stop_event.set()
        progress_thread.join(timeout=5)

        if result.get("images") and result["images"][0]:
            filename = upload_result_and_get_key(result["images"][0])
            return {**restart_info, **timing(), "videoKey": filename}
        elif result.get("final_frame"):
            filename = upload_result_and_get_key(result["final_frame"])
            return {**restart_info, **timing(), "videoKey": filename}
        else:
            return {**restart_info, **timing(), "error": "No image data returned", "raw": result}

    except Exception as e:
        if should_cancel is not None and should_cancel():
            return {**restart_info, **timing(), "cancelled": True, "force_killed": force_killed_flag["value"]}
        return {**restart_info, **timing(), "error": str(e)}
    finally:
        stop_event.set()
        progress_thread.join(timeout=5)
        cancel_watch_stop.set()
        if cancel_thread:
            cancel_thread.join(timeout=3)
        gpu_monitor_stop.set()
        gpu_monitor_thread.join(timeout=3)


# --- Session mode (held-open worker) -------------------------------------

def is_session_active(session_id):
    """A session is active only while gpu_sessions.ended_at is still null
    AND its active_gpu_sessions claim row still exists. Checking both
    covers every way a session can be told to stop: server.js's manual
    Stop route sets ended_at and deletes the claim row together;
    session-reaper.js's hard-cancel path may only manage to do one or the
    other depending on what state it found. Either signal alone is enough
    to end this loop."""
    try:
        sessions = sb_get(
            "gpu_sessions",
            {"id": f"eq.{session_id}", "select": "ended_at"},
        )
        if not sessions or sessions[0].get("ended_at") is not None:
            return False

        claims = sb_get(
            "active_gpu_sessions",
            {"session_id": f"eq.{session_id}", "select": "user_id"},
        )
        return len(claims) > 0
    except Exception as e:
        print(f"Could not check session state (treating as still active): {e}")
        # A transient Supabase/network hiccup shouldn't end a real,
        # paying session on the spot — same reasoning as railway.tsx's
        # own consecutive-failures tolerance for its status polling.
        return True


def touch_session_heartbeat(session_id):
    """Refreshes active_gpu_sessions.last_activity_at so session-reaper.js
    can tell a worker whose loop is genuinely alive (idling on purpose,
    waiting for queued work) apart from one that's actually dead or
    hung — a crashed container can't be trusted to report its own death,
    so the reaper needs a freshness signal it doesn't depend on the
    worker judging correctly. This is separate from the idle-timeout
    logic in run_session() below, which decides when to end an idle
    session on purpose; this just proves the loop is still running at
    all, regardless of whether it currently has work."""
    try:
        sb_patch(
            "active_gpu_sessions",
            {"session_id": f"eq.{session_id}"},
            {"last_activity_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        )
    except Exception as e:
        print(f"Could not write heartbeat for session {session_id}: {e}")


def mark_session_worker_started(session_id):
    """Flips active_gpu_sessions.worker_started_at from null to now() the
    moment this loop actually begins. Thanks to start_kobold_if_needed()
    running at module import time (see the bottom of this file), before
    RunPod's SDK will even register this worker as available to receive a
    job, koboldcpp is already guaranteed warm by the time this runs — so
    this single flag is enough for railway.tsx to show a real
    starting-vs-ready state, instead of treating the session row's mere
    existence (written the instant server.js's /api/session/start returns,
    long before RunPod has actually found/booted a worker) as if it meant
    ready."""
    try:
        sb_patch(
            "active_gpu_sessions",
            {"session_id": f"eq.{session_id}"},
            {"worker_started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        )
    except Exception as e:
        print(f"Could not mark worker started for session {session_id}: {e}")


def reset_worker_started(session_id):
    """The inverse of mark_session_worker_started() -- flips
    worker_started_at back to null. Used when Force Cancellation kills
    koboldcpp outright: the engine genuinely isn't ready anymore for the
    seconds it takes to restart, and railway.tsx's workerReady flag
    (driven by this same column) needs to reflect that honestly, showing
    the starting state again instead of continuing to claim ready for a
    worker that's momentarily dead."""
    try:
        sb_patch(
            "active_gpu_sessions",
            {"session_id": f"eq.{session_id}"},
            {"worker_started_at": None},
        )
    except Exception as e:
        print(f"Could not reset worker_started for session {session_id}: {e}")


def mark_session_ended(session_id, reason):
    try:
        sb_patch(
            "gpu_sessions",
            {"id": f"eq.{session_id}", "ended_at": "is.null"},
            {"ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "end_reason": reason},
        )
    except Exception as e:
        print(f"Could not mark session {session_id} ended: {e}")
    try:
        sb_delete("active_gpu_sessions", {"session_id": f"eq.{session_id}"})
    except Exception as e:
        print(f"Could not release active_gpu_sessions claim for {session_id}: {e}")


def claim_next_queued_job(session_id):
    """Finds the oldest queued job for this session and atomically claims
    it (PATCH ... WHERE status = 'queued' — an empty result means someone/
    something else already claimed it first). In practice this worker is
    the only thing ever draining this specific session's queue, so the
    race this guards against shouldn't happen — kept anyway because it's
    free correctness, not because it's expected to matter."""
    queued = sb_get(
        "gpu_session_jobs",
        {
            "session_id": f"eq.{session_id}",
            "status": "eq.queued",
            "order": "created_at.asc",
            "limit": "1",
        },
    )
    if not queued:
        return None

    job_row = queued[0]
    claimed = sb_patch(
        "gpu_session_jobs",
        {"id": f"eq.{job_row['id']}", "status": "eq.queued"},
        {"status": "processing"},
    )
    if not claimed:
        return None
    return claimed[0]


def write_job_progress(job_row_id, data):
    try:
        sb_patch("gpu_session_jobs", {"id": f"eq.{job_row_id}"}, {"progress": data})
    except Exception as e:
        print(f"Could not write progress for job {job_row_id}: {e}")


def is_job_cancel_requested(job_row_id):
    """Polled by run_generation's cancel-watch thread (see should_cancel
    there) every ~1.5s for as long as a generation is in flight. Set by
    server.js's POST /api/session/job/:jobId/cancel — a plain Supabase
    flag, not a RunPod API call, since the worker that would need to act
    on it is mid-request to koboldcpp, not listening for anything else."""
    try:
        rows = sb_get(
            "gpu_session_jobs",
            {"id": f"eq.{job_row_id}", "select": "cancel_requested"},
        )
        return bool(rows and rows[0].get("cancel_requested"))
    except Exception as e:
        print(f"Could not check cancel flag for job {job_row_id}: {e}")
        return False


def is_job_force_cancel_requested(job_row_id):
    """Sibling of is_job_cancel_requested for Force Cancellation. Checked
    first, ahead of the graceful flag, in run_generation's cancel-watch
    thread -- force wins if both are somehow set."""
    try:
        rows = sb_get(
            "gpu_session_jobs",
            {"id": f"eq.{job_row_id}", "select": "force_cancel_requested"},
        )
        return bool(rows and rows[0].get("force_cancel_requested"))
    except Exception as e:
        print(f"Could not check force cancel flag for job {job_row_id}: {e}")
        return False


def finish_job(job_row_id, output):
    if output.get("cancelled"):
        status = "cancelled"
    elif output.get("error"):
        status = "failed"
    else:
        status = "completed"
    try:
        sb_patch(
            "gpu_session_jobs",
            {"id": f"eq.{job_row_id}"},
            {"status": status, "output": output},
        )
    except Exception as e:
        print(f"Could not write final result for job {job_row_id}: {e}")


def run_session(session_id):
    """The held-open loop. Runs entirely inside one RunPod job that never
    returns until the session itself ends — that's what keeps this worker
    excluded from RunPod's pool of workers eligible for anyone else's
    /run call for the session's whole lifetime. New generation requests
    can't reach an already-busy worker through RunPod's own routing, so
    they arrive here by polling Supabase instead (see claim_next_queued_job).

    RunPod's own per-job retry/backpressure only ever covered this ONE
    outer job — it has no visibility into the many generations processed
    inside this loop, so each one gets its own try/except below rather
    than assuming a crash here gets retried the way a classic per-click
    job would have been.
    """
    session_start = time.time()
    last_activity = time.time()
    last_heartbeat = 0.0
    # Separate from last_activity (which drives the session idle-timeout
    # and must only move on REAL use, or a session would never time out
    # once keep-warm pings start) — this tracks the last time the GPU
    # actually ran anything, startup warmup and keep-warm pings included,
    # so the idle branch below knows when it's genuinely due for another
    # one. Set for real right after the startup warmup call below; the
    # placeholder here just keeps the name defined before that point.
    last_gpu_touch = time.time()
    jobs_processed = 0
    kobold_restarts = 0
    print(f"Session {session_id}: held-open loop starting.")

    # Ensuring koboldcpp is actually up now lives here instead of an
    # unprotected call at module import time (see that former call site's
    # comment for why) — this is the first real chance to handle a
    # startup failure the same deliberate way every other worker-death
    # case in this file already is: a clean session-ended write and a
    # controlled self-termination, instead of an uncaught crash RunPod's
    # own infrastructure has to silently clean up after. mark_session_
    # worker_started() only fires once this actually succeeds — the
    # frontend's "ready" (green) state depends on that ordering.
    try:
        start_kobold_if_needed()
    except Exception as e:
        print(f"Session {session_id}: koboldcpp failed to start ({e}) — ending session.")
        mark_session_ended(session_id, "error")
        terminate_worker_soon()
        return {
            "sessionEnded": True,
            "reason": "worker_error",
            "session_id": session_id,
            "jobs_processed": 0,
            "kobold_restarts": 0,
            "session_duration_seconds": round(time.time() - session_start, 1),
            "last_error": str(e),
        }

    # Absorbs the one-time GPU/CUDA first-inference warm-up cost here,
    # during the "Starting..." wait, instead of the user's first real
    # prompt — see warmup_kobold's own docstring for the investigation
    # that found this. Deliberately best-effort and never raises: koboldcpp
    # already proved responsive above, so this failing doesn't mean the
    # worker is broken.
    warmup_seconds = warmup_kobold(session_id)
    last_gpu_touch = time.time()

    mark_session_worker_started(session_id)

    # RunPod's own dashboard only ever shows this outer job's return value
    # once, at the very end — every generation processed inside the loop
    # is otherwise invisible there (the real per-job detail lives in
    # gpu_session_jobs.output in Supabase instead). This one dict is the
    # only chance to make that final RunPod-side view actually useful for
    # a quick glance, so every return below carries the same base fields
    # plus whatever's specific to how it ended.
    def session_summary(reason, **extra):
        summary = {
            "sessionEnded": True,
            "reason": reason,
            "session_id": session_id,
            "jobs_processed": jobs_processed,
            "kobold_restarts": kobold_restarts,
            "warmup_seconds": warmup_seconds,
            "session_duration_seconds": round(time.time() - session_start, 1),
        }
        summary.update(extra)
        return summary

    while True:
        now = time.time()
        if now - last_heartbeat > HEARTBEAT_INTERVAL_SECONDS:
            touch_session_heartbeat(session_id)
            last_heartbeat = now

        if now - session_start > SESSION_SAFETY_MAX_SECONDS:
            print(f"Session {session_id}: hit the {SESSION_SAFETY_MAX_SECONDS}s safety cutoff, ending.")
            mark_session_ended(session_id, "error")
            return session_summary("safety_timeout")

        if not is_session_active(session_id):
            print(f"Session {session_id}: no longer active, ending loop.")
            return session_summary("stopped")

        try:
            job_row = claim_next_queued_job(session_id)
        except Exception as e:
            print(f"Session {session_id}: could not check queue: {e}")
            job_row = None

        if job_row is None:
            if time.time() - last_activity > SESSION_IDLE_TIMEOUT_SECONDS:
                print(f"Session {session_id}: idle past {SESSION_IDLE_TIMEOUT_SECONDS}s, ending.")
                mark_session_ended(session_id, "timeout")
                return session_summary("timeout")
            # Keeps the GPU from sitting genuinely idle long enough to
            # lose the warmup's benefit — see warmup_kobold's own
            # docstring for what this is working around. Piggybacks on
            # this same poll tick rather than its own timer/thread.
            #
            # This blocks the loop (same thread that later calls
            # claim_next_queued_job/run_generation) for however long the
            # throwaway gen takes — typically well under a second, since
            # it's 1 step at the model's 32x32 minimum. A real job queued
            # while this is in flight isn't lost or corrupted (koboldcpp
            # only ever runs one generation at a time regardless, so it
            # would have had to wait anyway) — it just sits in 'queued' a
            # little longer. Re-checking the queue immediately after,
            # instead of also sleeping the full poll interval on top,
            # keeps that added wait as small as it can be. Deliberately
            # NOT touching last_activity here — that clock drives the
            # session idle-timeout above and must only move on real use,
            # or a session with keep-warm pings running would never time
            # out on its own.
            if time.time() - last_gpu_touch > KEEPWARM_INTERVAL_SECONDS:
                warmup_kobold(session_id, reason="keep-warm", size=32)
                last_gpu_touch = time.time()
                continue
            time.sleep(SESSION_POLL_INTERVAL_SECONDS)
            continue

        genkey = f"session_{session_id}_job_{job_row['id']}"
        print(f"Session {session_id}: processing job {job_row['id']}.")

        # A job can be cancelled while it was still queued (e.g. the user
        # queued a second prompt, then cancelled it before this worker
        # even reached it) — claim_next_queued_job's PATCH already
        # returns the full row, cancel_requested included, so this is
        # free: skip koboldcpp entirely rather than start a generation
        # only to abort it on the very first cancel-watch tick.
        if job_row.get("cancel_requested"):
            print(f"Session {session_id}: job {job_row['id']} was cancelled before it started — skipping.")
            finish_job(job_row["id"], {"cancelled": True})
            jobs_processed += 1
            last_activity = time.time()
            continue

        # run_generation() below blocks this thread for the whole
        # generation — it can run for many minutes (see its own 2500s
        # request timeout) without this loop ever returning to the top,
        # where touch_session_heartbeat() normally runs once per
        # SESSION_POLL_INTERVAL_SECONDS-ish iteration. Without a separate
        # thread keeping the heartbeat alive here, any generation longer
        # than server.js's REAP_HEARTBEAT_STALE_MS (5 minutes) makes this
        # worker look dead to reapDeadWorkers(), which cancels the RunPod
        # job outright — killing a generation that was actively running,
        # not idle. A real user hit exactly this.
        heartbeat_stop = threading.Event()

        def keep_heartbeat_alive():
            while not heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                touch_session_heartbeat(session_id)

        heartbeat_thread = threading.Thread(target=keep_heartbeat_alive, daemon=True)
        heartbeat_thread.start()
        try:
            try:
                result = run_generation(
                    job_row["input"],
                    genkey,
                    on_progress=lambda data: write_job_progress(job_row["id"], data),
                    should_cancel=lambda: is_job_cancel_requested(job_row["id"]),
                    should_force_kill=lambda: is_job_force_cancel_requested(job_row["id"]),
                )
            except WorkerDeadError as e:
                # Unlike an ordinary generation failure (below, never takes
                # the session down with it), a dead worker means every OTHER
                # queued job in this session would hit the exact same
                # restart-and-fail cycle too — ending the session here, on
                # the very first failure, avoids silently burning the
                # restart wait again for each one.
                finish_job(job_row["id"], {"error": f"GPU worker is unavailable: {e}"})
                jobs_processed += 1
                kobold_restarts += 1
                print(f"Session {session_id}: worker appears dead ({e}) — ending session.")
                mark_session_ended(session_id, "error")
                terminate_worker_soon()
                return session_summary("worker_error", last_error=str(e))
            except Exception as e:
                # A single bad generation must never take the whole session
                # down with it — this is the "our own responsibility now"
                # error handling RunPod's per-job retries used to provide.
                result = {"error": str(e)}
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)

        if result.get("kobold_restarted"):
            kobold_restarts += 1
            print(
                f"Session {session_id}: job {job_row['id']} needed a "
                f"koboldcpp restart ({result.get('restart_seconds')}s) before running."
            )

        finish_job(job_row["id"], result)
        jobs_processed += 1

        if result.get("force_killed"):
            # Force Cancellation killed koboldcpp outright — the engine
            # is genuinely down now, not just between jobs. Restart and
            # re-warm it proactively here, right away, rather than
            # waiting for the next job to discover it's dead via
            # run_generation's own is_kobold_ready() check: that path
            # exists for a real crash and doesn't reset worker_started_at
            # first, so the frontend would keep claiming "ready" for a
            # worker that's actually reloading from scratch.
            print(f"Session {session_id}: koboldcpp was force-killed — restarting and re-warming.")
            reset_worker_started(session_id)
            try:
                start_kobold_if_needed()
                warmup_seconds = warmup_kobold(session_id)
                mark_session_worker_started(session_id)
                kobold_restarts += 1
            except Exception as e:
                print(f"Session {session_id}: failed to restart after force-kill ({e}) — ending session.")
                mark_session_ended(session_id, "error")
                terminate_worker_soon()
                return session_summary("worker_error", last_error=str(e))

        # Stamped now, at the moment the worker actually goes back to
        # idle, not when this job was claimed — a single generation can
        # easily run longer than SESSION_IDLE_TIMEOUT_SECONDS itself (see
        # run_generation's own 2500s request timeout), and the idle-timeout
        # check above only ever fires while job_row is None. Stamping this
        # at claim time meant a long-running generation was already
        # "overdue" by the idle clock the instant it finished, ending the
        # session on the very next loop iteration despite the worker never
        # having sat idle for even a second — a real user hit exactly this,
        # losing a completed generation to an immediate false timeout.
        last_activity = time.time()
        # Same moment covers last_gpu_touch too, whether or not a
        # force-kill restart/re-warm happened above -- either way the GPU
        # was just genuinely used, so the keep-warm branch's clock should
        # start fresh from here rather than firing an immediate, pointless
        # ping right after real work.
        last_gpu_touch = time.time()


def handler(job):
    job_input = job["input"]

    # Session mode: this one RunPod job IS the held-open worker for the
    # named session. It stays inside run_session() — never returning,
    # never marked COMPLETED by RunPod — until the session itself ends,
    # at which point returning here is what finally releases the worker
    # back to RunPod's available pool.
    session_id = job_input.get("session_id")
    if session_id:
        return run_session(session_id)

    if job_input.get("get_schema"):
        try:
            r = requests.get(f"{KOBOLD_URL}/api", timeout=15)
            r.raise_for_status()
            html = r.text
            marker = "let spec = "
            start = html.find(marker)
            if start == -1:
                return {"error": "Could not find spec in /api page", "preview": html[:500]}
            start += len(marker)
            end = html.find("</script>", start)
            raw_json = html[start:end].strip()
            return {"raw_spec": raw_json}
        except Exception as e:
            return {"error": f"Could not extract schema: {str(e)}"}

    genkey = job_input.get("genkey") or f"job_{job.get('id', 'unknown')}"

    def classic_progress(data):
        runpod.serverless.progress_update(job, data)

    try:
        return run_generation(job_input, genkey, on_progress=classic_progress)
    except WorkerDeadError as e:
        # This one job still gets an honest error back — but the worker
        # itself terminates right after, so RunPod can't hand the NEXT
        # classic click to this same possibly-broken container. See
        # terminate_worker_soon's own docstring for why restarting the
        # process in place isn't trusted to have actually fixed anything.
        terminate_worker_soon()
        return {"error": f"GPU worker is unavailable: {e}"}


runpod.serverless.start({"handler": handler})
