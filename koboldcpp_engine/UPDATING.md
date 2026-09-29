# Updating this koboldcpp fork

This directory is a patched copy of koboldcpp's own public source
(LostRuins/koboldcpp, `concedo` branch), pinned at the version noted at the
top of `koboldcpp.py`. It does **not** auto-update — koboldcpp ships new
releases regularly, and none of those land here until someone manually
repeats the steps below.

## What's actually patched

Exactly one thing: a new route, `/api/extra/abort_image`, added right after
the existing `/api/extra/abort` route in `koboldcpp.py`. It calls
`handle.sd_abort_generation()` — a native function koboldcpp already ships,
previously only triggered automatically on client disconnect — so a
generation can be cancelled on demand instead. Search this file for
`abort_image` to see the exact block.

Nothing in `koboldcpp_cublas.so` itself was patched — this route is pure
Python. The compiled CUDA backend is built from source by the worker on
first cold start (see `handler.py`'s `ensure_koboldcpp_engine()` and
`build_convrot_cuda_engine.sh`), from the exact upstream commit pinned in
that script, not from this directory's files at all.

## Why this needs manual updates

There was no way to slip this one-route patch into a from-source build
without it living somewhere — so the worker runs this patched
`koboldcpp.py` directly (`python3 koboldcpp.py`) next to the separately
built `koboldcpp_cublas.so`. That means the Python layer here is a frozen
snapshot — new features/fixes koboldcpp ships in `koboldcpp.py` itself
won't reach this worker until this snapshot is refreshed by hand. The
compiled engine (`koboldcpp_cublas.so`) is unaffected by any of this; it
tracks whatever commit `build_convrot_cuda_engine.sh` is pinned to,
independent of this file.

## How to update (should take a few minutes)

1. Download the latest `koboldcpp.py` from
   https://github.com/LostRuins/koboldcpp/blob/concedo/koboldcpp.py
2. Search the new file for `/api/extra/abort` (the *existing*, unpatched
   text-abort route) — it's the anchor point.
3. Re-insert this block immediately after it (copy verbatim from the
   current patched `koboldcpp.py` here — search for `/api/extra/abort_image`
   to find it):

   ```python
   elif clean_path.endswith('/api/extra/abort_image'):
       if not self.secure_endpoint():
           return
       multiuserkey = ""
       try:
           tempbody = json.loads(body)
           if isinstance(tempbody, dict):
               multiuserkey = tempbody.get('genkey', "")
       except Exception:
           multiuserkey = ""
       if multiuserkey != "" and multiuserkey != currgenimgkey:
           response_code = 403
           response_body = (json.dumps({"success": "false", "done":"false"}).encode())
       else:
           try:
               handle.sd_abort_generation()
               time.sleep(0.1)
               response_body = (json.dumps({"success": "true", "done":"true"}).encode())
               print("\nImage/Video Generation Aborted")
           except Exception as e:
               utfprint("Abort Image Error: " + str(e))
               response_code = 400
               response_body = (json.dumps({"success": "false", "done":"false"}).encode())
   ```

   The `genkey` check (added after the original patch) scopes an abort to
   the generation that actually owns that key — the same pattern
   `/api/extra/abort` already uses via `currentusergenkey` — instead of
   blindly killing whatever image/video job happens to be running. An
   empty/missing `genkey` still aborts whatever's in flight (matches the
   original behavior and every current caller in this repo).

4. Update the version/commit noted at the top of `koboldcpp.py`.
5. Also refresh `json_to_gbnf.py` and `kcpp_adapters/` from the same
   upstream release while you're at it — cheap, and keeps things consistent.
6. Commit, push, deploy. The next fresh RunPod volume (or one where you
   manually delete `/runpod-volume/koboldcpp_engine/koboldcpp.py`) will pick
   up the new file automatically — `ensure_koboldcpp_engine()` always
   overwrites the `.py` files on the volume from what the worker image
   ships, every single start, on purpose (it's cheap, unlike the engine
   `.so`, which stays cached once downloaded).

If koboldcpp ever restructures this part of the file enough that the patch
doesn't apply cleanly, that's a sign to sanity-check nothing else nearby
changed shape, not just paste-and-hope.
