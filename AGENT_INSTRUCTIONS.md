# CLAWRION — Agent Instructions

Rules for any agent working on this repo. Read before making changes.

---

## 1. Never pin torch/torchaudio/torchvision separately

**Don't do this:**
```python
.pip_install("torch>=2.6.0", "torchaudio>=2.6.0")
.run_commands("pip install -e '.[server]'")
```

**Do this:**
```python
.run_commands("pip install -e '.[server]'")
```

Fish-speech's `[server]` extras declare exact compatible versions of torch, torchaudio, and all other dependencies. Pre-installing torch separately creates version conflicts because pip tries to satisfy two competing constraints.

This was the root cause of 6+ failed deploys on 2026-03-29. sglang required torch 2.9.1, fish-speech required torch 2.8.0. Removing sglang and letting fish-speech manage everything in one `pip install` resolved it.

**Exception:** If you need a package that's NOT part of fish-speech's dependency tree (like faster-whisper), install it BEFORE the fish-speech install so pip can satisfy both.

---

## 2. Don't use sed for code patches — use Python with verification

**Don't do this:**
```python
.run_commands("sed -i 's/old/new/' /app/fish-speech/some_file.py")
```

`sed -i` exits with code 0 whether or not the pattern was found. If the upstream file changes, the sed silently does nothing and you only find out at runtime.

**Do this:**
```python
.run_commands("""python3 -c "
import pathlib
p = pathlib.Path('/app/fish-speech/some_file.py')
src = p.read_text()
old = '''the exact old text'''
new = '''the replacement text'''
assert old in src, f'Patch target not found in {p}'
p.write_text(src.replace(old, new))
print('Patched successfully')
"
""")
```

Python gives you:
- `assert` to fail the build if the pattern isn't found
- Exact multi-line matching (sed chokes on special characters)
- Clear error messages

---

## 3. Pin dependencies to specific versions

Always pin git clones and pip installs to known-good versions:

```python
# Git: use --branch or checkout a commit
.run_commands("git clone --depth 1 --branch v2.0.0 https://github.com/fishaudio/fish-speech.git /app/fish-speech")

# Pip: use ==
.uv_pip_install("faster-whisper==1.1.1")
```

Unpinned deps break silently when upstream changes. You won't know until the next deploy fails or inference produces wrong results.

---

## 4. Image layer order matters

Modal caches image layers. Put stable, rarely-changing layers first:

```
1. Base image (never changes)
2. apt_install (rarely changes)
3. pip_install stable deps (rarely changes)
4. git clone pinned version (changes only when you bump version)
5. pip install fish-speech (changes when deps change)
6. Model weight download (changes only when model changes)
7. pip_install fastapi/uvicorn (rarely changes)
```

Code changes (server.py) are mounted separately and don't trigger image rebuilds. Only change layers 3-6 when you intentionally want to update dependencies.

---

## 5. The `min_containers` cost trap

`min_containers=1` means the GPU runs 24/7 at ~$0.80/hr = ~$19/day even when nobody is using it.

Use `min_containers=0` (scale to zero) unless you specifically need zero cold starts. The dashboard should poll the lightweight health endpoint (`MODAL_HEALTH_URL`), not the GPU endpoint, to avoid keeping the GPU alive.
