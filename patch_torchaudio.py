"""Patch fish-speech reference_loader.py to fix torchaudio UnboundLocalError.

The except block in __init__ does `import torchaudio.io._load_audio_fileobj`
which triggers a circular import, leaving `torchaudio` unbound. This replaces
the entire try/except block with a simple fallback.
"""
import pathlib
import sys

target = pathlib.Path("/app/fish-speech/fish_speech/inference_engine/reference_loader.py")
src = target.read_text()

old = """        try:
            backends = torchaudio.list_audio_backends()
            if "ffmpeg" in backends:
                self.backend = "ffmpeg"
            else:
                self.backend = "soundfile"
        except AttributeError:
            # torchaudio 2.9+ removed list_audio_backends()
            # Try ffmpeg first, fallback to soundfile
            try:
                import torchaudio.io._load_audio_fileobj  # noqa: F401

                self.backend = "ffmpeg"
            except (ImportError, ModuleNotFoundError):
                self.backend = "soundfile\""""

new = """        try:
            backends = torchaudio.list_audio_backends()
            self.backend = "ffmpeg" if "ffmpeg" in backends else "soundfile"
        except (AttributeError, UnboundLocalError):
            self.backend = "soundfile\""""

if old not in src:
    print(f"ERROR: Patch target not found in {target}")
    print("The file may have changed upstream. Check the current content.")
    sys.exit(1)

target.write_text(src.replace(old, new))
print(f"Patched {target} successfully")
