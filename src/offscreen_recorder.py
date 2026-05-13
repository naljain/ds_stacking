"""Small Isaac Sim viewport GIF recorder.

This avoids Replicator's orchestrator path, which can fail in standalone
headless launches on some Isaac Sim installs. Instead it schedules lightweight
viewport buffer captures and assembles the completed buffers into a GIF at
shutdown.
"""

from pathlib import Path

import numpy as np


class OffscreenGifRecorder:
    def __init__(self, camera_prim_path="/World/Camera", out_path=None,
                 size=(640, 480), fps=20, stride=4):
        self.out_path = Path(out_path) if out_path else None
        self.fps = int(fps)
        self.stride = max(1, int(stride))
        self.frames = []
        self._step = 0
        self._capture_attempts = 0
        self._pending = 0
        self._last_problem = None
        self._warned = False
        self._viewport = None
        self.enabled = bool(self.out_path)

        if not self.enabled:
            return

        try:
            from pxr import Sdf
            from omni.kit.viewport.utility import (
                create_viewport_window,
                get_active_viewport,
            )

            self._viewport = get_active_viewport()
            if self._viewport is None:
                self._viewport = create_viewport_window(
                    name="OffscreenCapture",
                    width=int(size[0]),
                    height=int(size[1]),
                    camera_path=Sdf.Path(camera_prim_path),
                )
            if self._viewport is None:
                raise RuntimeError("no active viewport available")
            try:
                self._viewport.camera_path = Sdf.Path(camera_prim_path)
            except Exception:
                pass
            try:
                self._viewport.resolution = (int(size[0]), int(size[1]))
            except Exception:
                pass
            print(f"[REC] Viewport GIF capture enabled -> {self.out_path}")
        except Exception as exc:
            self.enabled = False
            self._last_problem = f"viewport recorder init failed: {exc}"
            print(f"[WARN] Could not initialise viewport GIF recorder: {exc}")

    def _on_capture(self, buffer, buffer_size, width, height, byte_format):
        self._pending = max(0, self._pending - 1)
        try:
            import omni.kit.renderer_capture

            rgba = omni.kit.renderer_capture.convert_raw_bytes_to_rgba_tuples(
                buffer, buffer_size, width, height, byte_format
            )
            frame = np.asarray(rgba, dtype=np.uint8).reshape(
                (int(height), int(width), 4)
            )[..., :3]
            self.frames.append(frame.copy())
        except Exception as exc:
            self._last_problem = f"viewport buffer conversion failed: {exc}"
            if not self._warned:
                print(f"[WARN] {self._last_problem}")
                self._warned = True

    def capture(self):
        if not self.enabled:
            return
        self._step += 1
        if self._step % self.stride != 0:
            return
        try:
            from omni.kit.viewport.utility import capture_viewport_to_buffer

            self._capture_attempts += 1
            self._pending += 1
            capture_viewport_to_buffer(self._viewport, self._on_capture)
        except Exception as exc:
            self._pending = max(0, self._pending - 1)
            self._last_problem = f"viewport capture scheduling failed: {exc}"
            if not self._warned:
                print(f"[WARN] {self._last_problem}")
                self._warned = True

    def close(self):
        if not self.out_path:
            return
        # Give async captures a few updates to flush their callbacks.
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
            for _ in range(8):
                if self._pending <= 0:
                    break
                app.update()
        except Exception:
            pass
        if not self.frames:
            detail = (
                f" after {self._capture_attempts} capture attempts"
                if self._capture_attempts else " because capture() was never called"
            )
            if self._last_problem:
                detail += f" ({self._last_problem})"
            print(f"[WARN] No offscreen frames captured for {self.out_path}{detail}")
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(self.out_path, self.frames, fps=self.fps)
        except Exception:
            from PIL import Image
            duration_ms = int(round(1000 / max(self.fps, 1)))
            images = [Image.fromarray(f) for f in self.frames]
            images[0].save(
                self.out_path,
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
            )
        print(f"[REC] Wrote {len(self.frames)} frames -> {self.out_path}")
