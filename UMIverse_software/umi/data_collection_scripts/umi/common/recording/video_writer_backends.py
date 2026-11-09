# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video encoding backends for recording.

Supports:
- opencv_mp4v: OpenCV VideoWriter with MPEG-4 Part 2 (fast, good compatibility).
- ffmpeg_h264: FFmpeg libx264 via subprocess (smaller files, uses a writer thread).
"""

import logging
import queue
import subprocess
import threading
from abc import ABC, abstractmethod
from typing import Any, Literal

import cv2
import numpy as np

log = logging.getLogger(__name__)

VideoEncoderKind = Literal["opencv_mp4v", "ffmpeg_h264"]

# Cap captured ffmpeg stderr so a chatty process cannot exhaust memory.
_MAX_FFMPEG_STDERR_BYTES = 256 * 1024


class VideoEncodingFailed(RuntimeError):
    """Raised when encoded video cannot be completed or ffmpeg exits with an error."""


class VideoWriterBackend(ABC):
    """Interface for writing video frames to a file."""

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        """Write one BGR frame (H, W, 3) uint8."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Finish and close the file. Idempotent."""
        pass


class OpenCVBackend(VideoWriterBackend):
    """Write video using OpenCV VideoWriter (MPEG-4 Part 2 / mp4v)."""

    def __init__(self, path: str, width: int, height: int, fps: float) -> None:
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open: {path}")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class FFmpegH264Backend(VideoWriterBackend):
    """Write video using FFmpeg libx264. Uses a queue and writer thread so the
    recording thread does not block on the pipe.
    """

    def __init__(self, path: str, width: int, height: int, fps: float) -> None:
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._queue: queue.Queue[np.ndarray | None] | None = queue.Queue(maxsize=60)
        self._proc: subprocess.Popen[Any] | None = None
        self._thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_buf = bytearray()
        self._stderr_lock = threading.Lock()
        self._writer_error: BaseException | None = None
        self._released = False
        self._start()

    def _start(self) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "medium",
            self._path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        stderr_r = self._proc.stderr
        if stderr_r is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(stderr_r,),
                daemon=True,
                name="ffmpeg-h264-stderr",
            )
            self._stderr_thread.start()
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=False, name="ffmpeg-h264-writer"
        )
        self._thread.start()

    def _drain_stderr(self, stderr_r: Any) -> None:
        try:
            while True:
                chunk = stderr_r.read(8192)
                if not chunk:
                    break
                with self._stderr_lock:
                    room = _MAX_FFMPEG_STDERR_BYTES - len(self._stderr_buf)
                    if room > 0:
                        self._stderr_buf.extend(chunk[:room])
        except Exception:
            log.exception("ffmpeg stderr drain failed (path=%s)", self._path)

    def _failure_detail(self) -> str:
        with self._stderr_lock:
            tail = self._stderr_buf.decode("utf-8", errors="replace")
        lines = [
            f"path={self._path}",
            f"width={self._width} height={self._height} fps={self._fps}",
        ]
        if self._writer_error is not None:
            lines.append(f"writer_thread_error={self._writer_error!r}")
        if tail.strip():
            lines.append("--- ffmpeg stderr (tail, may be truncated) ---")
            lines.append(tail[-8000:] if len(tail) > 8000 else tail)
        return "\n".join(lines)

    def _writer_loop(self) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            while True:
                if self._queue is None:
                    break
                frame = self._queue.get()
                if frame is None:
                    break
                self._proc.stdin.write(frame.tobytes())
        except Exception as e:
            self._writer_error = e
            log.exception(
                "ffmpeg H.264 writer thread failed (path=%s). "
                "Video file may be incomplete or corrupt.",
                self._path,
            )
        finally:
            try:
                self._proc.stdin.close()
            except Exception as e:
                log.warning(
                    "closing ffmpeg stdin failed (path=%s): %s",
                    self._path,
                    e,
                    exc_info=True,
                )

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise VideoEncodingFailed(
                f"write() after release() is invalid (path={self._path})"
            )
        if self._writer_error is not None:
            err = VideoEncodingFailed(
                "Video encoding already failed; cannot write more frames.\n"
                + self._failure_detail()
            )
            raise err from self._writer_error
        if self._queue is None:
            raise VideoEncodingFailed(f"Video writer queue is gone (path={self._path})")
        try:
            self._queue.put(frame, block=True, timeout=5.0)
        except queue.Full:
            msg = (
                f"ffmpeg H.264 encode backlog: queue full after 5s (path={self._path}). "
                "Encoder cannot keep up; recording cannot continue safely."
            )
            log.error(msg)
            exc = VideoEncodingFailed(msg)
            self._writer_error = exc
            raise exc

    def _force_stop_ffmpeg(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                log.error("terminating stuck ffmpeg process (path=%s)", self._path)
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    log.error("killing ffmpeg process (path=%s)", self._path)
                    proc.kill()
                    proc.wait(timeout=5.0)
        except Exception:
            log.exception("failed to stop ffmpeg (path=%s)", self._path)

    def release(self) -> None:
        if self._released:
            return
        self._released = True

        q = self._queue
        if q is not None:
            try:
                q.put(None, block=True, timeout=5.0)
            except queue.Full:
                log.error(
                    "cannot enqueue ffmpeg shutdown sentinel — queue full (path=%s)",
                    self._path,
                )
                self._force_stop_ffmpeg()

        th = self._thread
        if th is not None:
            th.join(timeout=30.0)
            if th.is_alive():
                log.error(
                    "ffmpeg writer thread did not finish in 30s (path=%s)",
                    self._path,
                )
                self._force_stop_ffmpeg()
                th.join(timeout=10.0)
                if th.is_alive():
                    detail = self._failure_detail()
                    log.error("writer thread still alive after force stop:\n%s", detail)
                    raise VideoEncodingFailed(
                        "ffmpeg writer thread hung; video output is unreliable.\n"
                        + detail
                    )

        proc = self._proc
        if proc is not None:
            try:
                proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                log.error("ffmpeg did not exit after stdin close (path=%s)", self._path)
                self._force_stop_ffmpeg()
                detail = self._failure_detail()
                raise VideoEncodingFailed(
                    "ffmpeg did not exit in time; output may be corrupt.\n" + detail
                ) from None
            rc = proc.returncode
            if rc != 0:
                detail = self._failure_detail()
                log.error(
                    "ffmpeg exited with code %s (path=%s)\n%s",
                    rc,
                    self._path,
                    detail,
                )
                err = VideoEncodingFailed(
                    f"ffmpeg failed (returncode={rc}) for {self._path}.\n{detail}"
                )
                raise err from self._writer_error

        if self._writer_error is not None:
            detail = self._failure_detail()
            log.error(
                "video encoding failed before finalize (path=%s)\n%s",
                self._path,
                detail,
            )
            raise VideoEncodingFailed(
                f"Video encoding failed before finalize: {self._path}\n{detail}"
            ) from self._writer_error

        self._thread = None
        self._proc = None
        self._queue = None
        self._stderr_thread = None


def create_video_writer(
    encoder: VideoEncoderKind,
    path: str,
    width: int,
    height: int,
    fps: float,
) -> VideoWriterBackend:
    """Create a video writer backend for the given encoder and parameters.

    Args:
        encoder: "opencv_mp4v" or "ffmpeg_h264".
        path: Output .mp4 file path.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.

    Returns:
        A backend instance with write(frame) and release().
    """
    if encoder == "opencv_mp4v":
        return OpenCVBackend(path, width, height, fps)
    if encoder == "ffmpeg_h264":
        return FFmpegH264Backend(path, width, height, fps)
    raise ValueError(f"Unknown video encoder: {encoder}")
