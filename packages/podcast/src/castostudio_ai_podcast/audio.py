import threading
import time
import logging
import numpy as np
import av

LOGGER = logging.getLogger(__name__)

class AudioVolumeReader:
    def __init__(self, url: str, label: str):
        self.url = url
        self.label = label
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_rms = 0.0
        self.latest_db = -100.0
        self.failures = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            name=f"AudioReader-{self.label}",
            daemon=True
        )
        self.thread.start()
        LOGGER.info("Audio reader thread started for %s (%s)", self.label, self.url)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        LOGGER.info("Audio reader thread stopped for %s", self.label)

    def get_volume(self) -> float:
        """Return the latest RMS volume level."""
        with self.lock:
            return self.latest_rms

    def get_volume_db(self) -> float:
        """Return the latest volume level in dB (-100 to 0)."""
        with self.lock:
            return self.latest_db

    def _run(self):
        while self.running:
            container = None
            try:
                # Open the RTMP/SRT stream. We use options to set timeout to avoid blocking indefinitely.
                options = {
                    "rtsp_transport": "tcp",
                    "stimeout": "5000000",  # 5 seconds
                    "rw_timeout": "5000000",
                }
                container = av.open(self.url, options=options)
                audio_streams = [s for s in container.streams if s.type == "audio"]
                
                if not audio_streams:
                    LOGGER.warning("No audio stream found for %s (%s)", self.label, self.url)
                    time.sleep(2)
                    continue

                audio_stream = audio_streams[0]
                self.failures = 0

                # Decode loop
                for frame in container.decode(audio_stream):
                    if not self.running:
                        break
                    
                    # Convert audio samples to numpy array
                    samples = frame.to_ndarray()
                    
                    if samples.size > 0:
                        # Normalize samples if they are integer type
                        if np.issubdtype(samples.dtype, np.integer):
                            info = np.iinfo(samples.dtype)
                            samples = samples.astype(np.float32) / info.max
                        
                        rms = np.sqrt(np.mean(samples**2))
                        
                        # Convert to dB
                        if rms > 1e-5:
                            db = 20 * np.log10(rms)
                        else:
                            db = -100.0

                        with self.lock:
                            self.latest_rms = float(rms)
                            self.latest_db = float(db)

                if container:
                    container.close()

            except Exception as exc:
                self.failures += 1
                LOGGER.warning(
                    "Error reading audio for %s (fail count: %d): %s",
                    self.label,
                    self.failures,
                    exc,
                )
                if container:
                    try:
                        container.close()
                    except Exception:
                        pass
                # Backoff before reconnecting
                time.sleep(min(2 ** self.failures, 10))

            time.sleep(0.1)
