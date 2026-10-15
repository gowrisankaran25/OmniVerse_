// Webcam + MediaPipe HandLandmarker (Tasks API, VIDEO mode). The render loop calls poll() once per frame;
// detection only runs when the camera delivered a new frame, and timestamps strictly increase.
const LOCAL_VISION = new URL('../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs', import.meta.url).href;
const LOCAL_WASM = new URL('../node_modules/@mediapipe/tasks-vision/wasm', import.meta.url).href;
const LOCAL_MODEL = new URL('../models/hand_landmarker.task', import.meta.url).href;
const CDN_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const CDN_VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';
const CDN_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';

async function loadVisionModule() {
  try {
    const head = await fetch(LOCAL_VISION, { method: 'HEAD' });
    if (head.ok) return await import(LOCAL_VISION);
  } catch {}
  return await import(CDN_VISION);
}

async function resolveFileset(FilesetResolver) {
  try {
    const head = await fetch(`${LOCAL_WASM}/vision_wasm_internal.js`, { method: 'HEAD' });
    if (head.ok) return await FilesetResolver.forVisionTasks(LOCAL_WASM);
  } catch {}
  return await FilesetResolver.forVisionTasks(CDN_WASM);
}

async function resolveModelPath() {
  try {
    const head = await fetch(LOCAL_MODEL, { method: 'HEAD' });
    if (head.ok) return LOCAL_MODEL;
  } catch {}
  return CDN_MODEL;
}

export class HandCamera {
  constructor(video) {
    this.video = video;
    this.landmarker = null;
    this.stream = null;
    this.running = false;
    this.delegate = null;
    this.lastVideoTime = -1;
    this.lastTs = 0;
    this.frames = 0;             // camera frames processed
    this.fps = 0;
    this.detectMs = 0;
    this._n = 0; this._t = performance.now();
    this.newFrame = false;       // a new video frame arrived since the last poll (renderer uploads it)
  }

  async start(onStatus = () => {}) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser cannot access a camera. Please ensure you are using HTTPS (https://gowrisankaran25.github.io/OmniVerse_/) and a supported browser.');
    }
    onStatus('Requesting camera permission…');
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
    } catch (err) {
      console.warn('Constrained camera request failed, falling back to basic video', err);
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    this.stream = stream;
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.setAttribute('playsinline', '');
    this.video.setAttribute('muted', '');

    try {
      await this.video.play();
    } catch (e) {
      console.warn('video.play() warning:', e);
    }

    if (this.video.readyState < 2) {
      await new Promise((resolve) => {
        const done = () => {
          this.video.removeEventListener('loadeddata', done);
          resolve();
        };
        this.video.addEventListener('loadeddata', done, { once: true });
        setTimeout(resolve, 1500);
      });
    }

    // Camera hardware stream is active and running
    this.running = true;
    onStatus('Camera active! Loading hand tracker…');

    // Load AI hand tracker asynchronously with graceful fallback
    try {
      const { FilesetResolver, HandLandmarker } = await loadVisionModule();
      const fileset = await resolveFileset(FilesetResolver);
      const modelAssetPath = await resolveModelPath();

      const make = (delegate, assetPath, fs = fileset) => HandLandmarker.createFromOptions(fs, {
        baseOptions: { modelAssetPath: assetPath, delegate },
        runningMode: 'VIDEO', numHands: 2,
        minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
      });

      try {
        this.landmarker = await make('GPU', modelAssetPath);
        this.delegate = 'GPU';
      } catch (eGPU) {
        console.warn('GPU delegate failed, falling back to CPU', eGPU);
        try {
          this.landmarker = await make('CPU', modelAssetPath);
          this.delegate = 'CPU';
        } catch (eCPU) {
          console.warn('CPU delegate failed with initial fileset/model, trying CDN wasm + CDN model', eCPU);
          const cdnFileset = await FilesetResolver.forVisionTasks(CDN_WASM);
          this.landmarker = await make('CPU', CDN_MODEL, cdnFileset);
          this.delegate = 'CPU';
        }
      }
      onStatus('📷 Camera active — hand tracking ready!');
    } catch (aiErr) {
      console.warn('AI hand landmark tracking unavailable, camera video active in background:', aiErr);
      onStatus('📷 Camera active — touch/mouse controls active');
    }
    setTimeout(() => onStatus(''), 2500);
  }

  /** Returns undefined when there is no new camera frame, else a list of hands (mirrored, selfie view), possibly empty. */
  poll() {
    const v = this.video;
    if (!this.running || v.readyState < 2 || v.currentTime === this.lastVideoTime) return undefined;
    this.lastVideoTime = v.currentTime;
    this.newFrame = true;
    if (!this.landmarker) return [];   // camera stream active, waiting or fallback mode
    const ts = Math.max(Math.round(performance.now()), this.lastTs + 1);      // must strictly increase
    this.lastTs = ts;
    const t0 = performance.now();
    try {
      const res = this.landmarker.detectForVideo(v, ts);
      this.detectMs = performance.now() - t0;
      this.frames++; this._n++;
      const now = performance.now();
      if (now - this._t > 1000) { this.fps = (this._n * 1000) / (now - this._t); this._n = 0; this._t = now; }
      return (res.landmarks || []).map((lm) => lm.map((p) => [1 - p.x, p.y]));   // mirror x: selfie view
    } catch (detectErr) {
      console.warn('MediaPipe detect warning:', detectErr);
      return [];
    }
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.landmarker?.close?.();
    this.landmarker = null;
  }
}
