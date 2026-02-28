import os
import uuid
import subprocess
import threading
import time
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template, abort

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER  = Path("uploads")
OUTPUT_FOLDER  = Path("outputs")
MAX_FILE_MB    = 2048
FILE_TTL       = 3600
FFMPEG_PATH    = "ffmpeg"

ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
ALLOWED_PDF   = {".pdf"}

RESOLUTIONS = {
    "144p":  {"width": 256,  "height": 144,  "bitrate": "150k",   "audio": "64k"},
    "240p":  {"width": 426,  "height": 240,  "bitrate": "350k",   "audio": "96k"},
    "360p":  {"width": 640,  "height": 360,  "bitrate": "700k",   "audio": "128k"},
    "480p":  {"width": 854,  "height": 480,  "bitrate": "1000k",  "audio": "128k"},
    "720p":  {"width": 1280, "height": 720,  "bitrate": "2500k",  "audio": "192k"},
    "1080p": {"width": 1920, "height": 1080, "bitrate": "5000k",  "audio": "192k"},
    "1440p": {"width": 2560, "height": 1440, "bitrate": "10000k", "audio": "256k"},
}

IMAGE_RESOLUTIONS = {
    "144p":  (256,  144),
    "240p":  (426,  240),
    "360p":  (640,  360),
    "480p":  (854,  480),
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "original": None,
}

PDF_QUALITY = {
    "low":    {"compress_streams": True,  "recompress_flate": True,  "quality": 40},
    "medium": {"compress_streams": True,  "recompress_flate": True,  "quality": 65},
    "high":   {"compress_streams": True,  "recompress_flate": False, "quality": 85},
}

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_filename(original: str) -> str:
    name   = Path(original).name
    stem   = "".join(c for c in Path(name).stem if c.isalnum() or c in "-_")[:40]
    suffix = Path(name).suffix.lower()
    return f"{stem}{suffix}" if stem else f"upload{suffix}"

def delete_file_later(path: Path, delay: int = FILE_TTL):
    def _delete():
        time.sleep(delay)
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Could not delete %s: %s", path, e)
    threading.Thread(target=_delete, daemon=True).start()

def get_video_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def set_job(job_id, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)

# ── Video conversion ──────────────────────────────────────────────────────────
def run_video_conversion(job_id, input_path, output_path, res):
    cfg      = RESOLUTIONS[res]
    duration = get_video_duration(input_path)

    scale_filter = (
        f"scale={cfg['width']}:{cfg['height']}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={cfg['width']}:{cfg['height']}:(ow-iw)/2:(oh-ih)/2"
    )

    cmd = [
        FFMPEG_PATH, "-y", "-i", str(input_path),
        "-vf", scale_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-b:v", cfg["bitrate"], "-maxrate", cfg["bitrate"],
        "-bufsize", str(int(cfg["bitrate"][:-1]) * 2) + "k",
        "-c:a", "aac", "-b:a", cfg["audio"],
        "-movflags", "+faststart",
        "-progress", "pipe:2", "-nostats",
        str(output_path)
    ]

    set_job(job_id, status="processing")
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        for line in proc.stderr:
            line = line.strip()
            if line.startswith("out_time_ms=") and duration > 0:
                try:
                    ms  = int(line.split("=")[1])
                    pct = min(int((ms / 1_000_000 / duration) * 100), 99)
                    set_job(job_id, progress=pct)
                except ValueError:
                    pass
        proc.wait()
        if proc.returncode == 0 and output_path.exists():
            set_job(job_id, status="done", progress=100, output_path=str(output_path))
            delete_file_later(input_path)
            delete_file_later(output_path)
        else:
            raise RuntimeError(f"FFmpeg exited with code {proc.returncode}")
    except Exception as e:
        logger.error("Video job %s failed: %s", job_id, e)
        set_job(job_id, status="error", error=str(e))
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

# ── Image conversion ──────────────────────────────────────────────────────────
def run_image_conversion(job_id, input_path, output_path, res, fmt, quality):
    set_job(job_id, status="processing", progress=10)
    try:
        from PIL import Image, ImageOps
        img = Image.open(input_path)
        img = ImageOps.exif_transpose(img)

        # Convert RGBA/P to RGB if saving as JPEG
        if fmt.lower() in ("jpg", "jpeg") and img.mode in ("RGBA", "P", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = bg

        set_job(job_id, progress=30)

        target = IMAGE_RESOLUTIONS.get(res)
        if target:
            img.thumbnail(target, Image.LANCZOS)

        set_job(job_id, progress=70)

        save_kwargs = {"optimize": True}
        pil_fmt = fmt.upper()
        if pil_fmt == "JPG":
            pil_fmt = "JPEG"
        if pil_fmt in ("JPEG", "WEBP"):
            save_kwargs["quality"] = int(quality)

        img.save(str(output_path), format=pil_fmt, **save_kwargs)
        set_job(job_id, status="done", progress=100, output_path=str(output_path))
        delete_file_later(input_path)
        delete_file_later(output_path)

    except Exception as e:
        logger.error("Image job %s failed: %s", job_id, e)
        set_job(job_id, status="error", error=str(e))
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

# ── PDF compression ───────────────────────────────────────────────────────────
def run_pdf_compression(job_id, input_path, output_path, quality_level):
    set_job(job_id, status="processing", progress=15)
    try:
        import pikepdf
        from pikepdf import Pdf, PdfImage
        from PIL import Image
        import io

        cfg = PDF_QUALITY[quality_level]
        set_job(job_id, progress=30)

        pdf = pikepdf.open(str(input_path))
        set_job(job_id, progress=50)

        # Compress images inside PDF
        q = cfg["quality"]
        for page_num, page in enumerate(pdf.pages):
            if "/Resources" not in page:
                continue
            resources = page["/Resources"]
            if "/XObject" not in resources:
                continue
            xobjects = resources["/XObject"]
            for key in list(xobjects.keys()):
                try:
                    xobj = xobjects[key]
                    if xobj.get("/Subtype") == "/Image":
                        pdfimage = PdfImage(xobj)
                        pil_img = pdfimage.as_pil_image()
                        if pil_img.mode in ("RGBA", "P"):
                            pil_img = pil_img.convert("RGB")
                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=q, optimize=True)
                        buf.seek(0)
                        new_img = pikepdf.open(buf)
                        xobjects[key] = pdf.copy_foreign(new_img.pages[0]["/Resources"]["/XObject"]["/Im0"])
                except Exception:
                    pass  # skip images that can't be recompressed

        set_job(job_id, progress=80)

        pdf.save(
            str(output_path),
            compress_streams=cfg["compress_streams"],
            recompress_flate=cfg["recompress_flate"],
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )
        pdf.close()

        set_job(job_id, status="done", progress=100, output_path=str(output_path))
        delete_file_later(input_path)
        delete_file_later(output_path)

    except Exception as e:
        logger.error("PDF job %s failed: %s", job_id, e)
        set_job(job_id, status="error", error=str(e))
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload/video", methods=["POST"])
def upload_video():
    f   = request.files.get("file")
    res = request.form.get("resolution", "720p")

    if not f or not f.filename:
        return jsonify({"error": "No file"}), 400
    if Path(f.filename).suffix.lower() not in ALLOWED_VIDEO:
        return jsonify({"error": "Unsupported video format"}), 400
    if res not in RESOLUTIONS:
        return jsonify({"error": "Invalid resolution"}), 400

    job_id   = uuid.uuid4().hex
    safe_fn  = safe_filename(f.filename)
    in_path  = UPLOAD_FOLDER / f"{job_id}_{safe_fn}"
    out_path = OUTPUT_FOLDER / f"{job_id}_{Path(safe_fn).stem}_{res}.mp4"

    f.save(in_path)
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "output_path": None, "error": None}

    threading.Thread(target=run_video_conversion,
                     args=(job_id, in_path, out_path, res), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.route("/upload/image", methods=["POST"])
def upload_image():
    f       = request.files.get("file")
    res     = request.form.get("resolution", "original")
    fmt     = request.form.get("format", "jpg").lower()
    quality = request.form.get("quality", "85")

    if not f or not f.filename:
        return jsonify({"error": "No file"}), 400
    if Path(f.filename).suffix.lower() not in ALLOWED_IMAGE:
        return jsonify({"error": "Unsupported image format"}), 400

    job_id   = uuid.uuid4().hex
    safe_fn  = safe_filename(f.filename)
    in_path  = UPLOAD_FOLDER / f"{job_id}_{safe_fn}"
    ext_map  = {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png", "webp": ".webp"}
    out_ext  = ext_map.get(fmt, ".jpg")
    out_path = OUTPUT_FOLDER / f"{job_id}_{Path(safe_fn).stem}_{res}{out_ext}"

    f.save(in_path)
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "output_path": None, "error": None}

    threading.Thread(target=run_image_conversion,
                     args=(job_id, in_path, out_path, res, fmt, quality), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.route("/upload/pdf", methods=["POST"])
def upload_pdf():
    f       = request.files.get("file")
    quality = request.form.get("quality", "medium")

    if not f or not f.filename:
        return jsonify({"error": "No file"}), 400
    if Path(f.filename).suffix.lower() not in ALLOWED_PDF:
        return jsonify({"error": "Not a PDF"}), 400
    if quality not in PDF_QUALITY:
        return jsonify({"error": "Invalid quality"}), 400

    job_id   = uuid.uuid4().hex
    safe_fn  = safe_filename(f.filename)
    in_path  = UPLOAD_FOLDER / f"{job_id}_{safe_fn}"
    out_path = OUTPUT_FOLDER / f"{job_id}_{Path(safe_fn).stem}_compressed.pdf"

    f.save(in_path)
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "output_path": None, "error": None}

    threading.Thread(target=run_pdf_compression,
                     args=(job_id, in_path, out_path, quality), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.route("/status/<job_id>")
def status(job_id):
    if not job_id.isalnum() or len(job_id) != 32:
        abort(400)
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"status": job["status"], "progress": job["progress"], "error": job["error"]})


@app.route("/download/<job_id>")
def download(job_id):
    if not job_id.isalnum() or len(job_id) != 32:
        abort(400)
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    out_path = Path(job["output_path"])
    try:
        out_path.resolve().relative_to(OUTPUT_FOLDER.resolve())
    except ValueError:
        abort(403)
    if not out_path.exists():
        abort(410)
    return send_file(out_path, as_attachment=True,
                     download_name=out_path.name.split("_", 1)[-1])


# ── Startup cleanup ───────────────────────────────────────────────────────────
def cleanup_old_files():
    cutoff = time.time() - FILE_TTL
    for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER):
        for f in folder.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)

cleanup_old_files()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)