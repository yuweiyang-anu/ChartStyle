# -*- coding: utf-8 -*-
"""
ChartForge — Stage IV, step 2 of 3: OCR F1 content-preservation filtering.

Complementary to the LLM judges, this stage runs PaddleOCR on the content image
and the target image of each pair and computes a token-level F1 between the two
recognized text sets. A high F1 confirms that the restyle preserved the figure's
textual content (labels, legends, annotations, cell values) — the property the
LLM content judge assesses semantically, verified here lexically.

----------------------------------------------------------------------
OCR-only evaluator (FAST, multi-GPU, image-cache, Manager Queue progress)

Enhancements:
- Start from an arbitrary JSONL line (--start-line) and optional forced sample_id numbering
  via (--force-id --id-prefix pair --id-start 12345) so outputs look like _ocr_pair_012345.json
- Optional path prefix rewriting while loading JSONL (--from-prefix/--to-prefix);
  disabled by default since the pipeline is fully local.

Core:
- Only PaddleOCR F1(content vs target)
- Per-GPU process; inside: simple prefetch queue -> main OCR loop
- Image-level OCR caching: each unique image path OCR once -> saved as a Counter JSON
- Angle classifier OFF by default (can enable with --use-angle-cls)
- Optional TensorRT + FP16; big rec batch; reduced det resolution with fallback retry
- Per-item JSON: <ratings_dir>/_ocr_<sample_id>.json
  {
    "sample_id": "...",
    "style_image": "...",
    "content_image": "...",
    "target_image": "...",
    "paddleocr_text_overlap_f1": <float>
  }
- Overall JSON: <ratings_dir>/_ocr_overall.json
"""

import os
import re
import io
import cv2
import json
import time
import argparse
import hashlib
import threading
from typing import Any, Optional, List, Dict, Tuple
from collections import Counter, deque
import multiprocessing as mp

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from paddleocr import PaddleOCR
    _paddleocr_available = True
except Exception:
    PaddleOCR = None
    _paddleocr_available = False


# ---------------- Utils ----------------
def _sanitize_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_") or "sample"

def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _alpha_over_checker_cv2(bgr: np.ndarray) -> np.ndarray:
    # Input: BGR or BGRA, output BGR
    if bgr is None:
        return bgr
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        # BGRA -> BGR with white background
        b, g, r, a = cv2.split(bgr)
        alpha = a.astype(np.float32) / 255.0
        bg = np.ones_like(b, dtype=np.float32) * 255.0
        def blend(ch, alpha_):
            return (ch.astype(np.float32) * alpha_ + bg * (1.0 - alpha_)).astype(np.uint8)
        b = blend(b, alpha)
        g = blend(g, alpha)
        r = blend(r, alpha)
        return cv2.merge([b, g, r])
    return bgr

def _read_bgr_fast(path: str) -> Optional[np.ndarray]:
    # Prefer OpenCV (faster), fallback to PIL
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cv2.imdecode failed")
        return _alpha_over_checker_cv2(img)
    except Exception:
        try:
            im = Image.open(path)
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                bg.paste(im, (0, 0), im)
                im = bg.convert("RGB")
            elif im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

def _cache_key_for_path(p: str) -> str:
    try:
        st = os.stat(p)
        key = f"{os.path.abspath(p)}::{st.st_size}::{int(st.st_mtime)}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha1(os.path.abspath(p).encode("utf-8")).hexdigest()

def _rewrite_path(p: Optional[str], from_prefix: Optional[str], to_prefix: Optional[str]) -> Optional[str]:
    if not p or not from_prefix or not to_prefix:
        return p
    if p.startswith(from_prefix):
        # Make sure we don't introduce duplicate slashes
        suffix = p[len(from_prefix):]
        return os.path.join(to_prefix.rstrip("/"), suffix.lstrip("/"))
    return p

def _load_jsonl_items(jsonl_path: str,
                      *,
                      max_items: Optional[int] = None,
                      start_line: int = 0,
                      from_prefix: Optional[str] = None,
                      to_prefix: Optional[str] = None,
                      force_id: bool = False,
                      id_prefix: str = "pair",
                      id_start: int = 0) -> List[Tuple[str, str, str, str]]:
    """
    Return: List of tuples (sample_id, style_path, content_path, target_path)
    - style_path can be empty string.
    - start_line: skip lines [0, start_line-1] and start from this index.
    - from_prefix/to_prefix: optional local path prefix rewrite (empty = disabled).
    - force_id: if True, ignore any provided id/sample_id and generate "id_prefix_{id_start + i:06d}".
    """
    items: List[Tuple[str, str, str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue
            if max_items is not None and len(items) >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            sp = obj.get("style_image") or ""
            cp = obj.get("content_image")
            tp = obj.get("target_image")
            # Path rewriting
            sp = _rewrite_path(sp, from_prefix, to_prefix) if sp else sp
            cp = _rewrite_path(cp, from_prefix, to_prefix) if cp else cp
            tp = _rewrite_path(tp, from_prefix, to_prefix) if tp else tp

            if not (cp and tp):
                continue
            if not (os.path.exists(cp) and os.path.exists(tp)):
                # Skip non-existent paths after rewrite
                continue

            if force_id:
                # Index within the kept range, not the original line index
                local_idx = len(items)
                sid_num = id_start + local_idx
                sid = f"{id_prefix}_{sid_num:06d}"
            else:
                sid = obj.get("id") or obj.get("sample_id")
                if not sid:
                    # If no id in JSONL, build from target basename
                    # but keep the legacy "pair_xxxxxx" pattern with start offset
                    local_idx = len(items)
                    sid_num = id_start + local_idx
                    sid = f"{id_prefix}_{sid_num:06d}"

            items.append((sid, sp, cp, tp))
    return items


# ---------------- OCR scorer with image cache ----------------
class PaddleOCROverlapScorer:
    def __init__(self,
                 *,
                 lang: str = "en",
                 confidence_threshold: float = 0.6,
                 min_word_len: int = 1,
                 use_gpu: bool = True,
                 use_tensorrt: bool = False,
                 trt_precision: str = "fp16",
                 rec_batch_num: int = 96,
                 det_limit_side_len: Optional[int] = 512,
                 use_angle_cls: bool = False):
        self.lang = lang
        self.conf_th = float(confidence_threshold)
        self.min_word_len = max(1, int(min_word_len))
        self.available = _paddleocr_available
        self.ocr = None
        self.det_limit_side_len = det_limit_side_len

        if not self.available:
            return

        init_kwargs = dict(
                use_angle_cls=bool(use_angle_cls),  # default False for speed
                lang=self.lang,
                use_gpu=bool(use_gpu),
                show_log=False,
        )
        # TRT / precision
        try:
            init_kwargs["use_tensorrt"] = bool(use_tensorrt)
            init_kwargs["precision"] = trt_precision
        except Exception:
            pass
        # Recognition batch size
        try:
            init_kwargs["rec_batch_num"] = int(rec_batch_num)
        except Exception:
            pass
        # Detector resolution
        try:
            if det_limit_side_len is not None and int(det_limit_side_len) > 0:
                init_kwargs["det_limit_side_len"] = int(det_limit_side_len)
        except Exception:
            pass

        # Initialize OCR
        self.ocr = PaddleOCR(**init_kwargs)

    @staticmethod
    def _normalize_to_words(text: str) -> List[str]:
        t = text.lower()
        t = re.sub(r"[^0-9a-z]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        ws = t.split(" ") if t else []
        return [w for w in ws if len(w) >= 1]

    @staticmethod
    def _iter_detections(results: Any):
        if isinstance(results, (list, tuple)):
            for item in results:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    box, text_info = item[0], item[1]
                    text, score = None, None
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text, score = text_info[0], text_info[1]
                    if isinstance(text, str):
                        yield box, text, score
                    else:
                        yield from PaddleOCROverlapScorer._iter_detections(item)
                else:
                    yield from PaddleOCROverlapScorer._iter_detections(item)

    def _ocr_to_counter(self, bgr: np.ndarray) -> Counter:
        if not self.available or self.ocr is None or bgr is None:
            return Counter()
        try:
            results = self.ocr.ocr(bgr)
        except Exception:
            return Counter()
        words: List[str] = []
        for _, text, score in self._iter_detections(results or []):
            try:
                conf = float(score) if score is not None else None
            except Exception:
                conf = None
            if conf is not None and conf < self.conf_th:
                continue
            words.extend(self._normalize_to_words(text))
        words = [w for w in words if len(w) >= self.min_word_len]
        return Counter(words)

    def extract_words_from_path(self,
                                img_path: str,
                                cache_dir: Optional[str] = None,
                                *,
                                min_tokens_retry: int = 0,
                                retry_det_side_len: Optional[int] = None,
                                reuse_cache_only: bool = False
                                ) -> Counter:
        """
        OCR with per-image cache -> Counter
        - cache_dir: cache directory; if hit, return cached result
        - min_tokens_retry: if tokens < threshold, retry with higher det side length
        - retry_det_side_len: fallback det_limit_side_len used for retry (e.g., 736)
        - reuse_cache_only: use cache only, do not run OCR (for offline stage 2)
        """
        if not self.available or self.ocr is None:
            return Counter()

        cpath = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = _cache_key_for_path(img_path)
            cpath = os.path.join(cache_dir, f"{key}.json")
            if os.path.exists(cpath):
                try:
                    with open(cpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    tokens = data.get("tokens", {})
                    if isinstance(tokens, dict):
                        return Counter({k: int(v) for k, v in tokens.items()})
                except Exception:
                    pass
            if reuse_cache_only:
                # cache-only mode but no hit
                return Counter()

        # Real OCR
        bgr = _read_bgr_fast(img_path)
        cnt = self._ocr_to_counter(bgr)

        # Adaptive retry
        if min_tokens_retry > 0 and sum(cnt.values()) < min_tokens_retry and retry_det_side_len:
            if self.ocr is not None:
                try:
                    init_kwargs = dict(
                        use_angle_cls=self.ocr.use_angle_cls if hasattr(self.ocr, "use_angle_cls") else False,
                        lang=self.lang,
                        use_gpu=True,
                        show_log=False,
                        rec_batch_num=getattr(self.ocr, "rec_batch_num", 96),
                        det_limit_side_len=int(retry_det_side_len),
                    )
                    # Try to inherit TRT settings
                    for k in ("use_tensorrt", "precision"):
                        if hasattr(self.ocr, k):
                            init_kwargs[k] = getattr(self.ocr, k)
                    ocr_retry = PaddleOCR(**init_kwargs)
                    results = ocr_retry.ocr(bgr)
                    words: List[str] = []
                    for _, text, score in self._iter_detections(results or []):
                        conf = float(score) if score is not None else None
                        if conf is not None and conf < self.conf_th:
                            continue
                        words.extend(self._normalize_to_words(text))
                    words = [w for w in words if len(w) >= self.min_word_len]
                    cnt = Counter(words)
                except Exception:
                    pass

        if cpath:
            try:
                _write_json(cpath, {"tokens": dict(cnt)})
            except Exception:
                pass
        return cnt

    @staticmethod
    def f1(c_cnt: Counter, o_cnt: Counter) -> float:
        if not c_cnt and not o_cnt:
            return 1.0
        if not c_cnt or not o_cnt:
            return 0.0
        tot_c = sum(c_cnt.values())
        tot_o = sum(o_cnt.values())
        matched = sum(min(c_cnt[w], o_cnt.get(w, 0)) for w in c_cnt)
        recall = matched / tot_c if tot_c else 0.0
        precision = matched / tot_o if tot_o else 0.0
        return float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0


# ---------------- Per-GPU worker ----------------
def _gpu_worker_proc(gpu_id: int,
                     items: List[Tuple[str, str, str, str]],
                     args_dict: Dict[str, Any],
                     progress_q,    # Manager().Queue()
                     result_q       # Manager().Queue()
                     ):
    use_gpu = bool(args_dict.get("use_gpu", True))
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            import paddle
            paddle.device.set_device("gpu:0")
        except Exception:
            pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import paddle
            paddle.device.set_device("cpu")
        except Exception:
            pass

    ratings_dir = args_dict["ratings_dir"]

    scorer = PaddleOCROverlapScorer(
        lang=args_dict["ocr_lang"],
        confidence_threshold=args_dict["ocr_conf_thres"],
        min_word_len=args_dict["ocr_min_word_len"],
        use_gpu=use_gpu,
        use_tensorrt=args_dict["ocr_use_tensorrt"],
        trt_precision=args_dict["ocr_trt_precision"],
        rec_batch_num=args_dict["ocr_rec_batch_num"],
        det_limit_side_len=args_dict["ocr_det_limit_side_len"],
        use_angle_cls=args_dict["ocr_use_angle_cls"],
    )

    cache_dir = args_dict.get("cache_dir")
    reuse_cache_only = bool(args_dict.get("reuse_cache_only", False))
    min_tokens_retry = int(args_dict.get("min_tokens_retry", 0))
    retry_det_side_len = args_dict.get("retry_det_side_len")

    # Prefetch queue (store tuples; OCR reads paths + uses cache)
    queue_max = max(64, args_dict["ocr_rec_batch_num"] * int(args_dict.get("prefetch_mult", 8)))
    q = deque()
    q_lock = threading.Lock()
    cv_not_full = threading.Condition(q_lock)
    cv_not_empty = threading.Condition(q_lock)
    done = {"flag": False}

    def loader():
        for tup in items:
            with cv_not_full:
                while len(q) >= queue_max:
                    cv_not_full.wait()
                q.append(tup)  # (sid, sp, cp, tp)
                cv_not_empty.notify()
        with cv_not_empty:
            done["flag"] = True
            cv_not_empty.notify_all()

    t = threading.Thread(target=loader, daemon=True)
    t.start()

    processed = 0
    f1_sum = 0.0

    while True:
        with cv_not_empty:
            while not q and not done["flag"]:
                cv_not_empty.wait()
            if not q and done["flag"]:
                break
            sid, sp, cp, tp = q.popleft()
            cv_not_full.notify()

        # Image-level OCR cache: OCR each image -> Counter
        c_cnt = scorer.extract_words_from_path(
            cp, cache_dir,
            min_tokens_retry=min_tokens_retry,
            retry_det_side_len=retry_det_side_len,
            reuse_cache_only=reuse_cache_only
        )
        o_cnt = scorer.extract_words_from_path(
            tp, cache_dir,
            min_tokens_retry=min_tokens_retry,
            retry_det_side_len=retry_det_side_len,
            reuse_cache_only=reuse_cache_only
        )
        f1 = PaddleOCROverlapScorer.f1(c_cnt, o_cnt)
        f1_sum += float(f1)
        processed += 1

        # Stream each item's result back to the main process, which aggregates
        # everything into a single consolidated scores file (no per-item files).
        try:
            progress_q.put({
                "sample_id": sid,
                "style_image": sp,
                "content_image": cp,
                "target_image": tp,
                "paddleocr_text_overlap_f1": float(f1),
            })
        except Exception:
            pass

    try:
        result_q.put({
            "gpu_id": gpu_id,
            "num_items": processed,
            "paddleocr_text_overlap_f1_avg": (f1_sum / processed) if processed else 0.0
        })
    except Exception:
        pass


# ---------------- Helpers ----------------
def _shard_roundrobin(items: List[Tuple[str, str, str, str]], n: int) -> List[List[Tuple[str, str, str, str]]]:
    shards = [[] for _ in range(n)]
    for i, it in enumerate(items):
        shards[i % n].append(it)
    return shards


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="FAST OCR-only filtering (multi-GPU, image-cache, progress queue)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--jsonl", type=str, help="JSONL with style_image(optional), content_image, target_image")
    mode.add_argument("--content", type=str, help="Single: content image path")
    parser.add_argument("--target", type=str, help="Single: target image path")
    parser.add_argument("--style", type=str, default="", help="Single: style image path (optional)")

    parser.add_argument("--ratings-dir", type=str, default="output/cvpr_chart_restyle/ratings")
    parser.add_argument("--max-items", type=int, default=None)

    # New: starting line & forced ID numbering
    parser.add_argument("--start-line", type=int, default=0,
                        help="Start reading JSONL from this 0-based line index (skip earlier lines).")
    parser.add_argument("--force-id", action="store_true",
                        help="Force sample_id format using --id-prefix and --id-start.")
    parser.add_argument("--id-prefix", type=str, default="pair",
                        help="Prefix for forced sample_id when --force-id is set (default: 'pair').")
    parser.add_argument("--id-start", type=int, default=0,
                        help="Starting number for forced sample_id when --force-id is set.")

    # Optional path prefix rewriting (disabled by default; the pipeline is fully local).
    parser.add_argument("--from-prefix", type=str, default="",
                        help="Optional prefix to rewrite FROM when loading JSONL (empty = no rewrite).")
    parser.add_argument("--to-prefix", type=str, default="",
                        help="Optional prefix to rewrite TO when loading JSONL (empty = no rewrite).")

    # Compute device. "gpu" needs CUDA 11.8 + cuDNN 8.6 on LD_LIBRARY_PATH;
    # "cpu" runs PaddleOCR on CPU (no CUDA/cuDNN required).
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="Run PaddleOCR on GPU (default) or CPU.")
    # Multi-GPU (ignored when --device cpu, where a single worker is used).
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")

    # OCR model/runtime
    parser.add_argument("--rec-batch-num", type=int, default=128)
    parser.add_argument("--det-limit-side-len", type=int, default=512)
    parser.add_argument("--use-tensorrt", action="store_true")
    parser.add_argument("--trt-precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--use-angle-cls", action="store_true", help="Default off for speed")

    # OCR text filtering
    parser.add_argument("--ocr-lang", type=str, default="en")
    parser.add_argument("--ocr-conf-thres", type=float, default=0.6)
    parser.add_argument("--ocr-min-word-len", type=int, default=1)

    # Cache / Retry
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for per-image OCR cache")
    parser.add_argument("--reuse-cache-only", action="store_true", help="Only use cache; no new OCR")
    parser.add_argument("--min-tokens-retry", type=int, default=0, help="If tokens < N, retry with higher det side")
    parser.add_argument("--retry-det-side-len", type=int, default=736)

    # Prefetch / Progress
    parser.add_argument("--prefetch-mult", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")

    args = parser.parse_args()
    show_progress = not args.no_progress

    # Single mode
    if not args.jsonl:
        if not (args.content and args.target):
            print("[ERROR] For single mode, provide --content and --target.")
            raise SystemExit(2)
        for p in (args.content, args.target):
            if not os.path.exists(p):
                print(f"[ERROR] File does not exist: {p}")
                raise SystemExit(2)
        sid = os.path.splitext(os.path.basename(args.target))[0] or "single"

        scorer = PaddleOCROverlapScorer(
            lang=args.ocr_lang,
            confidence_threshold=args.ocr_conf_thres,
            min_word_len=args.ocr_min_word_len,
            use_gpu=(args.device == "gpu"),
            use_tensorrt=args.use_tensorrt,
            trt_precision=args.trt_precision,
            rec_batch_num=args.rec_batch_num,
            det_limit_side_len=(None if args.det_limit_side_len <= 0 else args.det_limit_side_len),
            use_angle_cls=args.use_angle_cls,
        )

        cache_dir = args.cache_dir
        c_cnt = scorer.extract_words_from_path(
            args.content, cache_dir,
            min_tokens_retry=args.min_tokens_retry,
            retry_det_side_len=(None if args.retry_det_side_len <= 0 else args.retry_det_side_len),
            reuse_cache_only=args.reuse_cache_only
        )
        o_cnt = scorer.extract_words_from_path(
            args.target, cache_dir,
            min_tokens_retry=args.min_tokens_retry,
            retry_det_side_len=(None if args.retry_det_side_len <= 0 else args.retry_det_side_len),
            reuse_cache_only=args.reuse_cache_only
        )
        f1 = PaddleOCROverlapScorer.f1(c_cnt, o_cnt)

        os.makedirs(args.ratings_dir, exist_ok=True)
        out_path = os.path.join(args.ratings_dir, f"_ocr_{_sanitize_sample_id(sid)}.json")
        _write_json(out_path, {
            "sample_id": sid,
            "style_image": args.style or "",
            "content_image": args.content,
            "target_image": args.target,
            "paddleocr_text_overlap_f1": float(f1)
        })
        print(json.dumps({
            "sample_id": sid,
            "style_image": args.style or "",
            "content_image": args.content,
            "target_image": args.target,
            "paddleocr_text_overlap_f1": float(f1)
        }, ensure_ascii=False, indent=2))
        print(f"Single OCR result written to: {out_path}")
        return

    # Batch mode
    items = _load_jsonl_items(
        args.jsonl,
        max_items=args.max_items,
        start_line=max(0, int(args.start_line)),
        from_prefix=args.from_prefix,
        to_prefix=args.to_prefix,
        force_id=bool(args.force_id),
        id_prefix=args.id_prefix,
        id_start=int(args.id_start),
    )  # (sid, sp, cp, tp)

    if not items:
        print(f"[ERROR] No valid items found in JSONL: {args.jsonl}")
        raise SystemExit(2)

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if args.device == "cpu":
        # On CPU there is no benefit to multiple device shards; use one worker.
        gpu_ids = [0]
    n_gpu = max(1, len(gpu_ids))
    shards = _shard_roundrobin(items, n_gpu)

    mp_ctx = mp.get_context("spawn")
    with mp_ctx.Manager() as manager:
        progress_q = manager.Queue()
        result_q = manager.Queue()

        args_dict = {
            "ratings_dir": args.ratings_dir,
            "ocr_lang": args.ocr_lang,
            "ocr_conf_thres": float(args.ocr_conf_thres),
            "ocr_min_word_len": int(args.ocr_min_word_len),
            "ocr_use_tensorrt": bool(args.use_tensorrt),
            "ocr_trt_precision": args.trt_precision,
            "ocr_rec_batch_num": int(args.rec_batch_num),
            "ocr_det_limit_side_len": (None if args.det_limit_side_len <= 0 else int(args.det_limit_side_len)),
            "ocr_use_angle_cls": bool(args.use_angle_cls),
            "cache_dir": args.cache_dir,
            "reuse_cache_only": bool(args.reuse_cache_only),
            "min_tokens_retry": int(args.min_tokens_retry),
            "retry_det_side_len": (None if args.retry_det_side_len <= 0 else int(args.retry_det_side_len)),
            "prefetch_mult": max(1, int(args.prefetch_mult)),
            "use_gpu": (args.device == "gpu"),
        }

        procs: List[mp.Process] = []
        for gid, shard in zip(gpu_ids, shards):
            p = mp_ctx.Process(
                target=_gpu_worker_proc,
                args=(gid, shard, args_dict, progress_q, result_q),
                daemon=False
            )
            p.start()
            procs.append(p)

        total = len(items)
        processed = 0
        all_items: List[Dict[str, Any]] = []
        if show_progress and tqdm is not None:
            with tqdm(total=total, unit="pair", desc="Stage IV-b OCR F1") as pbar:
                while processed < total:
                    try:
                        item = progress_q.get(timeout=0.5)
                        all_items.append(item)
                        processed += 1
                        pbar.update(1)
                    except Exception:
                        pass
        else:
            while processed < total:
                item = progress_q.get()
                all_items.append(item)
                processed += 1

        for p in procs:
            p.join()

        shard_results: List[Dict[str, Any]] = []
        while not result_q.empty():
            try:
                shard_results.append(result_q.get_nowait())
            except Exception:
                break

    total_items = sum(r.get("num_items", 0) for r in shard_results)
    weighted_avg = 0.0
    for r in shard_results:
        ni = r.get("num_items", 0)
        fa = r.get("paddleocr_text_overlap_f1_avg", 0.0)
        if total_items > 0:
            weighted_avg += fa * (ni / total_items)

    summary = {
        "num_items": int(total_items),
        "paddleocr_text_overlap_f1_avg": float(weighted_avg),
        "shards": shard_results,
    }
    # Write ONE consolidated scores file (summary + every item's F1).
    all_items.sort(key=lambda d: d.get("sample_id", ""))
    overall_path = os.path.join(args.ratings_dir, "ocr_scores.json")
    _write_json(overall_path, {"summary": summary, "items": all_items})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"OCR scores written to: {overall_path}")


if __name__ == "__main__":
    main()

# Example:
# pip install paddlepaddle-gpu==2.6.1 paddleocr==2.7.0.3
# python stage4b_ocr_filtering.py \
#   --jsonl ./training_set_300k_all_for_chart_style_transfer.jsonl \
#   --ratings-dir output/cvpr_chart_restyle/ratings_fast \
#   --cache-dir output/cvpr_chart_restyle/ocr_cache \
#   --gpus 0,1,2,3,4,5,6,7 \
#   --rec-batch-num 256 \
#   --det-limit-side-len 736 \
#   --use-tensorrt --trt-precision fp16 \
#   --ocr-conf-thres 0.6 --ocr-min-word-len 2 \
#   --min-tokens-retry 8 --retry-det-side-len 736 --use-angle-cls \
#   --start-line 290000 \
#   --force-id --id-prefix pair --id-start 290000

