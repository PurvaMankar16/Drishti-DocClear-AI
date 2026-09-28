"""Private subprocess entry point for the Python 3.12 PaddleOCR runtime."""

import contextlib
import io
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "invalid OCR input"}))
        return 2

    input_path = Path(sys.argv[1]).resolve()
    backend_dir = Path(__file__).resolve().parents[3]
    allowed_dir = (backend_dir / "processed" / "ocr-input").resolve()
    try:
        input_path.relative_to(allowed_dir)
    except ValueError:
        print(json.dumps({"error": "invalid OCR input"}))
        return 2
    if not input_path.is_file():
        print(json.dumps({"error": "OCR input not found"}))
        return 2

    try:
        import paddle
        from paddleocr import PaddleOCR

        paddle.set_device("cpu")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ocr = PaddleOCR(
                lang="en",
                device="cpu",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            results = list(ocr.predict(str(input_path)))
        detections = _parse_results(results)
        text = "\n".join(item["text"] for item in detections)
        confidence = sum(item["confidence"] for item in detections) / len(detections) if detections else 0.0
        print(json.dumps({
            "engine": "paddleocr",
            "text": text,
            "confidence": confidence,
            "word_count": len(text.split()),
            "line_count": len(detections),
            "detections": detections,
        }))
        return 0
    except Exception:
        print(json.dumps({"error": "OCR processing failed"}))
        return 1


def _parse_results(results) -> list[dict]:
    detections: list[dict] = []
    for result in results:
        data = getattr(result, "json", None)
        data = data() if callable(data) else data
        payload = data.get("res", data) if isinstance(data, dict) else {}
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        boxes = payload.get("rec_boxes", payload.get("dt_polys", []))
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            box = boxes[index].tolist() if index < len(boxes) and hasattr(boxes[index], "tolist") else boxes[index] if index < len(boxes) else []
            box = _normalize_box(box)
            detections.append({"text": str(text), "confidence": max(0.0, min(1.0, score)), "bbox": box})
    return detections


def _normalize_box(box) -> list[list[float]]:
    if len(box) == 4 and not isinstance(box[0], (list, tuple)):
        x1, y1, x2, y2 = map(float, box)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return [[float(point[0]), float(point[1])] for point in box]


if __name__ == "__main__":
    raise SystemExit(main())
