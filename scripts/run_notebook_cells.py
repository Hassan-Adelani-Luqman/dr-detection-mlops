"""Execute selected notebook cells without re-running the whole notebook.

The training notebook takes hours to run end to end (8 experiments). This runs
only the cells you name, in order, in one kernel:

  --setup  cells needed for state (imports, paths, helpers, data); their saved
           outputs are left exactly as they are
  --run    cells whose new outputs are written back into the notebook

Everything else in the file is untouched: the notebook is edited as raw JSON
and written back with its original formatting and line endings, so the git
diff shows only the cells that ran.

    python scripts/run_notebook_cells.py notebook/diabetic_retinopathy_detection.ipynb \\
        --setup 6,7,8,9 --run 37,38 --prelude "print('hello')"
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def _clean_outputs(outputs, keep_stderr: bool, jpeg_over_kb: int, quality: int = 85) -> list:
    """Drop stderr streams (TensorFlow log noise) and shrink large photo figures.

    PNG figures bigger than ``jpeg_over_kb`` are re-encoded as JPEG: grids of
    fundus photographs compress badly as PNG and would bloat the notebook.
    """
    from PIL import Image

    cleaned = []
    for out in json.loads(json.dumps(outputs)):
        if out["output_type"] == "stream" and out["name"] == "stderr" and not keep_stderr:
            continue
        data = out.get("data", {})
        png = data.get("image/png")
        if png and jpeg_over_kb and len(base64.b64decode(png)) > jpeg_over_kb * 1024:
            img = Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality, optimize=True)
            data["image/jpeg"] = base64.b64encode(buf.getvalue()).decode()
            del data["image/png"]
        cleaned.append(out)
    return cleaned


def _indices(text: str) -> list:
    return [int(x) for x in text.split(",") if x.strip()] if text else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--setup", default="", help="Comma-separated cell indices run for state only.")
    ap.add_argument("--run", required=True, help="Comma-separated cell indices whose outputs are saved.")
    ap.add_argument("--prelude", default="", help="Python run first in the kernel (not saved).")
    ap.add_argument("--timeout", type=int, default=1800, help="Seconds per cell.")
    ap.add_argument("--keep-stderr", action="store_true",
                    help="Keep stderr streams in saved outputs (dropped by default).")
    ap.add_argument("--jpeg-over-kb", type=int, default=300,
                    help="Re-encode PNG figures larger than this as JPEG (0 = never).")
    args = ap.parse_args(argv)

    raw_bytes = args.notebook.read_bytes()
    text = raw_bytes.decode("utf-8")
    crlf = "\r\n" in text
    trailing_newline = text.endswith("\n")
    raw = json.loads(text)

    setup, run = _indices(args.setup), _indices(args.run)
    for i in setup + run:
        if raw["cells"][i]["cell_type"] != "code":
            raise SystemExit(f"Cell {i} is not a code cell.")

    node = nbformat.reads(text, as_version=4)            # independent copy for execution
    client = NotebookClient(node, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(args.notebook.parent.resolve())}})
    order = [("setup", i) for i in setup] + [("run", i) for i in run]
    with client.setup_kernel():
        if args.prelude:
            client.execute_cell(nbformat.v4.new_code_cell(args.prelude), -1)
        for kind, i in order:
            first = next((l for l in node.cells[i].source.splitlines() if l.strip()), "")
            print(f"[{kind}] cell {i}: {first[:70]}", flush=True)
            try:
                client.execute_cell(node.cells[i], i)
            except CellExecutionError as exc:
                print(f"\nCell {i} failed; notebook NOT written.\n{exc}", file=sys.stderr)
                return 1
            if kind == "run":
                raw["cells"][i]["outputs"] = _clean_outputs(node.cells[i].outputs,
                                                            args.keep_stderr, args.jpeg_over_kb)
                raw["cells"][i]["execution_count"] = node.cells[i].execution_count

    out = json.dumps(raw, indent=1, ensure_ascii=False)
    if trailing_newline:
        out += "\n"
    if crlf:
        out = out.replace("\n", "\r\n")
    args.notebook.write_bytes(out.encode("utf-8"))
    print(f"Wrote outputs of cells {run} to {args.notebook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
