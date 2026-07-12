from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import webbrowser
from urllib.request import urlopen
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn


APP_TITLE = "Speech to Speech Monitor"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "cache" / "huggingface" / "hub"
DEFAULT_REALTIME_STATS_URL = os.environ.get("S2S_REALTIME_STATS_URL", "http://127.0.0.1:8765/v1/stats")


@dataclass
class ModelEntry:
    name: str
    path: str
    size_gb: float
    active: bool = False
    process_vram_gb: float | None = None
    process_cpu_percent: float | None = None
    process_ram_percent: float | None = None
    note: str | None = None


def _run_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return (completed.stdout or completed.stderr or "").strip()


def _windows_cpu_percent() -> float | None:
    if platform.system() != "Windows":
        return None
    text = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average",
        ]
    )
    try:
        return round(float(text), 1)
    except Exception:
        return None


def _windows_ram_percent() -> float | None:
    if platform.system() != "Windows":
        return None
    text = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$os=Get-CimInstance Win32_OperatingSystem; [math]::Round((($os.TotalVisibleMemorySize-$os.FreePhysicalMemory)/$os.TotalVisibleMemorySize)*100,1)",
        ]
    )
    try:
        return round(float(text), 1)
    except Exception:
        return None


def _parse_csv_lines(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows


def _gpu_stats() -> dict[str, Any]:
    summary = {
        "name": "Unavailable",
        "used_gb": None,
        "total_gb": None,
        "free_gb": None,
        "processes": [],
    }

    text = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = _parse_csv_lines(text)
    if rows:
        first = rows[0]
        if len(first) >= 4:
            summary["name"] = first[0]
            summary["used_gb"] = round(float(first[1]) / 1024, 2)
            summary["total_gb"] = round(float(first[2]) / 1024, 2)
            summary["free_gb"] = round(float(first[3]) / 1024, 2)

    proc_text = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    for row in _parse_csv_lines(proc_text):
        if len(row) >= 3:
            try:
                pid = int(row[0])
                cpu_percent = None
                ram_percent = None
                try:
                    import psutil

                    proc = psutil.Process(pid)
                    cpu_percent = round(proc.cpu_percent(interval=0.0), 1)
                    ram_percent = round(proc.memory_percent(), 1)
                except Exception:
                    pass
                summary["processes"].append(
                    {
                        "pid": pid,
                        "name": row[1],
                        "used_gb": round(float(row[2]) / 1024, 2),
                        "cpu_percent": cpu_percent,
                        "ram_percent": ram_percent,
                    }
                )
            except Exception:
                continue

    return summary


def _find_active_llama_model() -> str | None:
    if platform.system() != "Windows":
        return None
    output = _run_command(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'llama-server*'} | Select-Object -ExpandProperty CommandLine"])
    if not output:
        return None
    for token in output.split():
        if token.endswith(".gguf") or "gemma-4" in token.lower() or "qwen" in token.lower():
            return token.strip('"')
    return None


def _scan_local_models() -> list[ModelEntry]:
    model_files = sorted(DEFAULT_CACHE.glob("models--*/snapshots/*/*.gguf"))
    active_model = _find_active_llama_model()
    gpu = _gpu_stats()
    active_gpu_gb = None
    active_gpu_cpu = None
    active_gpu_ram = None
    if gpu["processes"]:
        for proc in gpu["processes"]:
            if proc["name"].lower().startswith("llama"):
                active_gpu_gb = proc["used_gb"]
                active_gpu_cpu = proc.get("cpu_percent")
                active_gpu_ram = proc.get("ram_percent")
                break

    entries: list[ModelEntry] = []
    for model_file in model_files:
        size_gb = round(model_file.stat().st_size / (1024**3), 2)
        active = active_model is not None and active_model.lower() in str(model_file).lower()
        note = "cached model file"
        if active:
            note = "currently selected by llama-server"
        entries.append(
            ModelEntry(
                name=model_file.name.replace(".gguf", ""),
                path=str(model_file),
                size_gb=size_gb,
                active=active,
                process_vram_gb=active_gpu_gb if active else None,
                process_cpu_percent=active_gpu_cpu if active else None,
                process_ram_percent=active_gpu_ram if active else None,
                note=note,
            )
        )
    return entries


def _overall_stats() -> dict[str, Any]:
    cpu = _windows_cpu_percent()
    ram = _windows_ram_percent()
    gpu = _gpu_stats()
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": cpu,
        "ram_percent": ram,
        "gpu": gpu,
    }


def _stats_payload() -> dict[str, Any]:
    stats = _overall_stats()
    models = [asdict(m) for m in _scan_local_models()]
    stats["models"] = models
    try:
        with urlopen(DEFAULT_REALTIME_STATS_URL, timeout=2) as response:
            stats["realtime"] = json.loads(response.read().decode("utf-8"))
    except Exception:
        stats["realtime"] = {}
    return stats


def _bar(label: str, value: float | None, unit: str, accent: str) -> str:
    if value is None:
        value_text = "Unavailable"
        width = 0
    else:
        value_text = f"{value:.1f}{unit}"
        width = max(0, min(100, int(value)))
    return f"""
    <div class="metric-card">
      <div class="metric-head">
        <span>{label}</span>
        <strong>{value_text}</strong>
      </div>
      <div class="meter"><span style="width:{width}%;background:{accent};"></span></div>
    </div>
    """


def _render_html() -> str:
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      --bg: #0a0f1d;
      --panel: rgba(13, 18, 32, 0.82);
      --panel-2: rgba(20, 26, 43, 0.9);
      --panel-3: rgba(9, 14, 26, 0.8);
      --text: #edf3ff;
      --muted: #8ea0bb;
      --border: rgba(148, 163, 184, 0.18);
      --blue: linear-gradient(90deg, #6ea8ff, #4dd0ff);
      --green: linear-gradient(90deg, #4ade80, #14b8a6);
      --amber: linear-gradient(90deg, #fbbf24, #fb7185);
      --shadow: 0 18px 48px rgba(0,0,0,.34);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(59,130,246,.18), transparent 26%),
        radial-gradient(circle at top right, rgba(20,184,166,.16), transparent 24%),
        radial-gradient(circle at bottom center, rgba(251,191,36,.08), transparent 28%),
        linear-gradient(180deg, #040812 0%, #0a0f1d 100%);
      color: var(--text);
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 20px 44px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(15, 23, 42, .96), rgba(2, 6, 23, .9));
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 22px 22px 20px;
      margin-bottom: 18px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -8% -35% auto;
      width: 260px;
      height: 260px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(77,208,255,.22), transparent 70%);
      pointer-events: none;
    }}
    .title-row {{
      display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; align-items: end;
      position: relative;
      z-index: 1;
    }}
    h1 {{ margin: 0; font-size: 30px; letter-spacing: -0.035em; line-height: 1.05; }}
    .subtitle {{ color: var(--muted); margin-top: 8px; font-size: 14px; max-width: 760px; }}
    .pill {{
      display: inline-flex; align-items: center; gap: 8px;
      padding: 10px 14px; border-radius: 999px;
      background: rgba(77,208,255,.08); color: #bdf1ff; border: 1px solid rgba(77,208,255,.22);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .span-4 {{ grid-column: span 4; }}
    .span-6 {{ grid-column: span 6; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    .metric-card {{ margin-top: 12px; }}
    .metric-head {{
      display: flex; justify-content: space-between; gap: 16px; align-items: center; font-size: 14px;
      color: #d9e6ff;
    }}
    .metric-head strong {{ font-size: 18px; color: #fff; }}
    .meter {{
      margin-top: 10px; height: 12px; border-radius: 999px; overflow: hidden;
      background: rgba(148,163,184,.12);
    }}
    .meter span {{ display: block; height: 100%; border-radius: 999px; }}
    .section-label {{
      color: #c8d3e6; text-transform: uppercase; letter-spacing: .14em; font-size: 11px; margin-bottom: 8px;
    }}
    .model {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      margin-top: 12px;
    }}
    .model-top {{
      display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; align-items: start;
    }}
    .model-name {{ font-size: 16px; font-weight: 700; }}
    .model-path {{ color: var(--muted); font-size: 12px; word-break: break-all; margin-top: 6px; }}
    .model-meta {{ color: #dbeafe; font-size: 13px; display: flex; gap: 10px; flex-wrap: wrap; }}
    .tag {{
      padding: 5px 10px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border);
      background: rgba(255,255,255,.04);
    }}
    .tag.active {{ background: rgba(16,185,129,.14); border-color: rgba(16,185,129,.28); color: #86efac; }}
    .muted {{ color: var(--muted); }}
    .footer {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid rgba(148,163,184,.14);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .summary-item {{
      background: var(--panel-3);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}
    .summary-item .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .summary-item .v {{ margin-top: 6px; font-size: 20px; font-weight: 700; }}
    .summary-item .s {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 900px) {{
      .span-4, .span-6, .span-8 {{ grid-column: span 12; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="title-row">
        <div>
          <div class="section-label">Local voice stack monitor</div>
          <h1>{APP_TITLE}</h1>
          <div class="subtitle">Clean live view of CPU, RAM, GPU/VRAM, and loaded local models.</div>
        </div>
        <div class="pill">Auto-refreshing every 2 seconds</div>
      </div>
    </div>

    <div class="grid">
      <div class="card span-4" id="cpu-card"></div>
      <div class="card span-4" id="ram-card"></div>
      <div class="card span-4" id="gpu-card"></div>
      <div class="card span-12" id="realtime-card"></div>
      <div class="card span-12">
        <div class="section-label">Loaded models</div>
        <div id="models"></div>
        <div class="footer">
          Exact per-model VRAM is only available when a model runs in its own process. For llama.cpp we show the active process VRAM and the cached model footprint.
        </div>
      </div>
    </div>
  </div>
  <script>
    function pct(v) {{
      if (v === null || v === undefined || Number.isNaN(v)) return 0;
      return Math.max(0, Math.min(100, Number(v)));
    }}

    function fmt(v, digits = 1, suffix = '%') {{
      if (v === null || v === undefined || Number.isNaN(v)) return 'Unavailable';
      return `${{Number(v).toFixed(digits)}}${{suffix}}`;
    }}

    function card(title, value, subtitle, width, accent) {{
      return `
        <div class="section-label">${{title}}</div>
        <div class="summary-grid">
          <div class="summary-item">
            <div class="k">Current</div>
            <div class="v">${{value}}</div>
            <div class="s">${{subtitle}}</div>
          </div>
          <div class="summary-item">
            <div class="k">Usage</div>
            <div class="v">${{width}}%</div>
            <div class="s">Live system monitor</div>
          </div>
          <div class="summary-item">
            <div class="k">Bar</div>
            <div class="meter" style="margin-top:14px"><span style="width:${{width}}%;background:${{accent}};"></span></div>
          </div>
        </div>
      `;
    }}

    async function refresh() {{
      const res = await fetch('/api/stats');
      const data = await res.json();
      const cpu = data.cpu_percent;
      const ram = data.ram_percent;
      const gpu = data.gpu || {{}};
      const realtime = data.realtime || {{}};
      const gpuPct = gpu.used_gb == null || gpu.total_gb == null ? null : (gpu.used_gb / gpu.total_gb) * 100;
      document.getElementById('cpu-card').innerHTML = card('CPU', fmt(cpu, 1), 'Overall processor load', pct(cpu), 'var(--blue)');
      document.getElementById('ram-card').innerHTML = card('RAM', fmt(ram, 1), 'Overall memory pressure', pct(ram), 'var(--green)');
      document.getElementById('gpu-card').innerHTML = `
        <div class="section-label">GPU / VRAM</div>
        <div style="font-size:18px;font-weight:700">${{gpu.name || 'Unavailable'}}</div>
        <div class="subtitle" style="margin:8px 0 0">Used: <strong>${{gpu.used_gb == null ? 'Unavailable' : gpu.used_gb.toFixed(2) + ' GB'}}</strong>${{gpu.total_gb == null ? '' : ' / ' + gpu.total_gb.toFixed(2) + ' GB'}}</div>
        <div class="metric-card">
          <div class="metric-head"><span>VRAM usage</span><strong>${{gpuPct == null ? 'Unavailable' : gpuPct.toFixed(1) + '%'}}</strong></div>
          <div class="meter"><span style="width:${{pct(gpuPct)}}%;background:var(--amber);"></span></div>
        </div>
        <div class="summary-grid">
          <div class="summary-item">
            <div class="k">Free VRAM</div>
            <div class="v">${{gpu.free_gb == null ? 'Unavailable' : gpu.free_gb.toFixed(2) + ' GB'}}</div>
            <div class="s">Available on the GPU</div>
          </div>
          <div class="summary-item">
            <div class="k">Total VRAM</div>
            <div class="v">${{gpu.total_gb == null ? 'Unavailable' : gpu.total_gb.toFixed(2) + ' GB'}}</div>
            <div class="s">Detected device capacity</div>
          </div>
          <div class="summary-item">
            <div class="k">Active process VRAM</div>
            <div class="v">${{(gpu.processes || []).length ? (gpu.processes[0].used_gb.toFixed(2) + ' GB') : 'Unavailable'}}</div>
            <div class="s">Measured from the llama process</div>
          </div>
        </div>
      `;
      const conn = (realtime.connections || [])[0] || {{}};
      const phase = conn.phase || realtime.current_phase || 'idle';
      const phaseElapsed = conn.phase_elapsed_s == null ? null : Number(conn.phase_elapsed_s);
      const phaseTimings = conn.phase_timings_s || {{}};
      document.getElementById('realtime-card').innerHTML = `
        <div class="section-label">Realtime pipeline</div>
        <div class="summary-grid">
          <div class="summary-item">
            <div class="k">Current phase</div>
            <div class="v">${{phase}}</div>
            <div class="s">${{conn.phase_note || 'No active phase note'}}</div>
          </div>
          <div class="summary-item">
            <div class="k">Phase elapsed</div>
            <div class="v">${{phaseElapsed == null ? 'Unavailable' : phaseElapsed.toFixed(2) + ' s'}}</div>
            <div class="s">Current stage timer</div>
          </div>
          <div class="summary-item">
            <div class="k">Active connections</div>
            <div class="v">${{realtime.active_connections ?? 0}}</div>
            <div class="s">Realtime sessions in flight</div>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-head"><span>Step latency</span><strong>${{Object.keys(phaseTimings).length ? 'Tracked' : 'Unavailable'}}</strong></div>
          <div class="subtitle" style="margin-top:10px">
            ${{Object.keys(phaseTimings).length ? Object.entries(phaseTimings).map(([k, v]) => `${{k}}: ${{Number(v).toFixed(2)}}s`).join(' • ') : 'No live timings yet'}}
          </div>
        </div>
      `;
      const models = data.models || [];
      document.getElementById('models').innerHTML = models.length ? models.map(m => `
        <div class="model">
          <div class="model-top">
            <div>
              <div class="model-name">${{m.name}}</div>
              <div class="model-path">${{m.path}}</div>
            </div>
            <div class="model-meta">
              ${{m.active ? '<span class="tag active">Active</span>' : '<span class="tag">Cached</span>'}}
              <span class="tag">${{m.size_gb.toFixed(2)}} GB file</span>
              ${{m.process_vram_gb == null ? '' : `<span class="tag">${{m.process_vram_gb.toFixed(2)}} GB VRAM in process</span>`}}
              ${{m.process_cpu_percent == null ? '' : `<span class="tag">${{m.process_cpu_percent.toFixed(1)}}% CPU</span>`}}
              ${{m.process_ram_percent == null ? '' : `<span class="tag">${{m.process_ram_percent.toFixed(1)}}% RAM</span>`}}
            </div>
          </div>
          <div class="metric-card">
            <div class="metric-head">
              <span>Model footprint</span>
              <strong>${{m.size_gb.toFixed(2)}} GB</strong>
            </div>
            <div class="meter"><span style="width:${{Math.min(100, m.size_gb * 5)}}%;background:var(--blue);"></span></div>
          </div>
          <div class="subtitle" style="margin-top:10px">${{m.note || ''}}</div>
        </div>
      `).join('') : '<div class="muted">No GGUF models found in the local cache.</div>';
    }}
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_render_html())

    @app.get("/api/stats")
    def stats() -> JSONResponse:
        return JSONResponse(_stats_payload())

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local clean dashboard for CPU, RAM, GPU, and loaded models.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = create_app()
    if not args.no_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
