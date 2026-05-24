"""
GARCH(1,1) - Projeção de Volatilidade para WIN (^BVSP)
Versão GitHub Actions — salva index.html na raiz do repositório
"""

import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from datetime import datetime, timedelta
import warnings, os, json
warnings.filterwarnings("ignore")

# ── CONFIGURAÇÕES ─────────────────────────────
TICKER         = "^BVSP"
ANOS_HISTORICO = 5
HORIZONTE_DIAS = 5
TRADING_DAYS   = 252
OUTPUT_HTML    = "index.html"

# ── DADOS ─────────────────────────────────────
print("Baixando dados e estimando GARCH(1,1)...")
data_fim    = datetime.today()
data_inicio = data_fim - timedelta(days=365 * ANOS_HISTORICO)
df = yf.download(TICKER, start=data_inicio, end=data_fim, progress=False)
if df.empty:
    print("Erro: sem dados.")
    exit()

fechamento = df["Close"].dropna()
retornos   = 100 * np.log(fechamento / fechamento.shift(1)).dropna()

# ── GARCH ─────────────────────────────────────
modelo    = arch_model(retornos, vol="Garch", p=1, q=1, dist="normal", rescale=False)
resultado = modelo.fit(disp="off")
omega     = resultado.params["omega"]
alpha     = resultado.params["alpha[1]"]
beta      = resultado.params["beta[1]"]
persist   = alpha + beta

vol_hoje  = float(np.sqrt(resultado.conditional_volatility.iloc[-1]) * np.sqrt(TRADING_DAYS))
vol_lp    = float(np.sqrt(omega / (1 - alpha - beta)) * np.sqrt(TRADING_DAYS))
var_hoje  = float(resultado.conditional_volatility.iloc[-1]) ** 2
var_lp    = omega / (1 - alpha - beta)
ibov      = float(np.array(fechamento.iloc[-1]).flat[0])

# ── PROJEÇÃO ──────────────────────────────────
hoje  = pd.Timestamp(datetime.today().date())
datas = pd.bdate_range(start=hoje + pd.offsets.BDay(1), periods=HORIZONTE_DIAS)
proj  = []
for h in range(1, HORIZONTE_DIAS + 1):
    var_h      = var_lp + (persist ** h) * (var_hoje - var_lp)
    vol_d      = float(np.sqrt(var_h))
    vol_a      = vol_d * np.sqrt(TRADING_DAYS)
    amp_pts    = ibov * vol_d / 100
    amp_fin    = amp_pts * 0.20
    proj.append({
        "h":          h,
        "data":       datas[h-1].strftime("%d/%m"),
        "dia":        datas[h-1].strftime("%a"),
        "vol_diaria": round(vol_d, 4),
        "vol_anual":  round(vol_a, 2),
        "amp_pts":    round(amp_pts),
        "amp_fin":    round(amp_fin),
    })

def regime(v):
    if v < 15:   return ("Baixa",   "#22c55e", "#dcfce7")
    elif v < 22: return ("Normal",  "#eab308", "#fef9c3")
    elif v < 30: return ("Elevada", "#f97316", "#ffedd5")
    else:        return ("Stress",  "#ef4444", "#fee2e2")

prox      = proj[0]
rg_hoje   = regime(vol_hoje)
rg_prox   = regime(prox["vol_anual"])
direcao   = "Arrefecendo" if vol_hoje > vol_lp else "Subindo"
dir_icon  = "↘" if direcao == "Arrefecendo" else "↗"
dir_color = "#22c55e" if direcao == "Arrefecendo" else "#f97316"

# ── VOL HISTÓRICA ─────────────────────────────
vol_hist    = resultado.conditional_volatility.iloc[-126:] * np.sqrt(TRADING_DAYS)
hist_labels = [d.strftime("%d/%m") for d in vol_hist.index]
hist_values = [round(float(v), 2) for v in vol_hist.values]

# ── TABELA SEMANAL ────────────────────────────
rows_semana = ""
for p in proj:
    rg = regime(p["vol_anual"])
    rows_semana += f"""
        <tr>
          <td><span class="dia-label">{p['dia']}</span> {p['data']}</td>
          <td>{p['vol_diaria']:.3f}%</td>
          <td><span class="badge" style="background:{rg[2]};color:{rg[1]}">{p['vol_anual']:.2f}%</span></td>
          <td><span class="regime-tag" style="color:{rg[1]}">{rg[0]}</span></td>
          <td>±{p['amp_pts']:,} pts</td>
          <td>±R$ {p['amp_fin']:,}</td>
        </tr>"""

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GARCH WIN — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:      #0a0e1a;
    --surface: #111827;
    --border:  #1f2937;
    --text:    #e5e7eb;
    --muted:   #6b7280;
    --accent:  #3b82f6;
    --green:   #22c55e;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; min-height: 100vh; padding: 32px 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 32px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
  .header-left h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }}
  .header-left h1 span {{ color: var(--accent); }}
  .header-left p {{ color: var(--muted); font-size: 13px; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }}
  .header-right {{ text-align: right; font-size: 12px; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
  .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 18px; }}
  .kpi label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; display: block; margin-bottom: 8px; }}
  .kpi .val {{ font-size: 26px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; line-height: 1; }}
  .kpi .sub {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
  .next-card {{ background: var(--surface); border: 1px solid var(--border); border-left: 4px solid {rg_prox[1]}; border-radius: 10px; padding: 22px 24px; margin-bottom: 24px; }}
  .next-card h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 16px; }}
  .next-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
  .next-item label {{ font-size: 11px; color: var(--muted); display: block; margin-bottom: 4px; }}
  .next-item .val {{ font-size: 20px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 22px 24px; margin-bottom: 24px; }}
  .card h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 8px 12px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; border-bottom: 1px solid var(--border); }}
  td {{ padding: 12px 12px; border-bottom: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
  .regime-tag {{ font-weight: 600; font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; }}
  .dia-label {{ color: var(--muted); font-size: 11px; }}
  .params {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .param {{ background: #1a2035; border: 1px solid var(--border); border-radius: 6px; padding: 8px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }}
  .param span {{ color: var(--muted); }}
  .chart-wrap {{ height: 220px; position: relative; }}
  .live-badge {{ display: inline-flex; align-items: center; gap: 6px; background: #0d2b1a; border: 1px solid #22c55e33; border-radius: 20px; padding: 4px 12px; font-size: 11px; color: var(--green); font-family: 'IBM Plex Mono', monospace; }}
  .live-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}
  .footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 32px; font-family: 'IBM Plex Mono', monospace; }}
  @media (max-width: 700px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .next-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="header-left">
      <h1>GARCH<span>(1,1)</span> — WIN / IBOV</h1>
      <p>^BVSP · {len(fechamento)} pregões · {fechamento.index[0].strftime('%d/%m/%Y')} → {fechamento.index[-1].strftime('%d/%m/%Y')}</p>
    </div>
    <div class="header-right">
      <div class="live-badge"><span class="live-dot"></span> Atualizado às 19h</div><br>
      <strong style="color:var(--text)">{datetime.now().strftime('%d/%m/%Y %H:%M')}</strong>
    </div>
  </div>
  <div class="kpi-grid">
    <div class="kpi">
      <label>IBOV Atual</label>
      <div class="val">{ibov:,.0f}</div>
      <div class="sub">pontos</div>
    </div>
    <div class="kpi">
      <label>Vol Hoje</label>
      <div class="val" style="color:{rg_hoje[1]}">{vol_hoje:.2f}%</div>
      <div class="sub">{rg_hoje[0]} · anualizada</div>
    </div>
    <div class="kpi">
      <label>Vol Longo Prazo</label>
      <div class="val" style="color:var(--accent)">{vol_lp:.2f}%</div>
      <div class="sub">média histórica</div>
    </div>
    <div class="kpi">
      <label>Direção</label>
      <div class="val" style="color:{dir_color}">{dir_icon} {direcao}</div>
      <div class="sub">convergindo para {vol_lp:.1f}%</div>
    </div>
  </div>
  <div class="next-card">
    <h2>📅 Próximo Pregão — {prox['dia']} {prox['data']}</h2>
    <div class="next-grid">
      <div class="next-item">
        <label>Vol Projetada (anual)</label>
        <div class="val" style="color:{rg_prox[1]}">{prox['vol_anual']:.2f}% &nbsp;<span style="font-size:14px">{rg_prox[0]}</span></div>
      </div>
      <div class="next-item">
        <label>Amplitude Estimada</label>
        <div class="val">±{prox['amp_pts']:,} pts</div>
      </div>
      <div class="next-item">
        <label>Valor por Contrato</label>
        <div class="val" style="color:var(--green)">±R$ {prox['amp_fin']:,}</div>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>📈 Volatilidade Condicional — Últimos 126 Pregões</h2>
    <div class="chart-wrap">
      <canvas id="volChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>📆 Projeção Semanal Dia a Dia</h2>
    <table>
      <thead>
        <tr>
          <th>Data</th><th>Vol Diária</th><th>Vol Anual</th>
          <th>Regime</th><th>Amplitude</th><th>Valor/Contrato</th>
        </tr>
      </thead>
      <tbody>{rows_semana}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>⚙️ Parâmetros GARCH(1,1)</h2>
    <div class="params">
      <div class="param"><span>ω</span> {omega:.6f}</div>
      <div class="param"><span>α (choque)</span> {alpha:.4f}</div>
      <div class="param"><span>β (memória)</span> {beta:.4f}</div>
      <div class="param"><span>α+β (persistência)</span> {persist:.4f}</div>
      <div class="param"><span>1 pt WIN =</span> R$ 0,20</div>
    </div>
  </div>
  <div class="footer">GARCH(1,1) · RiskMetrics · Nobel Engle 2003 · Dados: Yahoo Finance (^BVSP)</div>
</div>
<script>
const labels = {json.dumps(hist_labels)};
const values = {json.dumps(hist_values)};
const lp     = {round(vol_lp, 2)};
const ctx    = document.getElementById('volChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'Vol Condicional (%)',
        data: values,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      }},
      {{
        label: 'Vol Longo Prazo',
        data: Array(labels.length).fill(lp),
        borderColor: '#6b7280',
        borderWidth: 1,
        borderDash: [6,3],
        pointRadius: 0,
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color: '#9ca3af', font: {{ family: 'IBM Plex Mono', size: 11 }} }} }},
      tooltip: {{ mode: 'index', intersect: false }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#4b5563', maxTicksLimit: 10, font: {{ family: 'IBM Plex Mono', size: 10 }} }}, grid: {{ color: '#1f2937' }} }},
      y: {{ ticks: {{ color: '#4b5563', font: {{ family: 'IBM Plex Mono', size: 10 }}, callback: v => v + '%' }}, grid: {{ color: '#1f2937' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Dashboard salvo em: {OUTPUT_HTML}")
print("🌐 Acesse: https://DayaneTeixeira.github.io/garch-win")
