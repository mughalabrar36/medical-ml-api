"""The web page (HTML + CSS + JS), kept in Python so no templates folder is needed."""
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Breast Cancer Prediction</title>
<style>
  :root {
    --bg1: #4f46e5; --bg2: #0ea5e9;
    --card: #ffffff; --text: #1e293b; --muted: #64748b; --line: #e2e8f0;
    --brand: #4f46e5; --brand-dark: #4338ca;
    --good: #16a34a; --good-bg: #dcfce7; --bad: #dc2626; --bad-bg: #fee2e2;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: linear-gradient(135deg, var(--bg1), var(--bg2));
    padding: 32px 16px;
  }
  .wrap { max-width: 860px; margin: 0 auto; }
  header { text-align: center; color: #fff; margin-bottom: 24px; }
  header .icon { font-size: 44px; }
  header h1 { margin: 6px 0 4px; font-size: 30px; }
  header p { margin: 0; opacity: .9; }
  .card {
    background: var(--card); border-radius: 18px; padding: 28px;
    box-shadow: 0 20px 45px rgba(15, 23, 42, .25); margin-bottom: 20px;
  }
  .card h2 { margin: 0 0 4px; font-size: 20px; }
  .card .sub { margin: 0 0 20px; color: var(--muted); font-size: 14px; }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px 20px; }
  label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
  label small { font-weight: 400; color: var(--muted); }
  input {
    width: 100%; padding: 11px 12px; font-size: 15px; color: var(--text);
    border: 1.5px solid var(--line); border-radius: 10px; outline: none;
    transition: border-color .15s, box-shadow .15s;
  }
  input:focus { border-color: var(--brand); box-shadow: 0 0 0 3px rgba(79, 70, 229, .15); }
  .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
  button {
    font: inherit; font-weight: 600; cursor: pointer; border-radius: 10px;
    padding: 12px 20px; border: 1.5px solid var(--line); background: #fff; color: var(--text);
    transition: transform .1s, background .15s;
  }
  button:hover { background: #f8fafc; }
  button:active { transform: scale(.98); }
  button.primary { background: var(--brand); border-color: var(--brand); color: #fff; flex: 1; min-width: 160px; }
  button.primary:hover { background: var(--brand-dark); }
  button:disabled { opacity: .6; cursor: wait; }
  #result { display: none; }
  .badge {
    display: inline-block; padding: 8px 18px; border-radius: 999px;
    font-weight: 700; font-size: 18px; text-transform: capitalize;
  }
  .badge.benign { background: var(--good-bg); color: var(--good); }
  .badge.malignant { background: var(--bad-bg); color: var(--bad); }
  .conf { margin: 12px 0 18px; color: var(--muted); }
  .bar-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; font-size: 14px; }
  .bar-row span:first-child { width: 80px; font-weight: 600; }
  .bar { flex: 1; height: 12px; background: var(--line); border-radius: 999px; overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; border-radius: 999px; transition: width .5s ease; }
  .bar i.benign { background: var(--good); }
  .bar i.malignant { background: var(--bad); }
  .bar-row span:last-child { width: 56px; text-align: right; }
  .error { display: none; background: var(--bad-bg); color: var(--bad); border-radius: 10px; padding: 12px 14px; margin-top: 16px; font-size: 14px; }
  details summary { cursor: pointer; font-weight: 600; }
  pre { background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 10px; overflow-x: auto; font-size: 13px; }
  .note { text-align: center; color: #fff; opacity: .9; font-size: 13px; margin-top: 8px; }
  @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } .card { padding: 20px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="icon">🩺</div>
    <h1>Breast Cancer Prediction</h1>
    <p>A machine learning model served through a real-time API</p>
  </header>

  <div class="card">
    <h2>Enter tumor measurements</h2>
    <p class="sub">All values are the mean measurements from the Wisconsin breast cancer dataset.</p>
    <form id="form" novalidate>
      <div class="grid">
        {% for f in fields %}
        <div>
          <label for="{{ f.key }}">{{ f.label }} <small>(typical {{ f.typical }})</small></label>
          <input type="number" id="{{ f.key }}" name="{{ f.key }}" required
                 min="{{ f.lo }}" max="{{ f.hi }}" step="any" placeholder="{{ f.typical }}">
        </div>
        {% endfor %}
      </div>
      <div class="actions">
        <button type="submit" class="primary" id="go">Predict</button>
        <button type="button" data-example="malignant">Malignant example</button>
        <button type="button" data-example="benign">Benign example</button>
        <button type="button" id="clear">Clear</button>
      </div>
      <div class="error" id="error"></div>
    </form>
  </div>

  <div class="card" id="result">
    <h2>Result</h2>
    <p class="sub">Prediction from the Random Forest model</p>
    <span class="badge" id="badge"></span>
    <p class="conf" id="conf"></p>
    <div class="bar-row"><span>Benign</span><div class="bar"><i class="benign" id="bar-benign"></i></div><span id="pct-benign"></span></div>
    <div class="bar-row"><span>Malignant</span><div class="bar"><i class="malignant" id="bar-malignant"></i></div><span id="pct-malignant"></span></div>
  </div>

  <div class="card">
    <details>
      <summary>Use the API directly</summary>
      <p class="sub" style="margin-top:12px">Send a POST request to <code>/predict</code> with JSON:</p>
<pre>curl -X POST {{ request.host_url }}predict \
  -H "Content-Type: application/json" \
  -d '{"radius_mean":17.99,"texture_mean":10.38,"perimeter_mean":122.8,
       "area_mean":1001,"smoothness_mean":0.1184,"compactness_mean":0.2776,
       "concavity_mean":0.3001,"concave_points_mean":0.1471,
       "symmetry_mean":0.2419,"fractal_dimension_mean":0.07871}'</pre>
    </details>
  </div>

  <p class="note">Educational project only. Not for real medical diagnosis.</p>
</div>

<script>
const EXAMPLES = {{ examples|tojson }};
const form = document.getElementById("form");
const errorBox = document.getElementById("error");
const result = document.getElementById("result");
const goBtn = document.getElementById("go");

function fillForm(values) {
  for (const [key, val] of Object.entries(values)) form.elements[key].value = val;
  errorBox.style.display = "none";
}

document.querySelectorAll("[data-example]").forEach(btn =>
  btn.addEventListener("click", () => fillForm(EXAMPLES[btn.dataset.example])));

document.getElementById("clear").addEventListener("click", () => {
  form.reset(); errorBox.style.display = "none"; result.style.display = "none";
});

function showError(messages) {
  errorBox.textContent = messages.join(" ");
  errorBox.style.display = "block";
  result.style.display = "none";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.style.display = "none";
  const payload = {};
  for (const input of form.querySelectorAll("input")) {
    if (input.value.trim() === "") { showError(["Please fill in all fields, or load an example."]); return; }
    payload[input.name] = Number(input.value);
  }
  goBtn.disabled = true; goBtn.textContent = "Predicting…";
  try {
    const res = await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) { showError(data.details || [data.error || "Something went wrong."]); return; }
    showResult(data);
  } catch (err) {
    showError(["Could not reach the server. Please try again."]);
  } finally {
    goBtn.disabled = false; goBtn.textContent = "Predict";
  }
});

function showResult(d) {
  const badge = document.getElementById("badge");
  badge.textContent = d.prediction;
  badge.className = "badge " + d.prediction;
  document.getElementById("conf").textContent = "Model confidence: " + (d.confidence * 100).toFixed(1) + "%";
  for (const k of ["benign", "malignant"]) {
    const pct = (d.probabilities[k] * 100).toFixed(1);
    document.getElementById("bar-" + k).style.width = pct + "%";
    document.getElementById("pct-" + k).textContent = pct + "%";
  }
  result.style.display = "block";
  result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
</script>
</body>
</html>
"""
