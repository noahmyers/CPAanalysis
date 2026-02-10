#!/usr/bin/env python3
import csv
import json
import math
import os
from collections import Counter
from statistics import mean

INPUT_CSV = "Alternative CPA Pathways Survey_December 31, 2025_09.45.csv"
OUT_DIR = "outputs"
REPORT_PATH = os.path.join(OUT_DIR, "report.md")
PLOT_PATH = os.path.join(OUT_DIR, "regression_coefficients.svg")

DV_KEYWORDS = ["intent", "plan", "graduate", "macc", "master", "enroll", "desire", "likely"]
DV_NEGATIVE = ["currently", "aware", "explain", "describe", "satisfied", "discipline", "specialization", "years of", "major"]
PRED_KEYWORDS = ["150", "credit", "hours", "requirement", "cpa", "pathway", "barrier", "cost", "time"]

ORDER_BANK = [
    ["Significantly decreased desire", "Decreased desire", "No change in desire", "Increased desire", "Significantly increased desire"],
    ["Extremely unlikely", "Somewhat unlikely", "Neither likely nor unlikely", "Somewhat likely", "Extremely likely"],
    ["Not at all attractive", "Somewhat unattractive", "Neither attractive nor unattractive", "Somewhat attractive", "Very attractive"],
    ["Strongly disagree", "Somewhat disagree", "Neither agree nor disagree", "Somewhat agree", "Strongly agree"],
    ["Very negative", "Somewhat negative", "Neutral", "Somewhat positive", "Very positive"],
]


def load_qualtrics(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 4:
        raise ValueError("CSV too short")

    row0, row1, row2 = rows[0], rows[1], rows[2]
    has_import = all((c.strip().startswith("{") and "ImportId" in c) or c.strip() == "" for c in row2)
    header_rows = 3 if has_import else 2

    cols = []
    for i, qid in enumerate(row0):
        qtext = row1[i] if i < len(row1) else qid
        import_id = ""
        if has_import and i < len(row2):
            try:
                import_id = json.loads(row2[i]).get("ImportId", "") if row2[i].strip() else ""
            except Exception:
                import_id = ""
        cols.append({"idx": i, "qid": qid, "question": qtext, "import_id": import_id})

    data = rows[header_rows:]
    return cols, data


def nonempty_values(data, idx):
    vals = []
    for r in data:
        if idx < len(r):
            v = r[idx].strip()
            if v:
                vals.append(v)
    return vals


def infer_order(values):
    uniq = sorted(set(values))
    if len(uniq) < 3 or len(uniq) > 7:
        return None

    lower_to_original = {u.lower(): u for u in uniq}
    for order in ORDER_BANK:
        order_lower = [o.lower() for o in order]
        if set(lower_to_original.keys()).issubset(set(order_lower)):
            mapping = {}
            for i, o in enumerate(order):
                ol = o.lower()
                if ol in lower_to_original:
                    mapping[lower_to_original[ol]] = i + 1
            return mapping

    # numeric strings (e.g., rank)
    numeric = True
    nums = []
    for u in uniq:
        try:
            nums.append(float(u))
        except ValueError:
            numeric = False
            break
    if numeric:
        nums_sorted = sorted(nums)
        return {str(int(n)) if n.is_integer() else str(n): i + 1 for i, n in enumerate(nums_sorted)}
    return None


def score_text(text, keywords):
    lt = (text or "").lower()
    return sum(1 for k in keywords if k in lt)


def choose_dv(cols, data):
    best = None
    for c in cols:
        txt = c["question"]
        pos = score_text(txt, DV_KEYWORDS)
        neg = sum(1 for k in DV_NEGATIVE if k in txt.lower())
        values = nonempty_values(data, c["idx"])
        order_map = infer_order(values)
        if pos <= 0 or order_map is None:
            continue
        # prefer intent-like items with decent coverage and not clearly demographic/open text
        score = pos * 4 - neg * 3 + min(len(values), 200) / 50
        if "desire" in txt.lower() or "likely" in txt.lower() or "enroll" in txt.lower():
            score += 3
        if best is None or score > best["score"]:
            best = {"col": c, "order_map": order_map, "score": score, "n": len(values)}
    if best is None:
        raise ValueError("No suitable DV found")
    return best


def choose_predictors(cols, data, dv_idx, max_k=3, min_n=60):
    candidates = []
    for c in cols:
        if c["idx"] == dv_idx:
            continue
        txt = c["question"]
        if "_" in c["qid"]:
            continue
        pos = score_text(txt, PRED_KEYWORDS)
        if pos <= 0:
            continue
        values = nonempty_values(data, c["idx"])
        order_map = infer_order(values)
        if order_map is None:
            continue
        uniq = len(set(values))
        if uniq < 3 or len(values) < min_n:
            continue
        score = pos * 3 + min(len(values), 200) / 50
        if any(k in txt.lower() for k in ["150", "credit", "hours", "requirement"]):
            score += 4
        if "overall perception" in txt.lower():
            score += 3
        if "rank" in txt.lower() or "please rank" in txt.lower() or " - " in txt and "most beneficial" in txt.lower():
            score -= 6
        if "aware" in txt.lower() and "pathway" in txt.lower():
            score -= 2
        candidates.append({"col": c, "order_map": order_map, "score": score, "n": len(values), "uniq": uniq})

    candidates.sort(key=lambda x: (-x["score"], -x["n"]))

    # reduce near-duplicate batteries: keep first per Q prefix stem
    selected = []
    seen_stems = set()
    for cand in candidates:
        qid = cand["col"]["qid"]
        stem = qid.split("_")[0]
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        selected.append(cand)
        if len(selected) >= max_k:
            break
    return selected


def build_dataset(data, dv, predictors):
    rows = []
    for r in data:
        try:
            y_raw = r[dv["col"]["idx"]].strip()
        except IndexError:
            continue
        if not y_raw or y_raw not in dv["order_map"]:
            continue
        x_vals = []
        ok = True
        for p in predictors:
            idx = p["col"]["idx"]
            if idx >= len(r):
                ok = False
                break
            raw = r[idx].strip()
            if not raw or raw not in p["order_map"]:
                ok = False
                break
            x_vals.append(float(p["order_map"][raw]))
        if not ok:
            continue
        rows.append((float(dv["order_map"][y_raw]), x_vals))
    return rows


def transpose(m):
    return list(map(list, zip(*m)))


def matmul(a, b):
    out = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            aik = a[i][k]
            for j in range(len(b[0])):
                out[i][j] += aik * b[k][j]
    return out


def inv(matrix):
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = i
        for r in range(i + 1, n):
            if abs(aug[r][i]) > abs(aug[pivot][i]):
                pivot = r
        aug[i], aug[pivot] = aug[pivot], aug[i]
        pv = aug[i][i]
        if abs(pv) < 1e-12:
            raise ValueError("Singular matrix")
        for j in range(2 * n):
            aug[i][j] /= pv
        for r in range(n):
            if r == i:
                continue
            factor = aug[r][i]
            for j in range(2 * n):
                aug[r][j] -= factor * aug[i][j]
    return [row[n:] for row in aug]


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ols(y, x):
    n = len(y)
    X = [[1.0] + row[:] for row in x]
    Y = [[v] for v in y]
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    XtX_inv = inv(XtX)
    beta = matmul(matmul(XtX_inv, Xt), Y)
    yhat = [sum(beta[j][0] * X[i][j] for j in range(len(beta))) for i in range(n)]
    resid = [y[i] - yhat[i] for i in range(n)]
    k = len(beta)
    rss = sum(e * e for e in resid)
    tss = sum((yi - mean(y)) ** 2 for yi in y)
    r2 = 1 - (rss / tss) if tss > 0 else 0
    dof = max(n - k, 1)
    s2 = rss / dof
    varb = [[s2 * v for v in row] for row in XtX_inv]
    se = [math.sqrt(max(varb[i][i], 0.0)) for i in range(k)]
    tvals = [beta[i][0] / se[i] if se[i] > 0 else 0.0 for i in range(k)]
    pvals = [2 * (1 - normal_cdf(abs(t))) for t in tvals]
    ci = [(beta[i][0] - 1.96 * se[i], beta[i][0] + 1.96 * se[i]) for i in range(k)]
    return {
        "beta": [b[0] for b in beta],
        "se": se,
        "t": tvals,
        "p": pvals,
        "ci": ci,
        "r2": r2,
        "n": n,
    }


def write_coef_svg(labels, betas, cis, out_path):
    min_x = min(c[0] for c in cis + [(0, 0)])
    max_x = max(c[1] for c in cis + [(0, 0)])
    pad = (max_x - min_x) * 0.15 + 0.05
    min_x -= pad
    max_x += pad

    w, h = 920, 130 + 70 * len(labels)
    left, right, top = 260, 40, 50
    plot_w = w - left - right

    def sx(x):
        return left + (x - min_x) / (max_x - min_x) * plot_w

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    lines.append('<style>text{font-family:Arial,sans-serif;} .lbl{font-size:15px;} .tick{font-size:13px;fill:#333;} .title{font-size:20px;font-weight:bold;} .sub{font-size:13px;fill:#555;}</style>')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append('<text x="20" y="28" class="title">Association model coefficients (OLS on ordered intent scale)</text>')
    lines.append('<text x="20" y="46" class="sub">Points = coefficients; horizontal bars = 95% CI; dashed line at 0.</text>')

    x0 = sx(0)
    lines.append(f'<line x1="{x0:.1f}" y1="{top}" x2="{x0:.1f}" y2="{h-35}" stroke="#999" stroke-dasharray="6,6"/>')

    # x ticks
    for t in range(6):
        val = min_x + t * (max_x - min_x) / 5
        x = sx(val)
        lines.append(f'<line x1="{x:.1f}" y1="{h-30}" x2="{x:.1f}" y2="{h-24}" stroke="#666"/>')
        lines.append(f'<text x="{x:.1f}" y="{h-8}" text-anchor="middle" class="tick">{val:.2f}</text>')

    for i, (lab, b, ci) in enumerate(zip(labels, betas, cis)):
        y = top + 45 + i * 70
        x1, x2 = sx(ci[0]), sx(ci[1])
        xb = sx(b)
        lines.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="#1f77b4" stroke-width="3"/>')
        lines.append(f'<circle cx="{xb:.1f}" cy="{y}" r="6" fill="#d62728"/>')
        lines.append(f'<text x="{left-10}" y="{y+5}" text-anchor="end" class="lbl">{lab}</text>')
        lines.append(f'<text x="{x2+8:.1f}" y="{y+5}" class="tick">β={b:.2f}</text>')

    lines.append('</svg>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cols, data = load_qualtrics(INPUT_CSV)

    # Basic respondent filter
    idx_finished = next((c["idx"] for c in cols if c["qid"] == "Finished"), None)
    idx_consent = next((c["idx"] for c in cols if c["qid"] == "Q61"), None)
    filtered = []
    for r in data:
        if idx_finished is not None and (idx_finished >= len(r) or r[idx_finished].strip().lower() != "true"):
            continue
        if idx_consent is not None and (idx_consent >= len(r) or r[idx_consent].strip().lower() not in {"yes", "true"}):
            continue
        filtered.append(r)

    dv = choose_dv(cols, filtered)
    predictors = choose_predictors(cols, filtered, dv["col"]["idx"], max_k=3, min_n=max(40, int(dv["n"]*0.55)))
    dataset = build_dataset(filtered, dv, predictors)

    if len(dataset) < 25:
        raise ValueError("Not enough complete cases for selected model")

    y = [d[0] for d in dataset]
    x = [d[1] for d in dataset]
    model = ols(y, x)

    names = ["Intercept"] + [p["col"]["qid"] for p in predictors]
    coef_labels = [f'{p["col"]["qid"]}: {p["col"]["question"].replace(chr(10)," ")[:65]}...' for p in predictors]
    write_coef_svg(coef_labels, model["beta"][1:], model["ci"][1:], PLOT_PATH)

    dv_counts = Counter(nonempty_values(filtered, dv["col"]["idx"]))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Cross-sectional association analysis: CPA pathway perceptions and graduate enrollment intent\n\n")
        f.write("## Research question\n")
        f.write("How are students' perceptions of the CPA 150-credit-hour requirement and alternative-pathway framing associated with their **stated intent** to enroll in a graduate accounting program?\n\n")
        f.write("## Data and preprocessing\n")
        f.write(f"- Source file: `{INPUT_CSV}` (Qualtrics export).\n")
        f.write(f"- Header handling: detected **3** metadata rows (QID row, question text row, ImportId JSON row), then respondent rows.\n")
        f.write(f"- Analytic sample filtering: retained completed + consented responses (`Finished=True`, `Q61=Yes`).\n")
        f.write(f"- Final respondent rows after filtering: **{len(filtered)}**.\n\n")

        f.write("## Selected dependent variable (DV)\n")
        f.write(f"Programmatic scan selected **{dv['col']['qid']}** as the primary intent outcome based on keyword matches and ordered response scale suitability.\n\n")
        f.write(f"- **QID:** `{dv['col']['qid']}`\n")
        f.write(f"- **Question text:** {dv['col']['question'].replace(chr(10),' ')}\n")
        f.write(f"- **Non-missing responses (filtered):** {dv['n']}\n")
        f.write("- **Ordered coding used (low → high intent):**\n")
        for k, v in sorted(dv["order_map"].items(), key=lambda kv: kv[1]):
            f.write(f"  - {v}: {k}\n")
        f.write("\n")

        f.write("### DV distribution (descriptive)\n")
        for k, v in dv_counts.most_common():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Selected perception predictors\n")
        f.write("Programmatic keyword scan over question text (`150`, `credit`, `hours`, `requirement`, `CPA`, `pathway`, `barrier`, `cost`, `time`) retained ordered-response items and selected up to three highest-scoring predictors.\n\n")
        for p in predictors:
            f.write(f"- **{p['col']['qid']}**: {p['col']['question'].replace(chr(10),' ')}\n")

        f.write("\n## Association model\n")
        f.write("Because the selected DV is an ordered 5-level intent/desire item, we estimate a **linear probability-style OLS on the ordered scale** (higher value = stronger graduate intent/desire). This is an association model, not a causal model.\n\n")
        f.write(f"- Complete-case N used in model: **{model['n']}**\n")
        f.write(f"- R-squared: **{model['r2']:.3f}**\n\n")

        f.write("### Coefficients\n")
        f.write("| Term | Beta | Std. Error | z/t (normal approx) | p-value (approx) | 95% CI |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for i, nm in enumerate(names):
            ci = model["ci"][i]
            f.write(f"| {nm} | {model['beta'][i]:.3f} | {model['se'][i]:.3f} | {model['t'][i]:.3f} | {model['p'][i]:.4f} | [{ci[0]:.3f}, {ci[1]:.3f}] |\n")

        f.write("\n### Regression visual\n")
        f.write(f"![Coefficient plot](regression_coefficients.svg)\n\n")

        f.write("## QID-to-text mapping for selected variables\n")
        f.write("| QID | ImportId | Question text |\n|---|---|---|\n")
        selected = [dv["col"]] + [p["col"] for p in predictors]
        for c in selected:
            f.write(f"| {c['qid']} | {c['import_id'] or 'NA'} | {c['question'].replace(chr(10),' ')} |\n")

        f.write("\n## Limitations (important)\n")
        f.write("- This is **cross-sectional** survey data. Each respondent appears once, so results reflect contemporaneous relationships only.\n")
        f.write("- The outcome is **self-reported intent/desire**, not observed later enrollment behavior.\n")
        f.write("- Associations may reflect unmeasured differences across respondents (e.g., finances, prior work plans, program context), so coefficients should be interpreted as **related to / associated with**, not impacts or effects.\n")
        f.write("- The OLS-on-ordered-scale approach is pragmatic and reproducible here; an ordinal logit model could be used in richer statistical environments.\n")

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
