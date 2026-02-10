# CPAanalysis

Small reproducible analysis of a Qualtrics export that evaluates how perceptions of CPA licensure pathway framing (including 150-credit-hour context) are **associated with** stated graduate enrollment intent.

## Run locally

```bash
python scripts/analyze.py
```

Outputs are written to:

- `outputs/report.md`
- `outputs/regression_coefficients.svg`

## Automation

GitHub Actions workflow: `.github/workflows/survey-analysis.yml`

It runs on every push and pull request, regenerates the analysis outputs, and uploads the report + coefficient plot as workflow artifacts.
