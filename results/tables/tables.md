# NoLBERT-Calls: results

## Table 1. Within-call answer retrieval, US calls Jan 2025 - Oct 2026 (20,323 calls, 254,575 questions; chance top-1 8.0%)

Each analyst question ranks every management answer of its call (ties count against it). % of questions; 95% interval resampling calls. '+ word overlap' blends z-scored TF-IDF with the model, weight chosen on the 300 validation calls of 2024.

| Model | Top-1 | 95% CI | Top-1 (call-weighted) | Top-3 | MRR |
|---|---|---|---|---|---|
| NoLBERT-Calls-2023 (released) | 78.0 | 77.8–78.2 | 80.2 | 94.0 | 86.4 |
| NoLBERT-Calls-2023 (released) + word overlap | 79.0 | 78.8–79.2 | 81.0 | 94.4 | 87.1 |

Top-1 (%) by call year:

| Model | 2025 | 2026 |
|---|---|---|
| NoLBERT-Calls-2023 (released) | 77.8 | 78.3 |
| NoLBERT-Calls-2023 (released) + word overlap | 78.8 | 79.3 |

## Time test: top-1 (%) by call year

500 random US calls a year for 2013-24 (seed 0), every test call for 2025-26.

| Model | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NoLBERT-Calls-2012 (stage 1, calls up to 2012) | 70.1 | 70.2 | 70.6 | 70.2 | 70.8 | 71.2 | 71.9 | 69.8 | 71.3 | 71.4 | 71.2 | 71.1 | 71.3 | 71.5 |
| NoLBERT-Calls-2023 (released) | 73.7 | 73.7 | 75.0 | 75.0 | 75.4 | 77.0 | 77.0 | 75.6 | 78.0 | 78.5 | 78.0 | 77.5 | 77.8 | 78.3 |

In-sample premium: lead over NoLBERT-Calls-2012 (stage 1, calls up to 2012) in 2013-23 (calls a 2023 model trained on) minus the lead in 2025-26 (calls no model saw), in points of top-1.

| Model | Lead 2013-23 | Lead 2025-26 | Premium |
|---|---|---|---|
| NoLBERT-Calls-2023 (released) | +5.3 | +6.7 | -1.4 |

## Table 2. Other tasks on 2025-26 calls (frozen representations)

Match: a prepared-remarks passage finds its call's answer passage among 1,000 calls (top-1, top-10). Role: analyst question vs executive answer, logistic probe (balanced accuracy). Masked: 100 finance terms masked in 3,000 passages (macro top-1, top-5). SubjECTive-QA probe: logistic regression on frozen embeddings, weighted F1 averaged over six properties and the authors' three splits. All in %.

| Model | Match top-1 | Match top-10 | Role | Masked top-1 | Masked top-5 | SubjECTive-QA probe F1 |
|---|---|---|---|---|---|---|
| NoLBERT-Calls-2023 (released) | 7.6 | 21.9 | 96.2 | 25.6 | 47.1 | 64.8 |

## Table 3. SubjECTive-QA, fine-tuned heads: test weighted F1 (%), mean (sd) over the authors' three splits

| Model | ASSERTIVE | CAUTIOUS | CLEAR | OPTIMISTIC | RELEVANT | SPECIFIC | Mean |
|---|---|---|---|---|---|---|---|
| NoLBERT-Calls-2023 (released) | 50.3 (4.3) | 61.3 (3.0) | 82.5 (0.8) | 59.4 (2.8) | 84.9 (1.1) | 56.3 (2.8) | 65.8 (1.5) |

## Table 4. GLUE dev (CoLA: Matthews correlation; QNLI, SST-2: accuracy; %)

| Model | cola | qnli | sst2 |
|---|---|---|---|
| NoLBERT-Calls-2023 (released) | 31.9 | 88.2 | 86.8 |
