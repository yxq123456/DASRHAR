# DASR-HAR

DASR-HAR is an open-source framework for robust multi-sensor human activity recognition under intermittent sensor outages. It provides Duration-Aware State Recovery, a causal event-triggered module that repairs the temporal state of a returning sensor stream using its retained state, current observation, outage duration, and context from sensors that remained online.

The project includes preprocessing for MHEALTH, PAMAP2, and WISDM; five-fold subject-wise evaluation; four rate-matched outage geometries; DASR and strong missing-sensor baselines; training and evaluation pipelines; fixed-window and continuous recovery analysis; statistical testing; and efficiency profiling.

## Environment

```powershell
git clone https://github.com/yxq123456/DASRHAR.git
cd DASRHAR
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

The reference environment uses Python 3.11.6, PyTorch 2.2.1, CUDA 12.1, and cuDNN 8.9.7. Core Python dependencies are pinned in `requirements.txt`.

## Raw data layout

Place the public datasets in the following locations:

```text
data/raw/mhealth/MHEALTHDATASET/mHealth_subject1.log
data/raw/pamap2/PAMAP2_Dataset/Protocol/subject101.dat
data/raw/wisdm/wisdm-dataset/raw/phone/accel/data_*.txt
data/raw/wisdm/wisdm-dataset/raw/phone/gyro/data_*.txt
data/raw/wisdm/wisdm-dataset/raw/watch/accel/data_*.txt
data/raw/wisdm/wisdm-dataset/raw/watch/gyro/data_*.txt
```

The datasets can also be downloaded with:

```powershell
python scripts/download_data.py --dataset all
```

If a data provider requires manual license confirmation, download and extract the corresponding archive into the layout above.

## Quick start

```powershell
python scripts/prepare_data.py --dataset mhealth
python scripts/make_outage_manifests.py --dataset mhealth --fold 1 --split val
python scripts/make_outage_manifests.py --dataset mhealth --fold 1 --split test
python scripts/train.py --dataset mhealth --model dasr --fold 1 --seed 3407
python scripts/evaluate.py --dataset mhealth --model dasr --fold 1 --seed 3407 --condition clean
python scripts/evaluate.py --dataset mhealth --model dasr --fold 1 --seed 3407 --condition single_burst_30
```

The default experiment protocol uses `mhealth`, `pamap2`, and `wisdm`, five subject-wise folds, and seeds `3407`, `2026`, and `930`. Robustness training uses 40% clean windows and 60% interrupted windows, equal probability over the four outage geometries, and missing rates sampled from `[0.1, 0.5]`. Checkpoints maximize:

```text
0.4 * clean Macro-F1 + 0.6 * mean outage Macro-F1
```

## Models

- `dasr`: complete Duration-Aware State Recovery model
- `clean_early_fusion`: early fusion trained only on clean windows
- `masked_early_fusion`: early fusion with stream availability masks
- `modality_dropout`: multi-branch modality-dropout baseline
- `state_hold`: preserves the interrupted stream state during an outage
- `hard_reset`: resets the interrupted stream state at re-entry
- `gru_d`: learned duration-decay baseline
- `matched_gru`: architecture-matched multi-branch GRU
- `brits`: offline bidirectional imputation reference followed by the matched classifier
- `cross_sensor_reconstruction`: cross-sensor reconstruction followed by classification
- `tcn_base`, `tcn_dasr`: cross-backbone evaluation models
- `dasr_no_duration`, `dasr_no_decay`, `dasr_no_context`, `dasr_fixed_gate`: DASR ablations

## Main scripts

- `scripts/prepare_data.py`: parse, align, filter, and window a dataset
- `scripts/audit_data.py`: generate subject, class, window, and native-missingness audits
- `scripts/make_splits.py`: materialize sample membership for a fixed fold
- `scripts/make_outage_manifests.py`: generate deterministic validation and test outages
- `scripts/train.py`: train one dataset, method, fold, and seed
- `scripts/evaluate.py`: run fixed-window evaluation and optionally save diagnostics
- `scripts/evaluate_continuous.py`: run causal continuous-stream recovery evaluation
- `scripts/analyze_mechanism.py`: aggregate StateGap and repair-gate diagnostics
- `scripts/statistical_analysis.py`: run paired subject-level bootstrap and Wilcoxon analysis
- `scripts/profile_efficiency.py`: measure parameters, FLOPs, model size, and latency
- `scripts/plot_results.py`: generate geometry, recovery, and mechanism figures

## Outputs

Checkpoints are written to `results/checkpoints`, raw predictions to `results/raw_predictions`, and aggregate metrics to `results/aggregated`. Each prediction record contains the dataset, fold, seed, subject, sample, label, prediction, probabilities, method, checkpoint, outage condition, geometry, missing rate, interrupted streams, outage duration, recovery horizon, and mask-manifest checksum.

## Protocol details

Synthetic outages are applied after normalization with training-subject statistics. PAMAP2 native availability and controlled synthetic outage masks remain separate. Controlled re-entry events are defined by the fixed synthetic mask manifest. The default protocol uses five-second windows, a 2.5-second step, and minimum label purity of 0.8. WISDM is balanced to at most 13 windows per subject-activity pair.

Validation and test manifests are deterministic and shared by every compared method. Training masks are generated online from a separate random stream. Missing observations never update the DASR recurrent cell. During an outage, the affected state decays toward a learned stream prior. At re-entry, repair uses the duration-decayed state, current observation, completed-outage duration, and masked-mean context from streams that were already online.

Result tables are derived directly from prediction files and do not use hard-coded values. Raw datasets, trained checkpoints, and generated results are not included in the repository.
