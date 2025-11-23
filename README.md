# Gradient Boosting Language Model (GBLM)

勾配ブースティング決定木を用いた言語モデルの実装

## 概要

このプロジェクトは、LightGBMを使用した言語モデルの学習データ生成パイプラインです。
従来のニューラルネットワークベースの言語モデルとは異なり、勾配ブースティング決定木を用いてテキスト生成を行います。

## ディレクトリ構成

```
.
├── gblm_data/              # データ処理モジュール
│   ├── __init__.py
│   ├── config.py           # 設定管理
│   ├── corpus.py           # コーパス読み込み
│   ├── vocab.py            # 語彙構築
│   ├── tokenizer.py        # トークナイザ
│   └── dataset.py          # データセット生成
├── gblm_model/             # モデル学習・推論モジュール
│   ├── __init__.py
│   ├── config.py           # モデル設定
│   ├── metrics.py          # 評価指標
│   ├── train.py            # 学習パイプライン
│   └── inference.py        # 推論・テキスト生成
├── scripts/
│   ├── build_vocab.py      # 語彙構築スクリプト
│   ├── make_dataset.py     # データセット生成スクリプト
│   ├── train_gblm.py       # モデル学習スクリプト
│   └── sample_gblm.py      # テキスト生成スクリプト
├── artifacts/              # 生成ファイル保存先
│   ├── vocab.json          # 語彙リスト
│   ├── tokenizer.json      # トークナイザ
│   ├── gblm_data.npz       # 学習データ
│   ├── train.npz           # 訓練データ
│   ├── val.npz             # 検証データ
│   └── gblm_model.txt      # 学習済みモデル
└── cleaned_merged_fairy_tales_without_eos.txt  # サンプルコーパス
```

## セットアップ

### 1. 仮想環境の作成

```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

### 2. 依存パッケージのインストール

```bash
uv pip install pandas numpy
```

LightGBMとscikit-learnも必要です：
```bash
uv pip install lightgbm scikit-learn
```

## 使用方法

### 1. 語彙とトークナイザの構築

テキストコーパスから語彙を構築し、トークナイザを作成します。

```bash
python scripts/build_vocab.py \
    --corpus cleaned_merged_fairy_tales_without_eos.txt \
    --output-dir artifacts \
    --min-freq 3 \
    --top-k 3000 \
    --verbose
```

**パラメータ:**
- `--corpus`: 入力テキストファイルのパス
- `--output-dir`: 出力ディレクトリ（デフォルト: artifacts）
- `--min-freq`: 語彙に含める最小出現頻度（デフォルト: 5）
- `--top-k`: 語彙サイズの上限（デフォルト: 5000）
- `--lowercase`: テキストを小文字化（デフォルト: True）
- `--max-docs`: 処理する最大ドキュメント数

### 2. 学習データセットの生成

構築したトークナイザを使用して、GBLM用の学習データを生成します。

```bash
python scripts/make_dataset.py \
    --corpus cleaned_merged_fairy_tales_without_eos.txt \
    --tokenizer artifacts/tokenizer.json \
    --output-dir artifacts \
    --context-length 16 \
    --max-samples 100000 \
    --split \
    --verbose
```

**パラメータ:**
- `--corpus`: 入力テキストファイルのパス
- `--tokenizer`: トークナイザファイルのパス
- `--context-length`: コンテキストウィンドウサイズ（デフォルト: 16）
- `--max-samples`: 生成する最大サンプル数
- `--split`: 訓練/検証セットの分割を行う
- `--val-ratio`: 検証セットの割合（デフォルト: 0.1）
- `--shuffle`: サンプルをシャッフル（デフォルト: True）
- `--seed`: 乱数シード（デフォルト: 42）

### 3. モデルの学習

LightGBMを使用してGBLMモデルを学習します。

```bash
python scripts/train_gblm.py \
    --num-boost-round 500 \
    --early-stopping-rounds 20 \
    --learning-rate 0.05 \
    --num-leaves 64
```

**主要パラメータ:**
- `--num-boost-round`: ブースティングラウンド数（デフォルト: 500）
- `--early-stopping-rounds`: 早期停止ラウンド数（デフォルト: 20）
- `--learning-rate`: 学習率（デフォルト: 0.1）
- `--num-leaves`: 決定木の葉の数（デフォルト: 64）
- `--min-data-in-leaf`: 葉の最小データ数（デフォルト: 20）

### 4. テキスト生成

学習済みモデルを使用してテキストを生成します。

```bash
python scripts/sample_gblm.py \
    --prompt "Once upon a time" \
    --max-new-tokens 100 \
    --sampling top_k \
    --top-k 10 \
    --temperature 0.8
```

**パラメータ:**
- `--prompt`: 生成開始テキスト
- `--max-new-tokens`: 生成する最大トークン数
- `--sampling`: サンプリング方法（greedy, top_k, top_p, temperature）
- `--top-k`: top-kサンプリングのk値
- `--top-p`: top-pサンプリングのp値
- `--temperature`: 温度パラメータ（低いほど決定的）

## データセット形式

生成されるデータセットは以下の形式です：

- **X**: shape=(N, L)のコンテキスト行列
  - N: サンプル数
  - L: コンテキスト長
  - 各行は過去LトークンのトークンID
- **y**: shape=(N,)の次トークンID配列

## Pythonでの使用例

### 学習

```python
from gblm_model.config import GBLMTrainConfig, PathsConfig, TrainSplitConfig, LightGBMConfig
from gblm_model.train import train_gblm
from pathlib import Path

# 設定
cfg = GBLMTrainConfig(
    paths=PathsConfig(artifacts_dir=Path("artifacts")),
    split=TrainSplitConfig(valid_size=0.1, shuffle=True),
    lgbm=LightGBMConfig(
        learning_rate=0.05,
        num_leaves=64,
        num_boost_round=500,
        early_stopping_rounds=20
    )
)

# 学習実行
booster, metrics = train_gblm(cfg)
print(f"Valid accuracy: {metrics['valid_accuracy']:.4f}")
print(f"Valid perplexity: {metrics['valid_perplexity']:.2f}")
```

### テキスト生成

```python
from gblm_model.inference import load_booster, generate_text
from gblm_model.train import load_tokenizer
from pathlib import Path

# モデルとトークナイザの読み込み
artifacts_dir = Path("artifacts")
tokenizer = load_tokenizer(artifacts_dir / "tokenizer.json")
booster = load_booster(artifacts_dir / "gblm_model.txt")

# テキスト生成
text = generate_text(
    booster=booster,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    context_length=16,
    max_new_tokens=100,
    sampling="top_k",
    top_k=10,
    temperature=0.8
)
print(text)
```

## カスタムコーパスの使用

### CSVファイルの場合

```bash
python scripts/build_vocab.py \
    --corpus your_data.csv \
    --is-csv \
    --text-column "text" \
    --output-dir artifacts
```

### テキストファイルの場合

各行または段落が1つのドキュメントとして扱われます。

## 設定ファイルの使用

JSONファイルで設定を管理することも可能です：

```json
{
  "vocab": {
    "min_freq": 5,
    "top_k": 5000,
    "lowercase": true,
    "max_docs": null
  },
  "dataset": {
    "context_length": 16,
    "max_samples": 200000,
    "shuffle": true,
    "random_seed": 42
  },
  "paths": {
    "corpus_file": "data/corpus.txt",
    "artifacts_dir": "artifacts",
    "is_csv": false
  }
}
```

設定ファイルを使用する場合：
```bash
python scripts/build_vocab.py --config config.json
```

## 生成されるファイル

- `vocab.json`: 語彙リストと頻度情報
- `tokenizer.json`: トークナイザの設定と語彙マッピング
- `gblm_data.npz`: 全データセット
- `train.npz`, `val.npz`: 訓練/検証分割データ
- `gblm_model.txt`: 学習済みLightGBMモデル
- `gblm_train_metrics.json`: 学習時の評価指標
- `*_metadata.json`: 各データセットのメタ情報
- `build_stats.json`, `dataset_stats.json`: 統計情報

## トラブルシューティング

### メモリ不足の場合

- `--max-docs`で処理するドキュメント数を制限
- `--max-samples`で生成サンプル数を制限
- `--context-length`を小さくする

### OOV率が高い場合

- `--min-freq`を下げる
- `--top-k`を増やす

## ライセンス

MIT