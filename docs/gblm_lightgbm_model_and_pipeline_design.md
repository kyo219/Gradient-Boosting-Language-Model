# GBLM モデル学習 & 推論パイプライン設計書  
Gradient Boosting Language Model (GBLM) – LightGBM Multiclass Vanilla 版

---

## 0. 目的

前段で用意したデータパイプライン（Kaggle Children Stories コーパス → vocab / tokenizer → GBLM 用 `(X, y)`）を前提として、

- LightGBM の **multiclass** を用いた **GBLM 本体の学習**
- 学習済みモデルを使った **推論（次トークン予測・テキスト生成）**
- それらをつなぐ **トレーニング & 推論パイプライン**

を実装できるようにするための設計をまとめる。

Optuna 等によるハイパラ探索は行わず、  
まずは **vanilla な LightGBM multiclass** を素直に適用する。

---

## 1. 前提

### 1.1 データ側の前提（既に用意済みとする）

- `artifacts/tokenizer.json`
  - 前段の設計で作成した `Tokenizer` の JSON
- `artifacts/gblm_data.npz`
  - `make_gblm_training_data()` による生成物
  - 中身:
    - `X`: shape `(N, L)` の `int32` 行列
    - `y`: shape `(N,)` の `int32` ベクトル

```python
npz = np.load("artifacts/gblm_data.npz")
X = npz["X"]  # (N, L)
y = npz["y"]  # (N,)
```

### 1.2 使用ライブラリ

- `lightgbm`
- `numpy`
- `scikit-learn`（train/val split 用に `train_test_split` を使ってよい）
- `dataclasses` / `pathlib` / `json` など標準ライブラリ

---

## 2. 全体アーキテクチャ（モデル側）

### 2.1 追加ディレクトリ構成（案）

```text
gblm_data/                  # 前段で作った「データ生成」モジュール

gblm_model/                 # 今回追加する「モデル学習・推論」モジュール
  __init__.py
  config.py                 # モデル & 学習設定 (dataclass)
  train.py                  # 学習パイプライン
  inference.py              # 推論 & テキスト生成ユーティリティ
  metrics.py                # 評価指標（accuracy, logloss, perplexity）

scripts/
  build_vocab.py            # (既存) vocab & tokenizer 生成
  make_dataset.py           # (既存) GBLM 用 X, y 生成
  train_gblm.py             # ★ 追加: LightGBM で GBLM 学習
  sample_gblm.py            # ★ 追加: 学習済みモデルでテキスト生成

artifacts/
  vocab.json
  tokenizer.json
  gblm_data.npz
  gblm_model.txt            # ★ 追加: 学習済み LightGBM モデル (テキスト or バイナリ)
```

---

## 3. Config 設計（モデル & 学習）

### 3.1 Config データクラス

```python
# gblm_model/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class TrainSplitConfig:
    valid_size: float = 0.1           # 検証用に用いる割合 (0.0〜1.0)
    shuffle: bool = True              # 分割前にシャッフルするか
    random_seed: int = 42             # シャッフル用シード


@dataclass
class LightGBMConfig:
    # LightGBM のパラメータ（vanilla）
    objective: str = "multiclass"
    num_class: Optional[int] = None   # 語彙サイズ V（コード側で上書き）
    learning_rate: float = 0.1
    num_leaves: int = 64
    max_depth: int = -1               # -1 = 制限なし
    min_data_in_leaf: int = 20
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    num_boost_round: int = 500
    early_stopping_rounds: int = 20
    n_jobs: int = -1

    def to_lgbm_params(self, num_class: int) -> Dict[str, Any]:
        """
        LightGBM Booster に渡すパラメータ dict を返す。
        num_class は外から上書きする。
        """
        return {
            "objective": self.objective,
            "num_class": num_class,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "n_jobs": self.n_jobs,
            # 評価指標は multi_logloss & multiclasserror をデフォルトで使う想定
            "metric": ["multi_logloss", "multi_error"],
        }


@dataclass
class PathsConfig:
    artifacts_dir: Path = Path("artifacts")
    tokenizer_json: str = "tokenizer.json"
    data_npz: str = "gblm_data.npz"
    model_file: str = "gblm_model.txt"   # LightGBM のモデル保存先


@dataclass
class GBLMTrainConfig:
    paths: PathsConfig
    split: TrainSplitConfig
    lgbm: LightGBMConfig
```

- `GBLMTrainConfig` を YAML / JSON / Python コードで定義し、`scripts/train_gblm.py` で読み込む想定。

---

## 4. モデル学習パイプライン設計

### 4.1 学習フロー（高レベル）

1. `tokenizer.json` を読み込んで `Tokenizer` を復元
2. `gblm_data.npz` を読み込んで `X`, `y` を取得
3. 語彙サイズ `V = len(tokenizer.itos)` を取得
4. `X`, `y` を学習用 / 検証用に分割（train/valid）
5. LightGBM Dataset を構築
6. LightGBM のパラメータを `LightGBMConfig` から構築
7. `lgb.train()` で学習（early stopping あり）
8. ベストイテレーションのモデルを保存
9. 学習ログ & 最終的な評価指標（accuracy, logloss, perplexity）を計算して表示

### 4.2 学習用関数インターフェース

```python
# gblm_model/train.py
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from gblm_model.config import GBLMTrainConfig
from gblm_data.tokenizer import Tokenizer   # 前段のモジュールを再利用
from gblm_model.metrics import (
    compute_accuracy,
    compute_multi_logloss,
    compute_perplexity,
)


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """
    artifacts/tokenizer.json から Tokenizer を復元する。
    """
    import json
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Tokenizer.from_dict(data)


def load_gblm_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    artifacts/gblm_data.npz から X, y を読み込む。
    """
    npz = np.load(data_path)
    X = npz["X"]
    y = npz["y"]
    return X, y


def train_gblm(
    cfg: GBLMTrainConfig,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    GBLM (LightGBM multiclass) の学習を実行し、学習済み Booster と
    評価メトリクスを返す。

    Returns:
        booster: 学習済み LightGBM Booster。
        metrics: dict (train/valid の accuracy, logloss, perplexity など)。
    """
    # 1. パス解決
    artifacts_dir = cfg.paths.artifacts_dir
    tokenizer_path = artifacts_dir / cfg.paths.tokenizer_json
    data_path = artifacts_dir / cfg.paths.data_npz
    model_path = artifacts_dir / cfg.paths.model_file

    # 2. Tokenizer & データ読み込み
    tokenizer = load_tokenizer(tokenizer_path)
    X, y = load_gblm_data(data_path)

    num_classes = len(tokenizer.itos)

    # 3. train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=cfg.split.valid_size,
        shuffle=cfg.split.shuffle,
        random_state=cfg.split.random_seed,
        stratify=y,  # クラス分布を維持
    )

    # 4. LightGBM Dataset
    n_features = X.shape[1]
    categorical_features = list(range(n_features))

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical_features,
        reference=dtrain,
        free_raw_data=False,
    )

    # 5. パラメータ構築
    params = cfg.lgbm.to_lgbm_params(num_class=num_classes)

    # 6. 学習 (early stopping あり)
    evals_result: Dict[str, Any] = {}
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=cfg.lgbm.num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        early_stopping_rounds=cfg.lgbm.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50,
    )

    # 7. 予測 & メトリクス計算
    # best_iteration_ を使って予測
    train_proba = booster.predict(X_train, num_iteration=booster.best_iteration)
    valid_proba = booster.predict(X_valid, num_iteration=booster.best_iteration)

    train_pred = train_proba.argmax(axis=1)
    valid_pred = valid_proba.argmax(axis=1)

    metrics = {
        "train_accuracy": float(compute_accuracy(y_train, train_pred)),
        "valid_accuracy": float(compute_accuracy(y_valid, valid_pred)),
        "train_logloss": float(compute_multi_logloss(y_train, train_proba)),
        "valid_logloss": float(compute_multi_logloss(y_valid, valid_proba)),
        "train_perplexity": float(compute_perplexity(y_train, train_proba)),
        "valid_perplexity": float(compute_perplexity(y_valid, valid_proba)),
        "best_iteration": int(booster.best_iteration),
        "lgbm_params": params,
        "config": {
            "train_split": asdict(cfg.split),
            "lgbm": asdict(cfg.lgbm),
        },
    }

    # 8. モデル保存
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))

    return booster, metrics
```

---

## 5. 評価指標 (metrics)

### 5.1 必要な指標

- `accuracy`（単純なクラストップ1精度）
- `multi_logloss`（LightGBM が内部で使うのと同じログロス）
- `perplexity` = `exp(平均 negative log likelihood)`

### 5.2 実装インターフェース

```python
# gblm_model/metrics.py
import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, list]


def compute_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    単純な分類精度を返す。

    Args:
        y_true: 正解ラベル (shape: (N,))
        y_pred: 予測ラベル (shape: (N,))

    Returns:
        accuracy (0.0〜1.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    return float((y_true == y_pred).mean())


def compute_multi_logloss(y_true: ArrayLike, proba: ArrayLike, eps: float = 1e-15) -> float:
    """
    multiclass の negative log-likelihood (logloss) を計算する。

    Args:
        y_true: 正解ラベル (shape: (N,))
        proba: 予測確率 (shape: (N, C))、各行はクラスごとの確率。
        eps: log の数値安定用の下限。

    Returns:
        平均 logloss。
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)

    N, C = proba.shape
    assert y_true.shape[0] == N

    # 正解クラスの確率を取り出す
    clipped = np.clip(proba, eps, 1.0)
    # advanced indexing
    p_true = clipped[np.arange(N), y_true]
    logloss = -np.log(p_true).mean()
    return float(logloss)


def compute_perplexity(y_true: ArrayLike, proba: ArrayLike, eps: float = 1e-15) -> float:
    """
    Perplexity = exp(平均 negative log-likelihood)
    logloss の値から計算する。
    """
    logloss = compute_multi_logloss(y_true, proba, eps=eps)
    return float(np.exp(logloss))
```

---

## 6. 推論 & テキスト生成ユーティリティ

### 6.1 目的

- 学習済み GBLM (LightGBM Booster) と Tokenizer を使って：
  - **次トークン予測**（1ステップ）
  - **テキスト生成ループ**（max_new_tokens まで）

を行う。

### 6.2 API 設計

```python
# gblm_model/inference.py
from pathlib import Path
from typing import List, Literal, Optional
import numpy as np
import lightgbm as lgb

from gblm_data.tokenizer import Tokenizer
from gblm_model.train import load_tokenizer  # 再利用


SamplingMethod = Literal["greedy", "top_k"]


def load_booster(model_path: Path) -> lgb.Booster:
    """
    保存された LightGBM モデルファイルから Booster を復元する。
    """
    booster = lgb.Booster(model_file=str(model_path))
    return booster


def prepare_context_ids(
    token_ids: List[int],
    context_length: int,
    pad_id: int,
) -> np.ndarray:
    """
    直近 context_length トークンを切り出し、左パディングした 1 行の ndarray を返す。

    Args:
        token_ids: 現在までのトークン ID 列。
        context_length: コンテキスト長 L。
        pad_id: PAD トークン ID。

    Returns:
        X: shape (1, L) の int32 ndarray。
    """
    context = token_ids[-context_length:]
    if len(context) < context_length:
        context = [pad_id] * (context_length - len(context)) + context
    X = np.asarray(context, dtype=np.int32).reshape(1, -1)
    return X


def predict_next_token_proba(
    booster: lgb.Booster,
    tokenizer: Tokenizer,
    token_ids: List[int],
    context_length: int,
) -> np.ndarray:
    """
    現在の token_ids から次トークンの確率分布を予測する。

    Returns:
        proba: shape (V,) の ndarray。各要素はクラス（トークン ID）の確率。
    """
    X = prepare_context_ids(
        token_ids=token_ids,
        context_length=context_length,
        pad_id=tokenizer.pad_id,
    )
    proba = booster.predict(X, num_iteration=booster.best_iteration)[0]
    # safety: 正規化
    proba = np.asarray(proba, dtype=np.float64)
    s = proba.sum()
    if s <= 0:
        proba = np.ones_like(proba) / len(proba)
    else:
        proba = proba / s
    return proba


def sample_from_proba_greedy(proba: np.ndarray) -> int:
    """
    greedy で最大確率のクラス ID を返す。
    """
    return int(np.argmax(proba))


def sample_from_proba_top_k(
    proba: np.ndarray,
    k: int = 10,
) -> int:
    """
    top-k サンプリング。
      - 確率上位 k クラスに絞る
      - その範囲で再正規化してサンプリング

    Args:
        proba: shape (V,) の確率分布。
        k: 上位何クラスに絞るか。

    Returns:
        サンプリングされたトークン ID。
    """
    V = proba.shape[0]
    k = min(k, V)
    # 上位 k 個のインデックス
    topk_idx = np.argpartition(-proba, k - 1)[:k]
    topk_proba = proba[topk_idx]
    topk_proba = topk_proba / topk_proba.sum()
    sampled_idx = np.random.choice(topk_idx, p=topk_proba)
    return int(sampled_idx)


def generate_text(
    booster: lgb.Booster,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    max_new_tokens: int = 50,
    sampling: SamplingMethod = "greedy",
    top_k: int = 10,
    stop_at_eos: bool = True,
) -> str:
    """
    学習済み GBLM からテキストを生成する。

    Args:
        booster: 学習済み LightGBM Booster。
        tokenizer: Tokenizer。
        prompt: 生成の起点となるテキスト。
        context_length: コンテキスト長 L。
        max_new_tokens: 生成する最大トークン数。
        sampling: "greedy" or "top_k"。
        top_k: sampling="top_k" の場合の k。
        stop_at_eos: True の場合、EOS トークンが出た時点で生成を終了。

    Returns:
        生成テキスト (prompt を含めた全体)。
    """
    # 初期トークン列
    # BOS は「文頭マーカー」として先頭に付与
    token_ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_bos_eos=False)

    for _ in range(max_new_tokens):
        proba = predict_next_token_proba(
            booster=booster,
            tokenizer=tokenizer,
            token_ids=token_ids,
            context_length=context_length,
        )

        if sampling == "greedy":
            next_id = sample_from_proba_greedy(proba)
        elif sampling == "top_k":
            next_id = sample_from_proba_top_k(proba, k=top_k)
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")

        token_ids.append(next_id)

        if stop_at_eos and next_id == tokenizer.eos_id:
            break

    # BOS/EOS をスキップして decode
    # Tokenizer.decode(skip_special=True) 前提
    return tokenizer.decode(token_ids, skip_special=True)
```

---

## 7. CLI スクリプト設計

### 7.1 `scripts/train_gblm.py`

目的:

- `GBLMTrainConfig` を定義 / 読み込み
- `train_gblm()` を実行
- 評価指標を print / JSON 保存

疑似コード:

```python
# scripts/train_gblm.py
from pathlib import Path
import json

from gblm_model.config import (
    GBLMTrainConfig,
    PathsConfig,
    TrainSplitConfig,
    LightGBMConfig,
)
from gblm_model.train import train_gblm


def main():
    # 1. Config をここでハードコード or 別途 YAML から読み込む
    cfg = GBLMTrainConfig(
        paths=PathsConfig(
            artifacts_dir=Path("artifacts"),
            tokenizer_json="tokenizer.json",
            data_npz="gblm_data.npz",
            model_file="gblm_model.txt",
        ),
        split=TrainSplitConfig(
            valid_size=0.1,
            shuffle=True,
            random_seed=42,
        ),
        lgbm=LightGBMConfig(
            # 必要に応じて上書き
            learning_rate=0.1,
            num_leaves=64,
            num_boost_round=500,
            early_stopping_rounds=20,
        ),
    )

    booster, metrics = train_gblm(cfg)

    # 評価結果の表示
    print("=== Training finished ===")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}: {json.dumps(v, indent=2)}")
        else:
            print(f"{k}: {v}")

    # metrics を artifacts に JSON で保存
    artifacts_dir = cfg.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "gblm_train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
```

### 7.2 `scripts/sample_gblm.py`

目的:

- 学習済みモデル & tokenizer をロード
- プロンプト文字列からテキスト生成
- 結果を print

疑似コード:

```python
# scripts/sample_gblm.py
from pathlib import Path
import argparse

from gblm_model.config import PathsConfig
from gblm_model.inference import (
    load_booster,
    generate_text,
)
from gblm_model.train import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--sampling", type=str, default="greedy", choices=["greedy", "top_k"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    paths = PathsConfig(
        artifacts_dir=Path("artifacts"),
        tokenizer_json="tokenizer.json",
        data_npz="gblm_data.npz",
        model_file="gblm_model.txt",
    )

    artifacts_dir = paths.artifacts_dir
    tokenizer_path = artifacts_dir / paths.tokenizer_json
    model_path = artifacts_dir / paths.model_file

    tokenizer = load_tokenizer(tokenizer_path)
    booster = load_booster(model_path)

    text = generate_text(
        booster=booster,
        tokenizer=tokenizer,
        prompt=args.prompt,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        sampling=args.sampling,  # type: ignore
        top_k=args.top_k,
        stop_at_eos=True,
    )

    print("=== Prompt ===")
    print(args.prompt)
    print("
=== Generated ===")
    print(text)


if __name__ == "__main__":
    main()
```

---

## 8. トップレベル使用例（end-to-end）

### 8.1 手順の流れ

1. **Children Stories コーパス CSV** をプロジェクト直下に配置  
   例: `children-stories-text-corpus.csv`

2. **語彙 & tokenizer 生成**
   ```bash
   python scripts/build_vocab.py
   ```
   → `artifacts/vocab.json`, `artifacts/tokenizer.json`

3. **GBLM 用データセット生成**
   ```bash
   python scripts/make_dataset.py
   ```
   → `artifacts/gblm_data.npz` (X, y)

4. **LightGBM による GBLM 学習**
   ```bash
   python scripts/train_gblm.py
   ```
   → `artifacts/gblm_model.txt`  
   → `artifacts/gblm_train_metrics.json`

5. **テキスト生成テスト**
   ```bash
   python scripts/sample_gblm.py --prompt "Once upon a time" --context-length 16 --max-new-tokens 50
   ```

---

## 9. まとめ

この設計に従って実装すれば、

- Kaggle Children Stories コーパスをベースに
- 独自 vocab / tokenizer で子ども向け英語レンジの言語空間を構築し
- LightGBM の multiclass で **Gradient Boosting Language Model (GBLM)** を学習し
- シンプルなテキスト生成まで動かせる

という **end-to-end の vanilla GBLM パイプライン** が完成する。

Optuna 等の高度なハイパラ探索は載せていないので、  
まずはこの vanilla 版で挙動とノリを掴みつつ、  
精度や挙動を見ながら徐々に拡張（特徴量追加・パラメータ調整・モデル置き換え）していくことを想定している。
