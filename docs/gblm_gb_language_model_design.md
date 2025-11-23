# GBLM 用データ生成パイプライン設計書  
Gradient Boosting Language Model (GBLM) – Kaggle Children Stories コーパス向け

---

## 0. 目的

- Kaggle の Children Stories コーパスを入力として、
  - 語彙リスト（vocab）
  - トークナイザ
  - GBLM（LightGBM）学習用の `(X, y)` データセット
- を **自動生成する Python ライブラリ & スクリプト** を実装できるようにする。

ここでは **API・型・処理フローまで** を明確に定義し、  
この設計をそのまま投げればコード生成エージェントが実装を作り切れる状態を目指す。

---

## 1. 前提 & 想定

### 1.1 使用コーパス

- Kaggle: Children Stories Text Corpus（例: `children-stories-text-corpus.csv`）
- 想定フォーマット（一例）:
  - CSV ファイル
  - 少なくとも `text` カラムに物語テキストが入っている

**前提:**  
- 行ごとに 1 ストーリー（ドキュメント）として扱う
- テキストは英語

### 1.2 目標とするモデル I/O（GBLM）

- 語彙サイズ = `V`
- コンテキスト長 = `L`（ハイパラ）
- LightGBM の multiclass を前提とする

学習時に最終的に欲しいデータ形式は:

- `X`: shape = `(N, L)` の `int32` 行列
  - 各行は「直前 L トークンの ID 列」
- `y`: shape = `(N,)` の `int32` ベクトル
  - 各要素は「次トークン ID」
- LightGBM 側の設定:
  - `objective = "multiclass"`
  - `num_class = V`
  - `categorical_feature = list(range(L))`（全列カテゴリー扱い）

---

## 2. 全体アーキテクチャ

### 2.1 ディレクトリ構成（案）

```text
gblm_data/
  __init__.py
  config.py        # 全体設定 (dataclass)
  corpus.py        # Kaggle コーパス読み込み
  vocab.py         # 語彙カウント & vocab 構築
  tokenizer.py     # Tokenizer クラス
  dataset.py       # GBLM 学習用 X, y 生成

scripts/
  build_vocab.py   # コーパスから vocab/tokenizer を作成して保存
  make_dataset.py  # vocab/tokenizer を使って GBLM 用データセット生成

artifacts/         # 出力例
  vocab.json
  tokenizer.json   # or tokenizer.pkl
  gblm_X.npz
  gblm_y.npz
```

---

## 3. 設定 (Config)

### 3.1 Config データクラス

```python
# gblm_data/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class VocabConfig:
    min_freq: int = 5          # vocab に含める最小出現回数
    top_k: Optional[int] = 5000  # 出現頻度上位 top_k に制限（None なら制限なし）
    lowercase: bool = True     # トークン化前に小文字化するか
    max_docs: Optional[int] = None  # 読み込む最大ドキュメント数 (None=全部)


@dataclass
class DatasetConfig:
    context_length: int = 16   # コンテキスト長 L
    max_samples: Optional[int] = None  # サンプル数上限 (None=全部)
    shuffle: bool = True
    random_seed: int = 42


@dataclass
class PathsConfig:
    corpus_csv: Path           # Kaggle コーパス CSV path
    text_column: str = "text"  # テキスト列名
    artifacts_dir: Path = Path("artifacts")


@dataclass
class GBLMConfig:
    vocab: VocabConfig
    dataset: DatasetConfig
    paths: PathsConfig
```

---

## 4. コーパス読み込みモジュール

### 4.1 関数仕様

```python
# gblm_data/corpus.py
from typing import List
from pathlib import Path

def load_corpus_texts(
    csv_path: Path,
    text_column: str = "text",
    max_docs: int | None = None,
) -> List[str]:
    """
    Kaggle Children Stories コーパスの CSV からテキスト列のみを読み込む。

    Args:
        csv_path: CSV ファイルのパス。
        text_column: テキストが入っているカラム名。
        max_docs: 読み込む最大行数。None の場合は全て読み込む。

    Returns:
        texts: 各行のテキストを格納した list[str]。
    """
    ...
```

- 実装は pandas を利用して OK (`pd.read_csv`)

---

## 5. トークナイズ & 語彙構築

### 5.1 単純トークナイザ（語彙構築用）

正規表現ベースの簡易 word-level tokenizer:

- 小文字化（`lower()`）オプションあり
- `a-z`, `0-9`, `'` 以外はスペースに置換
- `.split()` でトークン列に変換

```python
# gblm_data/vocab.py
import re
from collections import Counter
from typing import Iterable, List

WORD_REGEX = re.compile(r"[^a-z0-9']+")

def simple_word_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    子ども向け英語テキスト用のシンプルな word-level トークナイザ。

    仕様:
      - lowercase=True の場合、英字をすべて小文字化
      - [^a-z0-9'] をスペースに置換
      - .split() でトークンに分割

    Args:
        text: 入力テキスト。
        lowercase: True の場合テキストを小文字化。

    Returns:
        tokens: トークン文字列のリスト。
    """
    if lowercase:
        text = text.lower()
    text = WORD_REGEX.sub(" ", text)
    tokens = text.split()
    return tokens
```

### 5.2 語彙カウント

```python
def count_vocab(
    texts: Iterable[str],
    lowercase: bool = True,
) -> Counter:
    """
    テキスト列から単語出現頻度をカウントする。

    Args:
        texts: テキストのイテラブル。
        lowercase: simple_word_tokenize に渡すフラグ。

    Returns:
        counter: collections.Counter。key=単語(str), value=出現回数。
    """
    ...
```

### 5.3 語彙リスト構築

```python
from typing import List, Tuple

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

def build_vocab(
    counter: Counter,
    min_freq: int = 5,
    top_k: int | None = 5000,
) -> List[str]:
    """
    出現頻度カウンタから vocab を構築する。

    仕様:
      - min_freq 未満の単語は除外。
      - 残った単語を出現頻度の降順でソートする。
        - 同頻度なら語彙の lexicographical order で安定。
      - top_k が指定されていれば、上位 top_k のみ残す。
      - SPECIAL_TOKENS は含めず、「通常トークン」のリストを返す。

    Args:
        counter: 単語 -> 出現回数 の Counter。
        min_freq: vocab に含める最小出現回数。
        top_k: 残す単語数の上限。None の場合は制限なし。

    Returns:
        vocab_tokens: SPECIAL_TOKENS を除いた通常トークンのリスト。
    """
    ...
```

---

## 6. Tokenizer クラス

### 6.1 Tokenizer 仕様

- 単語ベースの tokenizer（今は BPE ではなく word-level）
- 内部に以下を保持:
  - `itos: list[str]`  … id から token へのマップ
  - `stoi: dict[str, int]` … token から id
  - special token の ID
    - `pad_id = 0`
    - `unk_id = 1`
    - `bos_id = 2`
    - `eos_id = 3`
  - `lowercase: bool`

### 6.2 クラス定義（インターフェース）

```python
# gblm_data/tokenizer.py
from dataclasses import dataclass
from typing import List, Dict

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

@dataclass
class Tokenizer:
    itos: List[str]           # index to string
    stoi: Dict[str, int]      # string to index
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    lowercase: bool = True

    @classmethod
    def from_vocab(
        cls,
        vocab_tokens: List[str],
        lowercase: bool = True,
    ) -> "Tokenizer":
        """
        SPECIAL_TOKENS + vocab_tokens から Tokenizer を構築。

        itos: [<PAD>, <UNK>, <BOS>, <EOS>, ...vocab_tokens]
        stoi: 上記に対応する dict
        """
        ...

    def tokenize(self, text: str) -> List[str]:
        """
        simple_word_tokenize と同等の処理で text を token 文字列列にする。
        lowercase フラグは self.lowercase を使用。
        """
        ...

    def encode(self, text: str, add_bos_eos: bool = False) -> List[int]:
        """
        text を token ID 列にエンコードする。

        Args:
            text: 入力テキスト。
            add_bos_eos: True の場合、先頭に BOS, 末尾に EOS を付与。

        Returns:
            token_ids: ID リスト。
        """
        ...

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        token ID 列を text にデコードする。

        Args:
            ids: トークン ID リスト。
            skip_special: True の場合、SPECIAL_TOKENS はスキップ。

        Returns:
            復元テキスト。
        """
        ...
```

### 6.3 永続化インターフェース

```python
    def to_dict(self) -> dict:
        """
        JSON シリアライズ可能な dict に変換する。
        """
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "Tokenizer":
        """
        to_dict で保存した dict から Tokenizer を復元する。
        """
        ...
```

`scripts/build_vocab.py` で `tokenizer.to_dict()` を JSON 保存する想定。

---

## 7. GBLM 学習用データセット生成

### 7.1 サンプル生成ロジック（仕様）

1. 各ドキュメント `text` に対して:
   - `token_ids = [bos_id] + tokenizer.encode(text) + [eos_id]`  
     （`encode` に `add_bos_eos=False` として明示的に付与してもよい）

2. `token_ids` の長さを `T` とする。

3. 位置 `i = 1 .. T-1` について以下を 1 サンプルとする：
   - `y = token_ids[i]` … 次トークンID
   - コンテキスト（長さ L）：
     - 生のコンテキスト:
       - `context_raw = token_ids[max(0, i - L) : i]`
     - 左側を `pad_id` でパディングして長さを L に揃える:
       - `context = [pad_id] * (L - len(context_raw)) + context_raw`

4. すべてのドキュメント・すべての位置から作成された `(context, y)` を結合する。

5. `max_samples` が指定されていれば
   - サンプルを shuffle（`random_seed` を用いて再現性あり）
   - 先頭 `max_samples` 個のみを残す。

### 7.2 API 仕様

```python
# gblm_data/dataset.py
from typing import List, Tuple
import numpy as np

from .tokenizer import Tokenizer

def make_gblm_training_data(
    texts: List[str],
    tokenizer: Tokenizer,
    context_length: int,
    max_samples: int | None = None,
    shuffle: bool = True,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GBLM (LightGBM) 学習用の特徴量行列 X とラベル y を生成する。

    仕様:
      - 各テキストに対して:
          token_ids = [BOS] + encode(text) + [EOS]
        を構成する。
      - i = 1 .. len(token_ids)-1 でそれぞれ以下を生成:
          y = token_ids[i]
          context_raw = token_ids[max(0, i-L) : i]
          context = 左パディングして長さ L に揃える
      - 全テキスト分を結合したものが最終的な (X, y)。
      - max_samples が指定されていれば、シャッフル後に先頭 max_samples を使う。

    Args:
        texts: ドキュメントテキストのリスト。
        tokenizer: 事前に構築された Tokenizer。
        context_length: コンテキスト長 L。
        max_samples: 最大サンプル数（None の場合は制限なし）。
        shuffle: True の場合、サンプルをランダムシャッフルする。
        random_seed: シャッフル時の乱数シード。

    Returns:
        X: shape (N, L) の int32 ndarray。各行はコンテキストの token ID。
        y: shape (N,) の int32 ndarray。各要素は次トークン ID。
    """
    ...
```

### 7.3 LightGBM への渡し方（コメントレベル）

```python
import lightgbm as lgb

def to_lgb_dataset(
    X: np.ndarray,
    y: np.ndarray,
    categorical: bool = True,
) -> lgb.Dataset:
    """
    GBLM 用の X, y から LightGBM Dataset を作るヘルパー。

    categorical=True の場合、全列をカテゴリ列として扱う。
    """
    n_features = X.shape[1]
    cat_feats = list(range(n_features)) if categorical else []
    dtrain = lgb.Dataset(
        X, label=y,
        categorical_feature=cat_feats,
        free_raw_data=False,
    )
    return dtrain
```

---

## 8. CLI スクリプト設計

### 8.1 `scripts/build_vocab.py`

処理フロー:

1. `GBLMConfig` を読み込む（yaml or json or 直接コード内で定義）
2. `load_corpus_texts()` でテキスト読み込み
3. `count_vocab()` で Counter 作成
4. `build_vocab()` で vocab_tokens 作成
5. `Tokenizer.from_vocab()` で Tokenizer を作成
6. `vocab` / `tokenizer` を `artifacts_dir` に保存

疑似コード:

```python
def main():
    cfg = load_config_somehow()

    texts = load_corpus_texts(
        cfg.paths.corpus_csv,
        text_column=cfg.paths.text_column,
        max_docs=cfg.vocab.max_docs,
    )

    counter = count_vocab(texts, lowercase=cfg.vocab.lowercase)

    vocab_tokens = build_vocab(
        counter,
        min_freq=cfg.vocab.min_freq,
        top_k=cfg.vocab.top_k,
    )

    tokenizer = Tokenizer.from_vocab(
        vocab_tokens,
        lowercase=cfg.vocab.lowercase,
    )

    # 保存
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # vocab は単純な list[str] として保存
    save_json(cfg.paths.artifacts_dir / "vocab.json", vocab_tokens)

    # tokenizer は to_dict() を JSON として保存
    save_json(cfg.paths.artifacts_dir / "tokenizer.json", tokenizer.to_dict())
```

`save_json` は単純な `json.dump` ラッパ。

### 8.2 `scripts/make_dataset.py`

処理フロー:

1. `GBLMConfig` を読み込む
2. `tokenizer.json` を読み込んで `Tokenizer.from_dict()` 復元
3. コーパスを再度読み込み（or キャッシュした subset）
4. `make_gblm_training_data()` で `(X, y)` 生成
5. `X`, `y` を `np.savez` などで保存する

疑似コード:

```python
def main():
    cfg = load_config_somehow()

    # tokenizer 復元
    tok_data = load_json(cfg.paths.artifacts_dir / "tokenizer.json")
    tokenizer = Tokenizer.from_dict(tok_data)

    texts = load_corpus_texts(
        cfg.paths.corpus_csv,
        text_column=cfg.paths.text_column,
        max_docs=cfg.vocab.max_docs,
    )

    X, y = make_gblm_training_data(
        texts,
        tokenizer=tokenizer,
        context_length=cfg.dataset.context_length,
        max_samples=cfg.dataset.max_samples,
        shuffle=cfg.dataset.shuffle,
        random_seed=cfg.dataset.random_seed,
    )

    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    np.savez(cfg.paths.artifacts_dir / "gblm_data.npz", X=X, y=y)
```

---

## 9. 簡易使用例（トップレベル）

### 9.1 Python からの利用イメージ

```python
from pathlib import Path
from gblm_data.config import VocabConfig, DatasetConfig, PathsConfig, GBLMConfig
from gblm_data.corpus import load_corpus_texts
from gblm_data.vocab import count_vocab, build_vocab
from gblm_data.tokenizer import Tokenizer
from gblm_data.dataset import make_gblm_training_data

# 1. Config 定義
cfg = GBLMConfig(
    vocab=VocabConfig(min_freq=5, top_k=5000, lowercase=True, max_docs=10000),
    dataset=DatasetConfig(context_length=16, max_samples=200_000),
    paths=PathsConfig(
        corpus_csv=Path("children-stories-text-corpus.csv"),
        text_column="text",
        artifacts_dir=Path("artifacts"),
    ),
)

# 2. コーパス読み込み
texts = load_corpus_texts(cfg.paths.corpus_csv, cfg.paths.text_column, cfg.vocab.max_docs)

# 3. vocab & tokenizer
counter = count_vocab(texts, lowercase=cfg.vocab.lowercase)
vocab_tokens = build_vocab(counter, cfg.vocab.min_freq, cfg.vocab.top_k)
tokenizer = Tokenizer.from_vocab(vocab_tokens, lowercase=cfg.vocab.lowercase)

# 4. GBLM 用データセット生成
X, y = make_gblm_training_data(
    texts,
    tokenizer=tokenizer,
    context_length=cfg.dataset.context_length,
    max_samples=cfg.dataset.max_samples,
)

# 5. LightGBM に渡す
import lightgbm as lgb
from gblm_data.dataset import to_lgb_dataset

dtrain = to_lgb_dataset(X, y)
```

---

この設計のまま実装すれば、

- Kaggle Children Stories から
  - 子ども向け語彙に基づいた vocab / tokenizer
  - GBLM(LightGBM) 用の `(X, y)` 学習データ
- を自動生成できる。

この Markdown をそのままコード生成エージェントに渡して実装させれば、  
GBLM の「データ側」の土台はほぼ完成するはず。
