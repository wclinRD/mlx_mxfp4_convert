# conv.py — MLX 模型重新量化（6-bit → 4-bit），vision 不受影響

`conv.py` 直接在 safetensors 層把一個**已經量化過的 MLX 模型**重新量化成更低位元，
例如 6-bit → 4-bit，同時**完整保留 vision tower 原樣**（同 dtype、同 bytes），
所以圖像／影片輸入的行為不變。

```
python conv.py LLM-6BIT LLM-4BIT
```

## 為什麼不會弄壞 vision

MLX 量化模型對每個量化過的 module 存三個 tensor：

| tensor | dtype | 意義 |
|---|---|---|
| `<module>.weight` | uint32 | 打包後的整數 |
| `<module>.scales` | bf16 | 每 group 一個 scale |
| `<module>.biases` | bf16 | 每 group 一個 bias |

`config.json` 的 `quantization` 欄位記錄每個 module 的 `bits` / `group_size`。

轉換流程是逐 module：

1. 用**它自己的**原始 `(bits, group_size)` 反量化回 bf16；
2. 用目標 `(bits, group_size)` 重新量化；
3. 沒有被量化的 tensor（**整個 `vision_tower`**、norm、conv1d、A_log …）原樣複製。

所以 vision 部分是 byte-level 相同的（轉換後會列出 copy 的 tensor 數量，測試也會驗證）。

## 用法

```bash
# 最常用：6-bit -> 4-bit (group size 64)
python3 conv.py LLM-6BIT LLM-4BIT

# 指定目標精度
python3 conv.py SRC DST --bits 4 --group-size 32

# 只看計畫，不動檔案
python3 conv.py SRC DST --dry-run

# 連 vision tower 也要量化（預設不建議；預設是原樣保留）
python3 conv.py SRC DST --vision-bits 8

# 目標資料夾已存在時覆蓋
python3 conv.py SRC DST --overwrite
```

`SRC` 需要是含 `config.json` 的資料夾，權重可以是單檔 `model.safetensors`
或分片 + `model.safetensors.index.json`。`DST` 不存在時會自動建立。

### 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--bits` | `4` | 目標位元數 |
| `--group-size` | `64` | 目標 group size；若某個 shape 不能整除會自動退到 64/32/16 |
| `--vision-bits` | 不啟用 | 指定時連 vision tower 一起量化 |
| `--shard-size` | `4.5G` | 輸出單檔上限（超過就分片並寫 index） |
| `--verify-sample` | `24` | 抽樣多少個 module 做數值比對（0 = 關閉） |
| `--cache-limit` | `1.5G` | MLX cache 上限，控制記憶體峰值 |
| `--overwrite` | 關 | 覆蓋已存在的 DST |
| `--dry-run` | 關 | 只印計畫 |

輸出資料夾會是可直接被 LM Studio / mlx-lm 讀取的完整模型：權重 + 更新過的
`config.json`（`quantization` 與 `quantization_config` 同步改寫）+ 原樣複製的
tokenizer、chat template、`preprocessor_config.json` 等。

## 範例（本次實際轉換）

```bash
python3 conv.py \
  ~/.lmstudio/models/nightmedia/Qwen3.6-35B-A3B-Continuum-qx64-hi-mlx \
  ~/.lmstudio/models/nightmedia/Qwen3.6-35B-A3B-Continuum-qx64-hi-mlx-4bit
```

- 512 個量化 module：422 個 `6b/g32 → 4b/g64`、90 個本來就是 `4b/g64`（原樣）
- 28.8 GiB → 19.0 GiB（約 0.66 倍）
- vision tower 的 333 個 tensor 原樣複製

轉完後 LM Studio 直接載入即可（重新掃描 models 資料夾）。

## 測試

### 1. 轉換邏輯（不需要真模型）

```bash
python3 tests/test_conv.py
```

在暫存資料夾建立一個小型假模型（含 MoE 3D tensor、已 4-bit 的 module、
bf16 vision tower、多分片 index），驗證：

- tensor 名稱集合、dtype、shape 全部正確；
- 抽樣 module 的反量化誤差在合理範圍；
- 未被量化的 tensor（含 vision tower）逐 byte 相同；
- 多分片 → 多分片、index 正確、拒絕覆寫、其他 bits 目標。

### 2. 真實模型的 vision 檢查（需要 LM Studio）

```bash
lms load <model-id> --gpu max -c 8192 -y
python3 tests/vision_check.py <model-id>
```

`tests/fixtures/test-vision.png` 是 ground truth 已知的圖（左上紅圓 7、
右上藍方 4、下方綠三角 2），問 12 題顏色／形狀／數字／位置。

## 驗證結果（本次實測，2026-10-26）

| 檢查項目 | 結果 |
|---|---|
| 轉換耗時 | 17.4 秒（6 個 shard → 5 個 shard） |
| 大小 | 28.8 GiB → 19.0 GiB（LM Studio 顯示 30.94 GB → 20.42 GB） |
| tensor 完整性 | 2090 → 2090，名稱集合完全相同、無缺漏 |
| config 一致性 | 每個 module 的 packed shape / scales shape 與 `quantization` 對得上 |
| 數值 | 無 NaN/Inf；抽樣 25 個 module 的二次量化誤差 mean rel-rms 10.7% |
| **vision tensor** | 333 個 `vision_tower.*` 逐 byte 相同（原 dtype、原 bytes） |
| LM Studio 載入 | `qwen3.6-35b-a3b-continuum-qx64-hi-mlx@4bit`，qwen3_5_moe，19.02 GiB，8.9 秒載入 |
| 圖像輸入 | 送 1024×1024 圖時 prompt 實際吃進 ~1058 tokens（含 ~1024 個影像 token）→ vision 路徑正常 |
| 文字品質（4-bit vs 6-bit） | 一致：bat/ball `$0.05`、Canberra、Fibonacci 迭代版、2027 是質數 |
| vision 分數（12 題，同一張圖） | 6-bit 4/12、4-bit 5/12（此 fine-tune 本身視覺就弱；對照組 Qwen3.8-MindMeld 12/12） |

結論：轉換本身沒有動到 vision 權重，圖像 token 正常進入模型；
這個 "Continuum" fine-tune 的視覺理解在 6-bit 時就已經很弱，
4-bit 與 6-bit 的表現沒有可測差異。若你要更保守的精度，可用
`--group-size 32`（誤差更小、體積略大）或 `--bits 5`。

## 限制

- 只處理**已經量化**的 MLX 模型（有 `.scales`/`.biases` 的 module）。純 bf16 的
  MLX 模型不會被量化（因為需要模型結構才能判斷哪些 layer 可以量化）。
- 量化誤差是「先反量化、再量化」的二次量化誤差，4-bit 通常比原 6-bit 模型明顯
  掉一點品質；要更保守可用 `--bits 5` 或 `--bits 6 --group-size 32`。
- 需要 `mlx`（`python3 -c "import mlx.core"` 可確認）。
