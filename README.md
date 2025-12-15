## Fusion 360 台灣山脈南北向剖面路徑產生器（CSV → DXF/SVG）

你提供一條「南北向路徑」的取樣點（含海拔），本工具會把它轉成 **剖面折線**：

- **X**：沿線累積距離
- **Y**：海拔（可自動以最低點為 0，或自行指定基準海拔）

輸出成 **DXF**（建議）或 **SVG**，即可在 Autodesk Fusion 360 匯入草圖後建模。

### 1) 安裝

```bash
pip install -r requirements.txt
```

> 若你不想裝任何套件也可以：DXF 仍可輸出（會使用極簡 R12 格式），但我仍建議安裝 `ezdxf` 讓相容性更穩。

### 2) 準備輸入 CSV

支援兩種格式（二擇一）：

#### A. 用經緯度+海拔（工具會自動算距離）

欄位：

- `lat`
- `lon`
- `ele_m`

專案內附：`profile_example_latlon.csv`

#### B. 你自己先算好距離+海拔

欄位：

- `dist_m`（從起點累積距離，單位 m）
- `ele_m`（海拔，單位 m）

### 2-1) 你沒有任何資料？用 Google Earth 畫「中央山脈主稜線」輸出 KML（最推薦）

你只需要把主稜線畫成一條 Path，存成 `.kml`，工具會自動用 SRTM 抽樣海拔並輸出 DXF：

1. 安裝並開啟 **Google Earth Pro**（桌面版）
2. 搜尋並移動到台灣，切到衛星/地形視圖
3. 點「新增」→「路徑（Path）」並沿著中央山脈主稜線逐點點選
4. 在「地點」清單中右鍵該路徑 →「另存為…」→ 存成 `your_ridge.kml`

### 3) 產生 DXF / SVG

### 3-1) 互動式 GUI（邊調參數邊預覽，mm 單位）

```bash
python gui.py
```

GUI 功能：

- 載入 `.kml` / `.csv`
- 以 **mm** 單位即時預覽剖面折線
- 可調：`fit-width / fit-height / vert-scale / smooth-window-m / simplify-epsilon / sample-step-m`
- **雙 Profile 預覽**：
  - Profile A：原始剖面（Normal）
  - Profile B：高度壓縮剖面（Compressed，讓高峰差距變小）
- **壓縮參數**：`gamma`（0~1）
  - `gamma = 1.0`：不壓縮（等同原始）
  - `gamma` 越小：高峰差距越小、低海拔相對被拉高（建議先試 0.6）
- 一鍵匯出 DXF / SVG

DXF 匯出會包含兩條線（不同 Layer）：

- `PROFILE_A_NORMAL`
- `PROFILE_B_COMPRESSED`

### 3-2) GUI 分段匯出（切 N 段 + 緩衝 M mm，避免硬切）

GUI 的「分段匯出」Tab 會把整體剖面（`fit-width × fit-height`，例如 240×80mm）沿 X 方向切成 **N 段**：

- **每段核心寬度**：\(fit\_width / N\)
- **緩衝距離 M（mm）**：每段左右各加上 M mm，並可選擇「端點平滑」讓緩衝區的高度逐漸衰減到基準線（看起來不會像被直接砍掉）

匯出時會產生 **N 個 DXF**（`segment_01.dxf`…`segment_NN.dxf`），每個檔案內依然是兩條 layer：

- `PROFILE_A_NORMAL`
- `PROFILE_B_COMPRESSED`

#### 產生 DXF（建議）

```bash
python main.py -i profile_example_latlon.csv -o profile.dxf --out-unit mm --vert-scale 1.0
```

#### 直接用 KML（只有經緯度）產生 DXF（會自動抓 SRTM 海拔）

```bash
python main.py -i your_ridge.kml -o profile.dxf --out-unit mm --vert-scale 1.0 --sample-step-m 200 --fit-width 240
```

### 設計上如何「放大起伏」與「省略無聊細節」

#### A) 放大山脈起伏（兩種做法）

- **做法 1：垂直誇張**（保留水平比例，只放大高度）

```bash
python main.py -i your_ridge.kml -o profile.dxf --out-unit mm --sample-step-m 200 --fit-width 240 --vert-scale 5
```

- **做法 2：指定輸出高度**（直接把 Y fit 到某個 mm，最直覺）

```bash
python main.py -i your_ridge.kml -o profile.dxf --out-unit mm --sample-step-m 200 --fit-width 240 --fit-height 80
```

> 小提醒：`--fit-box` 是「等比例縮放 X/Y」；如果你想要「寬度固定 240mm、但高度再放大」，請用 `--fit-width 240` 搭配 `--vert-scale` 或 `--fit-height`。

#### B) 省略無聊特徵（推薦流程：先平滑再簡化）

- **平滑（去掉小抖動）**：`--smooth-window-m` 建議先從 1000~3000m 試
- **簡化（刪掉小折線）**：`--simplify-epsilon` 單位是輸出單位（mm），建議先從 0.3~2.0mm 試

範例（常用組合）：

```bash
python main.py -i your_ridge.kml -o profile.dxf --out-unit mm --sample-step-m 200 --fit-width 240 --fit-height 80 --smooth-window-m 2000 --simplify-epsilon 1.0
```

#### 產生 SVG

```bash
python main.py -i profile_example_latlon.csv -o profile.svg --out-unit mm --vert-scale 1.0
```

### 4) 匯入 Fusion 360

- **DXF**：在草圖環境使用「Insert DXF」
- **SVG**：使用「Insert SVG」

匯入後你會得到一條折線（layer：`PROFILE`），可用來：

- 拉伸（Extrude）
- 掃掠（Sweep）
- 放樣（Loft）

### 5) 常用參數

- `--vert-scale`：垂直誇張倍率（例如 2.0 代表海拔放大 2 倍）
- `--horiz-scale`：水平縮放倍率（可用來調整匯入後大小）
- `--base-elevation-m`：指定剖面基準海拔；不填則會用最低點當 0
- `--simplify-epsilon`：折線簡化（RDP），單位是輸出單位（如 mm）
- `--sample-step-m`：KML/經緯度路徑在抽樣海拔前的加密步距（m）；數值越小剖面越細，但點數越多

---

### 我需要你先回答的 4 個關鍵問題（確定「台灣山脈南北向剖面」的定義）

1. 你說的「南北向剖面」是 **沿著中央山脈主稜線走一條路徑**，還是 **固定某個經度做南北剖切**？
2. 你希望剖面 **總長大約多少 km**，以及要不要含北端/南端的平原段？
3. 你要的輸入資料你比較方便提供哪一種？
   - A) GPX/CSV（有經緯度，海拔你也能提供）
   - B) 只有經緯度（需要我幫你接 DEM 自動取樣海拔）
4. 你希望 Fusion 裡的單位是 **mm / cm / m**？垂直誇張倍率要不要預設例如 **5 倍**？


