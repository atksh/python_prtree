# Float32 Precision Issue Analysis

## 報告された問題

float32への丸め込みで偽陰性（false negative）で重なりが検出されないという報告。

## 調査結果

### コードベースの分析

#### 1. 内部表現
- すべてのバウンディングボックスは内部的に`Real`型（=`float`）で保存される
- `include/prtree/core/detail/bounding_box.h:17`で定義: `using Real = float;`

#### 2. 精度補正メカニズム（float64入力時のみ）

**float64入力の場合** (`include/prtree/core/prtree.h:213-291`):
```cpp
// Constructor for float64 input (float32 tree + double refinement)
PRTree(const py::array_t<T> &idx, const py::array_t<double> &x)
```
- 内部的にfloat32に変換してツリーを構築
- しかし、元のdouble精度の座標を`idx2exact`に保存（line 274）
- クエリ時に`refine_candidates()`メソッド（line 805-831）でdouble精度で再チェック

**float32入力の場合** (`include/prtree/core/prtree.h:138-210`):
```cpp
// Constructor for float32 input (no refinement, pure float32 performance)
PRTree(const py::array_t<T> &idx, const py::array_t<float> &x)
```
- float32のまま処理
- `idx2exact`は空のまま（line 157のコメント: "idx2exact is NOT populated for float32 input"）
- **補正が行われない**

#### 3. 交差判定ロジック

`include/prtree/core/detail/bounding_box.h:106-125`:
```cpp
bool operator()(const BB &target) const {
    // ... (省略)
    for (int i = 0; i < D; ++i) {
        flags[i] = -minima[i] <= maxima[i];  // 閉区間セマンティクス
    }
    // ...
}
```

- すべてfloat32で計算される
- `<=`を使用（touching boxes are considered intersecting）

#### 4. query_intersections()メソッド

`include/prtree/core/prtree.h:894-1040`:
- line 963-965と993-996で、`idx2exact`が空でない場合のみ補正を行う
- **float32入力の場合は補正がスキップされる**（line 808-810）

### 理論的な脆弱性

1. **丸め込みによる精度損失**:
   - float64で表現された微小な重なりがfloat32に変換される際に失われる可能性
   - 特に大きな座標値（例: 10^7以上）では、float32のULP（Unit in Last Place）が大きくなる

2. **補正メカニズムの非対称性**:
   - float64入力: float32ツリー + double補正
   - float32入力: float32ツリーのみ（補正なし）
   - これにより、同じデータでも入力型によって結果が異なる可能性

3. **潜在的な偽陰性シナリオ**:
   - 2つのAABBが非常に小さな量で重なる
   - その重なりがfloat32の精度限界以下
   - float32に丸められた後、重ならなくなる

### テスト結果

複数のテストケースを作成して検証を試みましたが、実際の偽陰性を再現することはできませんでした：

1. **test_float32_overlap_issue.py**: 基本的な精度テスト
2. **test_float32_refined.py**: より厳密な精度境界テスト
3. **test_float32_extreme.py**: 極端なケース（ULP境界、負の座標、subnormal値など）

#### テスト結果の考察

すべてのテストで偽陰性は検出されませんでした。理由として考えられるのは：

1. **閉区間セマンティクス**: `<=`比較により、境界で接触するボックスは常に交差と判定される
2. **一貫した丸め込み**: 両方のボックスが同じ精度（float32）で保存されるため、比較は一貫している
3. **テストケースの限界**: 実際の報告された問題を再現する特定のデータセットが必要な可能性

## 問題の根本原因（理論的分析）

報告された偽陰性の問題は、以下の条件で発生する可能性があります：

### ケース1: 異なる精度での構築とクエリ

```python
# float32でツリーを構築
boxes_f32 = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
tree = PRTree2D(np.array([0]), boxes_f32)

# float64でクエリ（わずかに異なる境界値）
query_f64 = np.array([100.0 + epsilon, 0.0, 200.0, 100.0], dtype=np.float64)
result = tree.query(query_f64)  # 偽陰性の可能性
```

### ケース2: 大きな座標値での微小な重なり

float32の精度限界：
- 100付近: ULP ≈ 7.6e-6
- 10,000付近: ULP ≈ 0.000977
- 1,000,000付近: ULP ≈ 0.0625
- 16,777,216 (2^24)付近: ULP = 2.0

大きな座標値では、微小な重なりが丸め込みで失われやすい。

### ケース3: 累積丸め込みエラー

複数の演算を経た座標値は、累積的な丸め込みエラーにより、
元の値から大きくずれる可能性がある。

## 推奨される対策

1. **float64入力の使用を推奨**:
   - float64入力を使用すれば、内部補正メカニズムにより高精度が保たれる

2. **float32入力にも補正メカニズムを追加**:
   - float32で構築されたツリーでも、クエリ時にはdouble精度で再チェック
   - ただし、元の精度情報が失われているため完全な補正は不可能

3. **ドキュメントの改善**:
   - float32使用時の精度制限を明示的に文書化
   - 高精度が必要な場合はfloat64の使用を推奨

4. **精度警告の追加**:
   - 大きな座標値（>10^6）でfloat32を使用する場合、警告を表示

## 検証スクリプト

以下の3つのテストスクリプトを作成しました：

1. `test_float32_overlap_issue.py`: 基本的な偽陰性テスト
2. `test_float32_refined.py`: より厳密な精度境界テスト
3. `test_float32_extreme.py`: 極端なエッジケーステスト

実行方法:
```bash
python test_float32_overlap_issue.py
python test_float32_refined.py
python test_float32_extreme.py
```

## 結論

1. **コードレベルでの脆弱性確認**: float32入力時の補正メカニズムの欠如を確認
2. **実際の偽陰性の再現**: テストケースでは再現できず
3. **理論的なリスク**: 特定の条件下（大きな座標値、微小な重なり）で偽陰性が発生する可能性あり
4. **推奨事項**: 高精度が必要な場合はfloat64入力を使用すること

この問題を完全に解決するには、報告者から具体的なデータセットや再現手順の提供が必要です。
