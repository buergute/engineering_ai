from pathlib import Path

import numpy as np
import pandas as pd

# 优先使用指定文件；若不存在，则兼容当前目录中的 Original(1).xlsx
source = Path("data/Original.xlsx")
if not source.exists():
    fallback = Path("data/Original(1).xlsx")
    if fallback.exists():
        source = fallback
    else:
        raise FileNotFoundError(
            f"找不到源文件：{source} 或 data/Original(1).xlsx"
        )

target = Path("data/Processed.xlsx")
target.parent.mkdir(parents=True, exist_ok=True)

# 默认将第一行作为表头
df = pd.read_excel(source, engine="openpyxl")

# 根据关键词定位列并重命名；每个关键词必须唯一匹配一列
keywords = {
    "time": "time",
    "三峡": "sanxia",
    "高坝洲": "gaobazhou",
    "沙市": "shashi",
}

rename = {}
for keyword, new_name in keywords.items():
    matched_columns = [
        column
        for column in df.columns
        if keyword.casefold() in str(column).casefold()
    ]

    if len(matched_columns) != 1:
        raise ValueError(
            f"关键词 {keyword!r} 应匹配唯一一列，实际匹配到：{matched_columns}"
        )

    rename[matched_columns[0]] = new_name

df = df.rename(columns=rename)

# 将小于 0 或大于 70000 的数值替换为 NaN
# 原有空值不计入异常值数量，也不会被填补或删除
replacement_counts = {}

for column in ("sanxia", "gaobazhou"):
    numeric_values = pd.to_numeric(df[column], errors="coerce")
    outlier_mask = numeric_values.notna() & (
        (numeric_values < 0) | (numeric_values > 70000)
    )

    replacement_counts[column] = int(outlier_mask.sum())
    df.loc[outlier_mask, column] = np.nan

# 保存时不写入行索引
df.to_excel(target, index=False, engine="openpyxl")

print(f"已保存：{target}")
print(f"sanxia 替换异常值数量：{replacement_counts['sanxia']}")
print(f"gaobazhou 替换异常值数量：{replacement_counts['gaobazhou']}")