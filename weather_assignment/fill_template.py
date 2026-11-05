"""Fill only column B values in the supplied competition template."""

from pathlib import Path
import re
import zipfile

import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/burgut/Downloads/2026-10-10风速预测提交模板.xlsx')
FORECAST = ROOT / 'output/forecast_2026-10-10_CNC.csv'
DEST = ROOT / 'output/布尔古特_X.xlsx'


def main():
    f = pd.read_csv(FORECAST, parse_dates=['target_time'])
    expected = load_workbook(SOURCE, read_only=True, data_only=False).active
    template_times = [expected.cell(r, 1).value for r in range(2, 74)]
    assert len(f) == len(template_times) == 72
    assert list(f.target_time.dt.to_pydatetime()) == template_times
    values = [round(float(v), 2) for v in f.forecast_knots]
    with zipfile.ZipFile(SOURCE) as source, zipfile.ZipFile(DEST, 'w') as dest:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == 'xl/worksheets/sheet1.xml':
                xml = data.decode('utf-8')
                for row, value in enumerate(values, 2):
                    pattern = rf'(<c r="B{row}"[^>]*><v>)[^<]*(</v></c>)'
                    xml, n = re.subn(pattern, rf'\g<1>{value:.2f}\g<2>', xml)
                    if n != 1:
                        raise ValueError(f'Cannot uniquely fill B{row}')
                data = xml.encode('utf-8')
            dest.writestr(item, data)
    final = load_workbook(DEST, read_only=True, data_only=False).active
    for row in range(2, 74):
        assert final.cell(row, 1).value == expected.cell(row, 1).value
        assert final.cell(row, 1).number_format == expected.cell(row, 1).number_format
        assert final.cell(row, 2).number_format == expected.cell(row, 2).number_format
        assert final.cell(row, 2).value == values[row - 2]
    print(DEST)


if __name__ == '__main__':
    main()
