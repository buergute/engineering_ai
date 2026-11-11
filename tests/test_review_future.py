import pytest
from common.schema import FutureDataError, ToolResult
from tools.review_context import review_context


def test_review_rejects_future_data():
    data = ToolResult(tool='read_asof', status='ok', as_of='2023-01-01T00:00',
                      result={'time': ['2023-01-01T00:10']})
    with pytest.raises(FutureDataError):
        review_context(data, 'T01')
