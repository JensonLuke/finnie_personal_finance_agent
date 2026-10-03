import pytest

from finnie.main import main


def test_main_prints_usage_and_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "streamlit run" in captured.out
