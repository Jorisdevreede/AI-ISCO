import json

import pytest

from aiisco.jsonio import load_json, write_json


def test_write_then_load_round_trips_unicode(tmp_path):
    path = tmp_path / "out.json"
    write_json(path, {"title": "café", "n": 1})
    assert load_json(path) == {"title": "café", "n": 1}
    assert "café" in path.read_text(encoding="utf-8")


def test_write_json_honours_indent(tmp_path):
    path = tmp_path / "out.json"
    write_json(path, [1], indent=1)
    assert path.read_text() == json.dumps([1], indent=1)


def test_load_json_exits_when_file_is_missing(tmp_path, capsys):
    missing = tmp_path / "nope.json"
    with pytest.raises(SystemExit) as exit_info:
        load_json(missing)
    assert exit_info.value.code == 1
    assert f"ERROR: {missing} not found." in capsys.readouterr().out
