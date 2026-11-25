"""The bulk-ingest command line tool (python -m app.ingest <folder>)."""
import pytest

from app.ingest import ingest_folder, main
from tests.helpers import OIL_DOC, PUMP_DOC, make_pipeline


def fill_folder(folder):
    folder.mkdir(exist_ok=True)
    (folder / "oil.txt").write_bytes(OIL_DOC)
    (folder / "pump.md").write_bytes(PUMP_DOC)
    return folder


def test_every_supported_file_is_indexed_and_other_files_are_ignored(tmp_path):
    folder = fill_folder(tmp_path / "docs")
    (folder / "picture.png").write_bytes(b"\x89PNG")
    (folder / "nested").mkdir()  # sub-folders are not entered
    docs, errors = ingest_folder(make_pipeline(), folder)
    assert sorted(doc.filename for doc in docs) == ["oil.txt", "pump.md"]
    assert errors == []


def test_a_bad_file_is_reported_but_does_not_stop_the_others(tmp_path):
    folder = fill_folder(tmp_path / "docs")
    (folder / "broken.pdf").write_bytes(b"not a pdf")
    docs, errors = ingest_folder(make_pipeline(), folder)
    assert len(docs) == 2
    assert len(errors) == 1
    assert errors[0].startswith("broken.pdf")


@pytest.fixture()
def offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCQA_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("DOCQA_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "index"))
    return tmp_path


def test_the_command_indexes_a_folder_into_the_persisted_index_and_can_be_repeated(offline_env, capsys):
    folder = fill_folder(offline_env / "docs")

    assert main([str(folder)]) == 0
    first = capsys.readouterr().out
    assert "2 file(s) indexed" in first
    assert (offline_env / "index" / "meta.json").exists()

    assert main([str(folder)]) == 0  # second run: already indexed, nothing is duplicated
    assert "The index now holds 2 document(s)" in capsys.readouterr().out


def test_the_command_exits_with_1_when_a_file_could_not_be_read(offline_env, capsys):
    folder = fill_folder(offline_env / "docs")
    (folder / "broken.pdf").write_bytes(b"not a pdf")
    assert main([str(folder)]) == 1
    captured = capsys.readouterr()
    assert "skipped - broken.pdf" in captured.err
    assert "2 file(s) indexed" in captured.out


def test_a_path_that_is_not_a_folder_is_a_usage_error(offline_env):
    with pytest.raises(SystemExit) as caught:
        main([str(offline_env / "missing")])
    assert caught.value.code == 2
