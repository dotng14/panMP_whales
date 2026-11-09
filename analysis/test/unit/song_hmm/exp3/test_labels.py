import json

import numpy as np
import pytest

from analysis.song_exp3.reference.labels import (
    load_all,
    load_clip,
    read_audacity,
    recording_theme,
)

EXPORT = (
    "0.000000\t0.000000\tS\n"
    "0.000000\t4.500000\tT1\n"
    "4.500000\t10.000000\tt2\n"
    "\\t100.0\t900.0\n"  # spectral-selection line, ignored
    "10.000000\t12.000000\tT3\n"
    "12.000000\t30.000000\tBG\n"
    "30.000000\t35.000000\tT6\n"
    "35.000000\t35.000000\tS\n"
    "35.000000\t40.000000\tT1\n"
)


@pytest.fixture
def files(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "clip1_labels.txt").write_text(EXPORT, encoding="utf-8")
    clips = tmp_path / "clips.json"
    clips.write_text(json.dumps({"clips": {
        "clip1": {"offset_s": 1000, "duration_s": 50},
        "clip2": {"offset_s": 2000, "duration_s": 50},
    }}), encoding="utf-8")
    return labels_dir, clips


def test_read_audacity_skips_frequency_lines(files):
    rows = read_audacity(files[0] / "clip1_labels.txt")
    assert len(rows) == 8
    assert rows[2] == (4.5, 10.0, "T2")


def test_load_clip(files):
    labels_dir, clips = files
    c = load_clip("clip1", labels_dir / "clip1_labels.txt", clips)
    assert c.seconds[0] == 1000 and c.seconds[-1] == 1049
    expected = np.full(50, -1)
    expected[0:4] = 0  # T1 0-4.5 s: midpoints 0.5..3.5 (4.5 is not < 4.5)
    expected[4:10] = 1
    expected[10:12] = 2
    expected[30:35] = 5
    expected[35:40] = 0
    np.testing.assert_array_equal(c.theme, expected)
    np.testing.assert_array_equal(c.song_starts, [1000.0, 1035.0])
    np.testing.assert_array_equal(c.bg_spans, [[1012.0, 1030.0]])
    assert c.theme_spans.shape == (5, 3)
    np.testing.assert_array_equal(c.theme_spans[0], [1000.0, 1004.5, 0])


def test_unknown_label_raises(tmp_path, files):
    _, clips = files
    bad = tmp_path / "bad.txt"
    bad.write_text("1\t2\tT7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        load_clip("clip1", bad, clips)


def test_point_theme_raises(tmp_path, files):
    _, clips = files
    bad = tmp_path / "bad.txt"
    bad.write_text("1\t1\tT2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no length"):
        load_clip("clip1", bad, clips)


def test_load_all_and_recording_theme(files):
    labels_dir, clips = files
    all_clips = load_all(labels_dir, clips)
    assert list(all_clips) == ["clip1"]  # clip2 has no labels file yet
    theme, labelled = recording_theme(all_clips, 3000)
    assert labelled.sum() == 50 and labelled[1000] and not labelled[999]
    assert theme[1000] == 0 and theme[1005] == 1 and theme[1020] == -1 and theme[1032] == 5
