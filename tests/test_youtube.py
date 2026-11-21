import pytest

from interview_prep.models import Segment
from interview_prep.youtube import (
    YouTubeError,
    _clean,
    fmt_ts,
    is_playlist_only,
    parse_playlist_id,
    parse_video_id,
    split_blocks,
    timestamped_text,
)

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    VID,
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtube.com/watch?v={VID}&t=42s&list=PL123",
    f"https://m.youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}?si=abc",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}",
    f"youtube.com/watch?v={VID}",
])
def test_parse_video_id(url):
    assert parse_video_id(url) == VID


@pytest.mark.parametrize("bad", ["", "hello", "https://vimeo.com/123", "https://www.youtube.com/watch?v=short",
                                 "https://www.youtube.com/playlist?list=PL123"])
def test_parse_video_id_rejects(bad):
    with pytest.raises(YouTubeError):
        parse_video_id(bad)


def test_playlist_detection():
    assert parse_playlist_id("https://www.youtube.com/playlist?list=PLabc") == "PLabc"
    assert parse_playlist_id(f"https://www.youtube.com/watch?v={VID}&list=PLabc") == "PLabc"
    assert parse_playlist_id(f"https://www.youtube.com/watch?v={VID}&list=RDxyz") is None  # auto mix
    assert parse_playlist_id(f"https://www.youtube.com/watch?v={VID}") is None
    assert is_playlist_only("https://www.youtube.com/playlist?list=PLabc")
    assert not is_playlist_only(f"https://www.youtube.com/watch?v={VID}&list=PLabc")


def test_fmt_ts():
    assert fmt_ts(0) == "00:00"
    assert fmt_ts(195.9) == "03:15"
    assert fmt_ts(3725) == "1:02:05"


def test_clean_removes_noise():
    assert _clean("[Music]  hello\nworld [Applause]") == "hello world"


def test_timestamped_text_groups_by_block():
    segs = [Segment(start=s, text=f"t{s}") for s in [0, 10, 20, 31, 45, 62]]
    out = timestamped_text(segs, block_seconds=30).splitlines()
    assert out == ["[00:00] t0 t10 t20", "[00:31] t31 t45", "[01:02] t62"]


def test_split_blocks_respects_limit():
    text = "\n".join(f"[00:{i:02d}] " + "x" * 50 for i in range(20))
    chunks = split_blocks(text, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "\n".join(chunks) == text
