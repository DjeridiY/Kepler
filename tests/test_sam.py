import importlib.metadata

import numpy as np
import pytest

from kepler.sam import Box, SamError, box_prompt, parse_records, track

REAL = (
    "<34f>0<|box;x1=395;y1=163;x2=429;y2=197;w=1280;h=360|>"
    "<|mask;x=0;y=0;data=35,35,!!!!!EGO$D&d=cNg:b=5](9_-K7$NKXJv`&AEv&lx3%dq|>\n"
    "<35f>0<|box;x1=402;y1=163;x2=437;y2=197;w=1280;h=360|>"
    "<|mask;x=0;y=0;data=35,36,!!!!!HG=Td^0=cGvUYYBXIuF`(EPb%U!odfU7(mt+:O4_V!!!|>\n"
)


def test_parse_real_output() -> None:
    records = parse_records(REAL)
    assert records == {34: {0: Box(395, 163, 429, 197)}, 35: {0: Box(402, 163, 437, 197)}}
    assert records[34][0].center == (412.0, 180.0)


def test_parse_two_objects_on_one_line() -> None:
    line = (
        "<3f>0<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|><|mask;x=0;y=0;data=1,1,!!|>,"
        "7<|box;x1=5;y1=6;x2=7;y2=8;w=10;h=10;c=0.9|>"
    )
    assert parse_records(line) == {3: {0: Box(1, 2, 3, 4), 7: Box(5, 6, 7, 8)}}


def test_parse_rejects_malformed_line() -> None:
    line = "garbage " * 20
    with pytest.raises(SamError) as err:
        parse_records("\n" + line)
    assert str(err.value) == "bad record: " + line[:80]


def test_box_prompt_format() -> None:
    assert box_prompt(Box(123, 163, 157, 197), 1280, 360) == "<0f>1<|box;x1=123;y1=163;x2=157;y2=197;w=1280;h=360|>"


def test_openai_client_is_a_declared_dependency() -> None:
    requires = importlib.metadata.requires("kepler") or []
    assert any(r.startswith("openai") for r in requires)


class FakeTransport:
    def __init__(self, image_out: str) -> None:
        self.image_out = image_out
        self.video_prompts: list[str] = []

    def segment_image(self, jpeg: bytes, prompt: str) -> str:
        assert jpeg[:2] == b"\xff\xd8"
        return self.image_out

    def segment_video(self, path: str, prompt: str) -> tuple[str, int]:
        self.video_prompts.append(prompt)
        lines = [f"<{i}f>1<|box;x1={10 + i};y1=20;x2={20 + i};y2=30;w=64;h=48|>" for i in range(3)]
        lines.insert(1, "<0f>2<|box;x1=0;y1=0;x2=2;y2=2;w=64;h=48|>")
        return "\n".join(lines), 20


FRAME0 = np.zeros((48, 64, 3), np.uint8)


def test_track_uses_detected_box_as_video_prompt() -> None:
    fake = FakeTransport("<0f>0<|box;x1=1;y1=1;x2=3;y2=3;w=64;h=48|>,1<|box;x1=10;y1=20;x2=20;y2=30;w=64;h=48;c=0.8|>")
    costs: list[tuple[str, float]] = []
    t = track(fake, "v.mp4", "red ball", frame0_bgr=FRAME0, on_cost=lambda k, c: costs.append((k, c)))
    assert fake.video_prompts == [box_prompt(Box(10, 20, 20, 30), 64, 48)]
    assert t.indices.tolist() == [0, 1, 2]
    assert t.indices.dtype.kind == "i"
    assert t.centers.tolist() == [[15.0, 25.0], [16.0, 25.0], [17.0, 25.0]]
    assert t.n_frames == 20
    assert costs == [("sam_image", 0.0025), ("sam_video", pytest.approx(0.01))]


def test_track_stops_when_frame0_has_no_object() -> None:
    fake = FakeTransport("")
    with pytest.raises(SamError, match="object not found on frame 0"):
        track(fake, "v.mp4", "red ball", frame0_bgr=FRAME0)
    assert fake.video_prompts == []
