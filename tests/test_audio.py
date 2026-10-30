"""Independent boundary tests for untrusted audio and the feature contract."""

from io import BytesIO

import numpy as np
import pytest
from scipy.io import wavfile

from cardio_audio.audio import decode_audio, extract_features, select_segment


@pytest.mark.parametrize("sample_rate", [4000, 8000, 44100])
@pytest.mark.parametrize("channels", [1, 2])
def test_pcm_is_decoded_to_finite_mono_float32(wav_factory, sample_rate, channels):
    result = decode_audio(wav_factory(duration=0.5, sample_rate=sample_rate, channels=channels))
    assert result.sample_rate == sample_rate
    assert result.channels == channels
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert len(result.samples) == int(0.5 * sample_rate)
    assert np.isfinite(result.samples).all()
    assert np.max(np.abs(result.samples)) <= 1
    assert result.duration_s == pytest.approx(0.5, abs=1 / sample_rate)


@pytest.mark.parametrize("payload", [b"", b"not an audio", b"RIFF\x00\x00\x00\x00WAVE", b"ID3fake"])
def test_corrupt_or_non_wav_audio_is_rejected(payload):
    with pytest.raises(ValueError):
        decode_audio(payload)


def test_byte_limit_is_enforced():
    with pytest.raises(ValueError):
        decode_audio(b"x" * (20 * 1024 * 1024 + 1))


@pytest.mark.parametrize("sample_rate", [1000, 200000])
def test_unsupported_sample_rates_are_rejected(wav_factory, sample_rate):
    with pytest.raises(ValueError):
        decode_audio(wav_factory(duration=0.5, sample_rate=sample_rate))


def test_more_than_two_channels_are_rejected(wav_factory):
    with pytest.raises(ValueError):
        decode_audio(wav_factory(channels=3))


def test_duration_limit_is_enforced(wav_factory):
    with pytest.raises(ValueError):
        decode_audio(wav_factory(duration=120.1, sample_rate=2000))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_float_wav_is_rejected(bad_value):
    buffer = BytesIO()
    samples = np.full(4000, 0.1, dtype=np.float32)
    samples[10] = bad_value
    wavfile.write(buffer, 4000, samples)
    with pytest.raises(ValueError):
        decode_audio(buffer.getvalue())


@pytest.mark.parametrize(
    "start,end",
    [(-0.1, 0.8), (0.8, 0.7), (0.5, 0.5), (0.0, 0.09),
     (0.0, 1.1), (np.nan, 0.5), (0.0, np.nan), (np.inf, 0.5), (0.0, np.inf)],
)
def test_invalid_segment_ranges_are_rejected(wav_factory, start, end):
    audio = decode_audio(wav_factory(duration=1.0))
    with pytest.raises(ValueError):
        select_segment(audio, start, end)


def test_segment_respects_sample_boundaries(wav_factory):
    audio = decode_audio(wav_factory(duration=1.0, sample_rate=8000))
    result = select_segment(audio, 0.25, 0.75)
    np.testing.assert_array_equal(result, audio.samples[2000:6000])


def test_features_have_stable_finite_shape_and_are_deterministic(wav_factory):
    audio = decode_audio(wav_factory(duration=0.75, sample_rate=8000))
    first = extract_features(audio.samples, audio.sample_rate)
    second = extract_features(audio.samples.copy(), audio.sample_rate)
    assert first.shape == (64, 256)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_features_include_the_end_of_a_long_cycle():
    sample_rate = 4000
    background_time = np.arange(5 * sample_rate) / sample_rate
    background = (0.05 * np.sin(2 * np.pi * 150 * background_time)).astype(np.float32)
    with_late_event = background.copy()
    time = np.arange(sample_rate // 2) / sample_rate
    with_late_event[-len(time):] = 0.2 * np.sin(2 * np.pi * 440 * time)
    baseline = extract_features(background, sample_rate)
    event = extract_features(with_late_event, sample_rate)
    assert np.isfinite(baseline).all()
    assert np.isfinite(event).all()
    assert not np.allclose(event[:, -25:], baseline[:, -25:])


def test_silence_cannot_be_classified_as_a_sound(wav_factory):
    audio = decode_audio(wav_factory(amplitude=0.0))
    with pytest.raises(ValueError):
        select_segment(audio, 0.0, 1.0)
    with pytest.raises(ValueError):
        extract_features(audio.samples, audio.sample_rate)
