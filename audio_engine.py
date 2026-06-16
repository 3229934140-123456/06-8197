import math
import cmath
import wave
import struct
import random


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n):
    if n == 0:
        return 1
    if is_power_of_two(n):
        return n
    return 1 << n.bit_length()


def bit_reverse(n, bits):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def fft(signal, inverse=False):
    n = len(signal)
    if not is_power_of_two(n):
        raise ValueError("FFT requires length to be power of 2")

    bits = int(math.log2(n))

    result = [complex(signal[i]) if not isinstance(signal[i], complex)
              else signal[i] for i in range(n)]

    for i in range(n):
        j = bit_reverse(i, bits)
        if i < j:
            result[i], result[j] = result[j], result[i]

    direction = -1 if not inverse else 1

    size = 2
    while size <= n:
        half_size = size // 2
        angle_step = direction * 2 * math.pi / size
        w_root = cmath.exp(complex(0, angle_step))

        for start in range(0, n, size):
            w = complex(1, 0)
            for k in range(half_size):
                even = result[start + k]
                odd = w * result[start + k + half_size]
                result[start + k] = even + odd
                result[start + k + half_size] = even - odd
                w *= w_root

        size *= 2

    if inverse:
        for i in range(n):
            result[i] /= n

    return result


def ifft(spectrum):
    return fft(spectrum, inverse=True)


def compute_signal_rms(signal):
    if len(signal) == 0:
        return 0.0
    return math.sqrt(sum(s ** 2 for s in signal) / len(signal))


def normalize_signal(signal, target_peak=0.95):
    if len(signal) == 0:
        return signal
    peak = max(abs(s) for s in signal)
    if peak < 1e-10:
        return signal
    scale = target_peak / peak
    return [s * scale for s in signal]


def compute_spectrum(signal, sample_rate, pad_to_power_of_two=True):
    n = len(signal)
    work_signal = list(signal)
    if pad_to_power_of_two and not is_power_of_two(n):
        padded_len = next_power_of_two(n)
        work_signal = work_signal + [0.0] * (padded_len - n)
        n = padded_len

    spectrum = fft(work_signal)
    magnitudes = [abs(s) for s in spectrum[:n // 2]]
    normalized_magnitudes = [2.0 * m / len(signal) for m in magnitudes]
    frequencies = [k * sample_rate / n for k in range(n // 2)]
    return frequencies, magnitudes, normalized_magnitudes, spectrum


def find_spectrum_peaks(frequencies, magnitudes, min_magnitude_ratio=0.02,
                        max_peaks=None, min_freq_separation=None):
    n = len(magnitudes)
    if n == 0:
        return []

    max_mag = max(magnitudes)
    if max_mag < 1e-10:
        return []

    peak_threshold = max_mag * min_magnitude_ratio
    raw_peaks = []

    for k in range(1, n - 1):
        if magnitudes[k] > magnitudes[k - 1] and magnitudes[k] > magnitudes[k + 1]:
            if magnitudes[k] >= peak_threshold:
                peak_freq = frequencies[k]
                peak_norm_mag = magnitudes[k] / max_mag
                raw_peaks.append({
                    'bin': k,
                    'frequency': peak_freq,
                    'magnitude': magnitudes[k],
                    'normalized_magnitude': peak_norm_mag
                })

    raw_peaks.sort(key=lambda x: x['magnitude'], reverse=True)

    if min_freq_separation is not None:
        selected = []
        for p in raw_peaks:
            ok = True
            for s in selected:
                if abs(p['frequency'] - s['frequency']) < min_freq_separation:
                    ok = False
                    break
            if ok:
                selected.append(p)
            if max_peaks is not None and len(selected) >= max_peaks:
                break
        result = selected
    else:
        result = raw_peaks

    if max_peaks is not None:
        result = result[:max_peaks]

    return result


def format_peak_table(peaks, title="频谱峰值"):
    if len(peaks) == 0:
        return f"{title}: (未检测到显著峰值)\n"

    lines = [f"\n{title}:"]
    lines.append("-" * 60)
    lines.append(f"  {'序号':>4}  {'Bin':>6}  {'频率(Hz)':>12}  {'幅度':>10}  {'归一化幅度':>10}")
    lines.append("-" * 60)

    for i, p in enumerate(peaks, 1):
        lines.append(
            f"  {i:>4}  {p['bin']:>6}  {p['frequency']:>12.3f}  "
            f"{p['magnitude']:>10.4f}  {p['normalized_magnitude']:>10.4f}"
        )
    lines.append("-" * 60)
    return "\n".join(lines) + "\n"


def apply_freq_filter(signal, sample_rate, cutoff_freq, filter_type='lowpass',
                      transition_width=None):
    n = len(signal)
    original_n = n
    if not is_power_of_two(n):
        padded_len = next_power_of_two(n)
        signal = list(signal) + [0.0] * (padded_len - n)
        n = padded_len

    spectrum = fft(signal)
    bins = n // 2
    cutoff_bin = min(int(cutoff_freq * n / sample_rate), bins)

    if transition_width is None:
        tw_bins = max(1, int(0.05 * bins))
    else:
        tw_bins = max(1, int(transition_width * n / sample_rate))

    filtered_spectrum = list(spectrum)

    if filter_type == 'lowpass':
        for k in range(bins + 1):
            if k <= cutoff_bin - tw_bins:
                pass
            elif k <= cutoff_bin + tw_bins:
                t = (k - (cutoff_bin - tw_bins)) / (2 * tw_bins)
                w = 0.5 * (1 + math.cos(math.pi * t))
                filtered_spectrum[k] *= w
                if k > 0:
                    filtered_spectrum[n - k] *= w
            else:
                filtered_spectrum[k] = 0
                if k > 0 and k < bins:
                    filtered_spectrum[n - k] = 0
    elif filter_type == 'highpass':
        for k in range(bins + 1):
            if k >= cutoff_bin + tw_bins:
                pass
            elif k >= cutoff_bin - tw_bins:
                t = ((cutoff_bin + tw_bins) - k) / (2 * tw_bins)
                w = 0.5 * (1 + math.cos(math.pi * t))
                filtered_spectrum[k] *= w
                if k > 0:
                    filtered_spectrum[n - k] *= w
            else:
                filtered_spectrum[k] = 0
                if k > 0:
                    filtered_spectrum[n - k] = 0
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    filtered_signal = ifft(filtered_spectrum)
    return [s.real for s in filtered_signal[:original_n]]


def _window_blackman(n, N):
    return 0.42 - 0.5 * math.cos(2 * math.pi * n / (N - 1)) + \
           0.08 * math.cos(4 * math.pi * n / (N - 1))


def _window_hamming(n, N):
    return 0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1))


def create_lowpass_kernel(cutoff_freq, sample_rate, kernel_size=127,
                          window_type='blackman'):
    if kernel_size % 2 == 0:
        kernel_size += 1

    nyquist = sample_rate / 2.0
    half = kernel_size // 2
    kernel = []

    M = kernel_size - 1
    if window_type == 'blackman':
        bw = 5.5 / M
    else:
        bw = 3.3 / M

    fc_corrected = (cutoff_freq + bw * nyquist / 2) / nyquist
    fc = min(0.95, max(0.001, cutoff_freq / nyquist))

    for i in range(kernel_size):
        n = i - half
        if n == 0:
            h = 2 * fc
        else:
            h = math.sin(2 * math.pi * fc * n) / (math.pi * n)

        if window_type == 'blackman':
            w = _window_blackman(i, kernel_size)
        elif window_type == 'hamming':
            w = _window_hamming(i, kernel_size)
        else:
            w = 1.0

        kernel.append(h * w)

    total = sum(kernel)
    if abs(total) > 1e-10:
        kernel = [k / total for k in kernel]

    return kernel


def create_highpass_kernel(cutoff_freq, sample_rate, kernel_size=127,
                           window_type='blackman'):
    lowpass = create_lowpass_kernel(cutoff_freq, sample_rate, kernel_size, window_type)
    highpass = [-k for k in lowpass]
    mid = kernel_size // 2
    highpass[mid] += 1.0
    return highpass


def apply_time_filter(signal, kernel):
    n = len(signal)
    k_len = len(kernel)
    half = k_len // 2
    result = [0.0] * n

    padded = [0.0] * half + list(signal) + [0.0] * half

    for i in range(n):
        s = 0.0
        for j in range(k_len):
            s += padded[i + j] * kernel[j]
        result[i] = s

    return result


def apply_gain(signal, gain_db):
    gain_linear = 10 ** (gain_db / 20.0)
    return [s * gain_linear for s in signal]


def apply_echo(signal, sample_rate, delay_ms, decay, feedback=False):
    delay_samples = int(delay_ms * sample_rate / 1000.0)
    n = len(signal)
    result = [0.0] * n

    if delay_samples <= 0:
        return list(signal)

    if not feedback:
        for i in range(n):
            result[i] = signal[i]
            if i >= delay_samples:
                result[i] += decay * signal[i - delay_samples]
    else:
        for i in range(n):
            result[i] = signal[i]
            if i >= delay_samples:
                result[i] += decay * result[i - delay_samples]

    return result


def generate_sine_wave(freq, sample_rate, duration, amplitude=1.0):
    n = int(sample_rate * duration)
    signal = []
    for i in range(n):
        t = i / sample_rate
        signal.append(amplitude * math.sin(2 * math.pi * freq * t))
    return signal


def generate_composite_signal(components, sample_rate, duration):
    n = int(sample_rate * duration)
    signal = [0.0] * n
    for freq, amp in components:
        for i in range(n):
            t = i / sample_rate
            signal[i] += amp * math.sin(2 * math.pi * freq * t)
    return signal


def add_white_noise(signal, amplitude=0.1, seed=None):
    if seed is not None:
        random.seed(seed)
    return [s + amplitude * (2 * random.random() - 1) for s in signal]


def read_wav(file_path):
    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"仅支持 16-bit PCM WAV 文件, 当前为 {sample_width * 8}-bit")

    fmt = f"<{n_frames * n_channels}h"
    samples = struct.unpack(fmt, raw_data)

    max_val = float(2 ** (sample_width * 8 - 1))

    if n_channels == 1:
        signal = [s / max_val for s in samples]
    else:
        signal = [((samples[i] + samples[i + 1]) / 2.0) / max_val
                  for i in range(0, len(samples), n_channels)]

    return signal, sample_rate, n_channels


def write_wav(file_path, signal, sample_rate, n_channels=1, normalize=True):
    work_signal = list(signal)

    if normalize:
        peak = max(abs(s) for s in work_signal) if work_signal else 0
        if peak > 1e-10 and peak > 0.95:
            scale = 0.95 / peak
            work_signal = [s * scale for s in work_signal]

    max_val = 2 ** 15 - 1
    int_samples = []
    for s in work_signal:
        v = int(max(-1.0, min(1.0, s)) * max_val)
        for _ in range(n_channels):
            int_samples.append(v)

    fmt = f"<{len(int_samples)}h"
    raw_data = struct.pack(fmt, *int_samples)

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.setnframes(len(work_signal))
        wf.writeframes(raw_data)


def compute_band_energy(signal, sample_rate, freq_low, freq_high):
    n = len(signal)
    n_fft = next_power_of_two(n)
    padded = list(signal) + [0.0] * (n_fft - n)
    spectrum = fft(padded)
    magnitudes = [abs(s) for s in spectrum[:n_fft // 2]]

    bin_low = max(0, int(freq_low * n_fft / sample_rate))
    bin_high = min(n_fft // 2 - 1, int(freq_high * n_fft / sample_rate))

    if bin_high <= bin_low:
        return 0.0

    total_energy = sum(m ** 2 for m in magnitudes)
    band_energy = sum(m ** 2 for m in magnitudes[bin_low:bin_high + 1])

    if total_energy < 1e-20:
        return 0.0
    return band_energy / total_energy
