import math
import cmath
import wave
import struct
import random
import os
import csv
import json


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


def compute_signal_peak(signal):
    if len(signal) == 0:
        return 0.0
    return max(abs(s) for s in signal)


def normalize_signal(signal, target_peak=0.95):
    if len(signal) == 0:
        return signal
    peak = compute_signal_peak(signal)
    if peak < 1e-10:
        return signal
    scale = target_peak / peak
    return [s * scale for s in signal]


def compute_spectrum(signal, sample_rate, pad_to_power_of_two=True):
    n = len(signal)
    work_signal = list(signal)
    original_n = n
    if pad_to_power_of_two and not is_power_of_two(n):
        padded_len = next_power_of_two(n)
        work_signal = work_signal + [0.0] * (padded_len - n)
        n = padded_len

    spectrum = fft(work_signal)
    magnitudes = [abs(s) for s in spectrum[:n // 2]]
    normalized_magnitudes = [2.0 * m / original_n for m in magnitudes]
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
    lines.append("-" * 65)
    lines.append(f"  {'序号':>4}  {'Bin':>6}  {'频率(Hz)':>12}  {'幅度':>10}  {'归一化':>8}  {'dB':>8}")
    lines.append("-" * 65)

    max_mag = max(p['magnitude'] for p in peaks) if peaks else 1.0

    for i, p in enumerate(peaks, 1):
        db = 20 * math.log10(p['magnitude'] / max_mag) if max_mag > 0 else -float('inf')
        lines.append(
            f"  {i:>4}  {p['bin']:>6}  {p['frequency']:>12.3f}  "
            f"{p['magnitude']:>10.4f}  {p['normalized_magnitude']:>8.4f}  {db:>+7.1f}"
        )
    lines.append("-" * 65)
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
        tw_bins = max(2, int(0.02 * bins))
    else:
        tw_bins = max(2, int(transition_width * n / sample_rate))

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


WINDOW_PARAMS = {
    'rectangular': {'beta': 0, 'transition_band': 1.8, 'stopband_atten': 21},
    'hann': {'beta': 0, 'transition_band': 3.1, 'stopband_atten': 44},
    'hamming': {'beta': 0, 'transition_band': 3.3, 'stopband_atten': 53},
    'blackman': {'beta': 0, 'transition_band': 5.5, 'stopband_atten': 74},
}


def _window_value(n, N, window_type='blackman'):
    if window_type == 'blackman':
        return 0.42 - 0.5 * math.cos(2 * math.pi * n / (N - 1)) + \
               0.08 * math.cos(4 * math.pi * n / (N - 1))
    elif window_type == 'hamming':
        return 0.54 - 0.46 * math.cos(2 * math.pi * n / (N - 1))
    elif window_type == 'hann':
        return 0.5 * (1 - math.cos(2 * math.pi * n / (N - 1)))
    elif window_type == 'rectangular':
        return 1.0
    else:
        return 0.42 - 0.5 * math.cos(2 * math.pi * n / (N - 1)) + \
               0.08 * math.cos(4 * math.pi * n / (N - 1))


def estimate_filter_order(sample_rate, transition_width_hz,
                          window_type='blackman'):
    params = WINDOW_PARAMS.get(window_type, WINDOW_PARAMS['blackman'])
    tb_norm = transition_width_hz / (sample_rate / 2.0)
    M = math.ceil(params['transition_band'] / tb_norm)
    if M % 2 == 0:
        M += 1
    return M


def create_lowpass_kernel(cutoff_freq, sample_rate, kernel_size=None,
                          window_type='blackman', transition_width=None):
    nyquist = sample_rate / 2.0

    if kernel_size is None:
        if transition_width is not None:
            kernel_size = estimate_filter_order(sample_rate, transition_width, window_type)
        else:
            kernel_size = 511

    if kernel_size % 2 == 0:
        kernel_size += 1

    half = kernel_size // 2
    M = kernel_size - 1

    fc_norm = cutoff_freq / nyquist

    kernel = []
    for i in range(kernel_size):
        n_val = i - half
        if n_val == 0:
            h = fc_norm
        else:
            h = math.sin(math.pi * fc_norm * n_val) / (math.pi * n_val)
        w = _window_value(i, kernel_size, window_type)
        kernel.append(h * w)

    total = sum(kernel)
    if abs(total) > 1e-10:
        kernel = [k / total for k in kernel]

    return kernel


def create_highpass_kernel(cutoff_freq, sample_rate, kernel_size=None,
                           window_type='blackman', transition_width=None):
    lowpass = create_lowpass_kernel(cutoff_freq, sample_rate, kernel_size,
                                    window_type, transition_width)
    k_len = len(lowpass)
    highpass = [-k for k in lowpass]
    mid = k_len // 2
    highpass[mid] += 1.0
    return highpass


def create_bandstop_kernel(f_low, f_high, sample_rate, kernel_size=None,
                           window_type='blackman'):
    nyquist = sample_rate / 2.0

    if kernel_size is None:
        tw = min(f_low, sample_rate / 2 - f_high) * 0.5
        kernel_size = estimate_filter_order(sample_rate, tw, window_type)

    if kernel_size % 2 == 0:
        kernel_size += 1

    half = kernel_size // 2

    fc1 = f_low / nyquist
    fc2 = f_high / nyquist

    kernel = []
    for i in range(kernel_size):
        n_val = i - half
        if n_val == 0:
            h = 1.0 - (fc2 - fc1)
        else:
            h = (math.sin(math.pi * n_val) - math.sin(math.pi * fc2 * n_val)
                 + math.sin(math.pi * fc1 * n_val)) / (math.pi * n_val)
        w = _window_value(i, kernel_size, window_type)
        kernel.append(h * w)

    total = sum(kernel)
    if abs(total) > 1e-10:
        kernel = [k / total for k in kernel]

    return kernel


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


def compute_filter_response(kernel, sample_rate, n_fft=4096):
    k_len = len(kernel)
    padded = list(kernel) + [0.0] * (n_fft - k_len)
    spectrum = fft(padded)
    magnitudes = [abs(s) for s in spectrum[:n_fft // 2]]
    max_mag = max(magnitudes) if max(magnitudes) > 0 else 1.0
    magnitudes_db = [20 * math.log10(m / max_mag + 1e-20) for m in magnitudes]
    frequencies = [k * sample_rate / n_fft for k in range(n_fft // 2)]
    return frequencies, magnitudes, magnitudes_db


def get_filter_cutoff_attenuation(kernel, sample_rate, target_freq):
    freqs, _, mags_db = compute_filter_response(kernel, sample_rate, n_fft=16384)
    for i in range(len(freqs)):
        if freqs[i] >= target_freq:
            return mags_db[i]
    return mags_db[-1]


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
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True) if os.path.dirname(file_path) else None

    work_signal = list(signal)

    if normalize:
        peak = compute_signal_peak(work_signal)
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


def analyze_signal_full(signal, sample_rate, label="", max_peaks=10,
                        min_freq_separation=None):
    rms = compute_signal_rms(signal)
    peak = compute_signal_peak(signal)
    duration = len(signal) / sample_rate if sample_rate > 0 else 0

    freqs, mags, norm_mags, _ = compute_spectrum(signal, sample_rate)
    peaks = find_spectrum_peaks(freqs, norm_mags, max_peaks=max_peaks,
                                 min_freq_separation=min_freq_separation)

    peak_freq = peaks[0]['frequency'] if peaks else 0.0
    peak_mag = peaks[0]['magnitude'] if peaks else 0.0

    low_band = compute_band_energy(signal, sample_rate, 0, 250)
    mid_band = compute_band_energy(signal, sample_rate, 250, 2000)
    high_band = compute_band_energy(signal, sample_rate, 2000, sample_rate / 2)

    result = {
        'label': label,
        'sample_rate': sample_rate,
        'n_samples': len(signal),
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'peak_db': 20 * math.log10(peak) if peak > 0 else -float('inf'),
        'crest_factor': peak / rms if rms > 0 else 0,
        'dominant_freq': peak_freq,
        'dominant_magnitude': peak_mag,
        'peaks': peaks,
        'band_energy_low': low_band,
        'band_energy_mid': mid_band,
        'band_energy_high': high_band,
    }
    return result


def format_analysis_summary(analysis, show_peaks=True):
    lines = []
    label = analysis.get('label', '')
    if label:
        lines.append(f"[{label}]")
        lines.append("-" * 55)

    lines.append(f"  采样点数:     {analysis['n_samples']}")
    lines.append(f"  时长:         {analysis['duration']:.3f} 秒")
    lines.append(f"  采样率:       {analysis['sample_rate']} Hz")
    lines.append(f"  RMS:          {analysis['rms']:.6f}")
    lines.append(f"  峰值幅度:     {analysis['peak']:.6f}")
    lines.append(f"  峰值/RMS 比:  {analysis['crest_factor']:.2f} ({20 * math.log10(analysis['crest_factor']):.1f} dB)")
    lines.append(f"  主导频率:     {analysis['dominant_freq']:.2f} Hz")

    lines.append(f"  频段能量分布: 低频 {analysis['band_energy_low']*100:.1f}% | "
                 f"中频 {analysis['band_energy_mid']*100:.1f}% | "
                 f"高频 {analysis['band_energy_high']*100:.1f}%")

    if show_peaks and analysis['peaks']:
        lines.append(format_peak_table(analysis['peaks'], title="频谱峰值"))

    return "\n".join(lines)


def export_spectrum_csv(file_path, analysis_info, signal=None, sample_rate=None,
                        include_full_spectrum=False):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True) if os.path.dirname(file_path) else None

    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        writer.writerow(['=== 信号分析汇总 ==='])
        writer.writerow(['项目', '值'])
        for key in ['filename', 'sample_rate', 'n_samples', 'duration',
                     'rms', 'peak', 'peak_db', 'crest_factor', 'dominant_freq',
                     'band_energy_low', 'band_energy_mid', 'band_energy_high']:
            if key in analysis_info:
                writer.writerow([key, analysis_info[key]])

        writer.writerow([])
        writer.writerow(['=== 频谱峰值表 ==='])
        writer.writerow(['排名', 'Bin索引', '频率(Hz)', '幅度', '归一化幅度', 'dB(相对于峰值)'])

        max_mag = max((p['magnitude'] for p in analysis_info.get('peaks', [])), default=1.0)
        for i, p in enumerate(analysis_info.get('peaks', []), 1):
            db = 20 * math.log10(p['magnitude'] / max_mag) if max_mag > 0 else -float('inf')
            writer.writerow([i, p['bin'], f"{p['frequency']:.4f}",
                            f"{p['magnitude']:.6f}",
                            f"{p['normalized_magnitude']:.6f}",
                            f"{db:.2f}"])

        if include_full_spectrum and signal is not None and sample_rate is not None:
            freqs, mags, norm_mags, _ = compute_spectrum(signal, sample_rate)
            writer.writerow([])
            writer.writerow(['=== 完整频谱 ==='])
            writer.writerow(['Bin', '频率(Hz)', '幅度', '归一化幅度'])
            for i in range(len(freqs)):
                writer.writerow([i, f"{freqs[i]:.4f}",
                                f"{mags[i]:.6f}", f"{norm_mags[i]:.6f}"])


def find_wav_files(directory):
    if not os.path.isdir(directory):
        return []
    wav_files = []
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith('.wav'):
            wav_files.append(os.path.join(directory, f))
    return wav_files


DEFAULT_PRESETS = {
    'vocal-clean': {
        'description': '人声清洁：去除低频隆隆声和高频嘶声外的杂音',
        'highpass': 80,
        'lowpass': 8000,
        'gain': 0,
        'filter_domain': 'time',
        'kernel_size': None,
    },
    'bass-cut': {
        'description': '切除低音：去除 200Hz 以下的低频成分',
        'highpass': 200,
        'lowpass': None,
        'gain': 0,
        'filter_domain': 'time',
        'kernel_size': 511,
    },
    'bright-boost': {
        'description': '提升明亮度：高通+增益，让声音更亮',
        'highpass': 300,
        'lowpass': None,
        'gain': 3,
        'filter_domain': 'freq',
    },
    'warm-vintage': {
        'description': '温暖复古：低通滤波去除高频',
        'highpass': None,
        'lowpass': 3000,
        'gain': 0,
        'filter_domain': 'time',
    },
    'echo-room': {
        'description': '房间回声：中等延迟带反馈',
        'echo_delay': 250,
        'echo_decay': 0.4,
        'echo_feedback': True,
        'gain': 0,
    },
    'echo-cathedral': {
        'description': '大教堂回声：长延迟高衰减',
        'echo_delay': 800,
        'echo_decay': 0.55,
        'echo_feedback': True,
        'gain': -2,
    },
    'telephone': {
        'description': '电话音效果：带通滤波',
        'highpass': 300,
        'lowpass': 3400,
        'gain': 0,
        'filter_domain': 'freq',
    },
    'subwoofer-test': {
        'description': '低音炮测试：只保留超低频',
        'highpass': None,
        'lowpass': 120,
        'gain': 6,
        'filter_domain': 'time',
        'kernel_size': 1023,
    },
}


def load_presets(custom_file=None):
    presets = dict(DEFAULT_PRESETS)
    if custom_file and os.path.exists(custom_file):
        try:
            with open(custom_file, 'r', encoding='utf-8') as f:
                custom = json.load(f)
            presets.update(custom)
        except Exception:
            pass
    return presets


def save_preset(name, config, custom_file='custom_presets.json'):
    presets = {}
    if os.path.exists(custom_file):
        try:
            with open(custom_file, 'r', encoding='utf-8') as f:
                presets = json.load(f)
        except Exception:
            presets = {}

    presets[name] = config

    with open(custom_file, 'w', encoding='utf-8') as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)


def list_presets(presets=None):
    if presets is None:
        presets = load_presets()
    lines = ["可用预设:"]
    for name, cfg in presets.items():
        desc = cfg.get('description', '')
        lines.append(f"  {name:20s} - {desc}")
    return "\n".join(lines)
