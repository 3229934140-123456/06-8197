import math
import cmath


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


def compute_spectrum(signal, sample_rate):
    n = len(signal)
    spectrum = fft(signal)
    magnitudes = [abs(s) for s in spectrum[:n//2]]
    frequencies = [k * sample_rate / n for k in range(n//2)]
    return frequencies, magnitudes, spectrum


def apply_freq_filter(signal, sample_rate, cutoff_freq, filter_type='lowpass'):
    n = len(signal)
    original_n = n
    if not is_power_of_two(n):
        padded_len = next_power_of_two(n)
        signal = list(signal) + [0.0] * (padded_len - n)
        n = padded_len

    spectrum = fft(signal)
    bins = n // 2
    cutoff_bin = min(int(cutoff_freq * n / sample_rate), bins)

    filtered_spectrum = list(spectrum)

    if filter_type == 'lowpass':
        for k in range(cutoff_bin + 1, bins + 1):
            if k < n:
                filtered_spectrum[k] = 0
            if k > 0 and k < bins:
                filtered_spectrum[n - k] = 0
    elif filter_type == 'highpass':
        for k in range(0, cutoff_bin):
            filtered_spectrum[k] = 0
            if k > 0:
                filtered_spectrum[n - k] = 0
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    filtered_signal = ifft(filtered_spectrum)
    return [s.real for s in filtered_signal[:original_n]]


def create_lowpass_kernel(cutoff_freq, sample_rate, kernel_size=31):
    if kernel_size % 2 == 0:
        kernel_size += 1

    nyquist = sample_rate / 2.0
    fc = cutoff_freq / nyquist
    half = kernel_size // 2
    kernel = []

    for i in range(kernel_size):
        n = i - half
        if n == 0:
            h = 2 * fc
        else:
            h = math.sin(2 * math.pi * fc * n) / (math.pi * n)
        window = 0.42 - 0.5 * math.cos(2 * math.pi * i / (kernel_size - 1)) + \
                 0.08 * math.cos(4 * math.pi * i / (kernel_size - 1))
        kernel.append(h * window)

    total = sum(kernel)
    kernel = [k / total for k in kernel]

    return kernel


def create_highpass_kernel(cutoff_freq, sample_rate, kernel_size=31):
    lowpass = create_lowpass_kernel(cutoff_freq, sample_rate, kernel_size)
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


def add_white_noise(signal, amplitude=0.1):
    import random
    random.seed(42)
    return [s + amplitude * (2 * random.random() - 1) for s in signal]
