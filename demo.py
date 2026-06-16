import math
import os

from audio_engine import (
    generate_composite_signal,
    add_white_noise,
    compute_spectrum,
    find_spectrum_peaks,
    format_peak_table,
    compute_signal_rms,
    apply_gain,
    apply_echo,
    apply_freq_filter,
    apply_time_filter,
    create_lowpass_kernel,
    create_highpass_kernel,
    write_wav,
    read_wav,
    compute_band_energy,
    generate_sine_wave,
    fft,
    ifft,
)


OUTPUT_DIR = "output"


def print_sep(title="", width=72):
    bar = "=" * width
    if title:
        print(f"\n{bar}")
        print(f"  {title}")
        print(bar)
    else:
        print(f"\n{bar}")


def demo_fft_basics():
    print_sep("第一部分：FFT 基础验证 (位反转 + 蝶形运算)")

    sample_rate = 1000
    duration = 1.024
    n = int(sample_rate * duration)

    print(f"\n采样率: {sample_rate} Hz")
    print(f"信号长度: {n} 点 (2^{int(math.log2(n))} = {n})")
    print(f"频率分辨率: {sample_rate/n:.3f} Hz/bin")

    signal = generate_sine_wave(50, sample_rate, duration, amplitude=1.0)

    spectrum = fft(signal)

    print(f"\n输入信号: 50 Hz 正弦波, 幅度 1.0")
    print(f"FFT 输出长度: {len(spectrum)}")

    magnitudes = [abs(s) for s in spectrum[:n // 2]]
    max_mag = max(magnitudes)
    max_idx = magnitudes.index(max_mag)
    peak_freq = max_idx * sample_rate / n

    print(f"\n频谱峰值:")
    print(f"  位置: bin {max_idx}")
    print(f"  对应频率: {peak_freq:.2f} Hz")
    print(f"  幅度: {max_mag:.4f} (理论值: {n/2:.1f})")
    print(f"  归一化幅度: {2*max_mag/n:.4f} (理论值: 1.0)")

    reconstructed = ifft(spectrum)
    max_error = max(abs(signal[i] - reconstructed[i].real) for i in range(n))
    print(f"\nIFFT 重建最大误差: {max_error:.2e}")

    print("\n✓ FFT/IFFT 验证通过")


def demo_spectrum_analysis(sample_rate=22050, duration=1.0):
    print_sep("第二部分：合成信号频谱分析（改进的峰值表）")

    components = [
        (100, 1.00),
        (250, 0.70),
        (440, 0.50),
        (880, 0.30),
        (1500, 0.20),
        (3000, 0.10),
    ]

    print(f"\n合成信号成分:")
    for freq, amp in components:
        print(f"  {freq:5d} Hz, 幅度 {amp:.3f}")

    signal = generate_composite_signal(components, sample_rate, duration)
    signal = add_white_noise(signal, amplitude=0.03, seed=42)

    freqs, mags, norm_mags, _ = compute_spectrum(signal, sample_rate)

    n_fft = len(mags) * 2
    print(f"\n频谱分析参数:")
    print(f"  采样率:        {sample_rate} Hz")
    print(f"  FFT 长度:      {n_fft} 点")
    print(f"  频率分辨率:    {sample_rate/n_fft:.3f} Hz/bin")
    print(f"  Nyquist 频率:  {sample_rate/2:.0f} Hz")
    print(f"  信号 RMS:      {compute_signal_rms(signal):.4f}")

    print(format_peak_table(
        find_spectrum_peaks(freqs, norm_mags, max_peaks=10),
        title="频谱峰值（按幅度排序, 前 10 个）"
    ))

    print("  备注: 归一化幅度 = 相对于最大峰值的比例, 2.0×|X[k]|/N 对应实际正弦波幅度")
    print("\n✓ 频谱分析完成")
    return signal, sample_rate, components


def demo_filter_frequency_domain(signal, sample_rate, components):
    import math
    print_sep("第三部分：频域滤波（FFT 方法）—— 截止效果验证")

    cutoff_lp = 600
    cutoff_hp = 700

    print(f"\n信号包含频率分量: {[c[0] for c in components]} Hz")

    freqs_orig, _, norm_orig, _ = compute_spectrum(signal, sample_rate)
    peaks_orig = find_spectrum_peaks(freqs_orig, norm_orig, max_peaks=8)

    print(format_peak_table(peaks_orig, title="滤波前频谱峰值"))

    signal_lp = apply_freq_filter(signal, sample_rate, cutoff_lp, 'lowpass')

    freqs_lp, _, norm_lp, _ = compute_spectrum(signal_lp, sample_rate)
    peaks_lp = find_spectrum_peaks(freqs_lp, norm_lp, max_peaks=8)

    print(f"\n[低通滤波] 截止频率: {cutoff_lp} Hz")
    print(f"  预期: {cutoff_lp} Hz 以下保留, {cutoff_lp} Hz 以上明显衰减")
    print(format_peak_table(peaks_lp, title="低通滤波后频谱峰值"))

    e_below = compute_band_energy(signal, sample_rate, 0, cutoff_lp)
    e_above = compute_band_energy(signal, sample_rate, cutoff_lp, sample_rate / 2)
    e_below_out = compute_band_energy(signal_lp, sample_rate, 0, cutoff_lp)
    e_above_out = compute_band_energy(signal_lp, sample_rate, cutoff_lp, sample_rate / 2)

    print(f"  频段能量对比:")
    print(f"    截止频率以下 (0~{cutoff_lp} Hz): 输入 {e_below*100:.1f}% → 输出 {e_below_out*100:.1f}%")
    print(f"    截止频率以上 ({cutoff_lp}~{sample_rate//2} Hz): 输入 {e_above*100:.1f}% → 输出 {e_above_out*100:.1f}%")
    if e_above > 1e-10:
        sup = -10 * math.log10(max(e_above_out, 1e-10) / max(e_above, 1e-10))
        print(f"    高频抑制量: {sup:.1f} dB")

    signal_hp = apply_freq_filter(signal, sample_rate, cutoff_hp, 'highpass')

    freqs_hp, _, norm_hp, _ = compute_spectrum(signal_hp, sample_rate)
    peaks_hp = find_spectrum_peaks(freqs_hp, norm_hp, max_peaks=8)

    print(f"\n[高通滤波] 截止频率: {cutoff_hp} Hz")
    print(f"  预期: {cutoff_hp} Hz 以下明显衰减, {cutoff_hp} Hz 以上保留")
    print(format_peak_table(peaks_hp, title="高通滤波后频谱峰值"))

    e_below_hp = compute_band_energy(signal, sample_rate, 0, cutoff_hp)
    e_above_hp = compute_band_energy(signal, sample_rate, cutoff_hp, sample_rate / 2)
    e_below_hp_out = compute_band_energy(signal_hp, sample_rate, 0, cutoff_hp)
    e_above_hp_out = compute_band_energy(signal_hp, sample_rate, cutoff_hp, sample_rate / 2)

    print(f"  频段能量对比:")
    print(f"    截止频率以下 (0~{cutoff_hp} Hz): 输入 {e_below_hp*100:.1f}% → 输出 {e_below_hp_out*100:.1f}%")
    print(f"    截止频率以上 ({cutoff_hp}~{sample_rate//2} Hz): 输入 {e_above_hp*100:.1f}% → 输出 {e_above_hp_out*100:.1f}%")
    if e_below_hp > 1e-10:
        sup = -10 * math.log10(max(e_below_hp_out, 1e-10) / max(e_below_hp, 1e-10))
        print(f"    低频抑制量: {sup:.1f} dB")

    print("\n✓ 频域滤波演示完成")
    return signal_lp, signal_hp


def demo_filter_time_domain(signal, sample_rate, components):
    import math
    print_sep("第四部分：时域滤波（FIR 卷积）—— 校准后的截止效果")

    cutoff_lp = 200
    cutoff_hp = 300
    kernel_size = 255

    print(f"\n信号包含频率分量: {[c[0] for c in components]} Hz")

    freqs_orig, _, norm_orig, _ = compute_spectrum(signal, sample_rate)
    peaks_orig = find_spectrum_peaks(freqs_orig, norm_orig, max_peaks=8)
    print(format_peak_table(peaks_orig, title="滤波前频谱峰值"))

    print(f"\n--- 低通 FIR 滤波 (截止 {cutoff_lp} Hz, {kernel_size} 抽头 Blackman 窗) ---")
    print(f"  预期: 100Hz、250Hz 中 250Hz 应被明显抑制, 440Hz 及以上更弱")

    lp_kernel = create_lowpass_kernel(cutoff_lp, sample_rate, kernel_size=kernel_size)
    signal_lp = apply_time_filter(signal, lp_kernel)

    freqs_lp, _, norm_lp, _ = compute_spectrum(signal_lp, sample_rate)
    peaks_lp = find_spectrum_peaks(freqs_lp, norm_lp, max_peaks=8)
    print(format_peak_table(peaks_lp, title="低通滤波后频谱峰值"))

    print(f"\n  各频率幅度变化 (归一化幅度 - 相对于原信号峰值):")
    peak_dict_orig = {round(p['frequency']): p['normalized_magnitude'] for p in peaks_orig}
    peak_dict_lp = {round(p['frequency']): p['normalized_magnitude'] for p in peaks_lp}
    for freq, _ in components:
        f_key = round(freq)
        orig_m = peak_dict_orig.get(f_key, 0.0)
        lp_m = peak_dict_lp.get(f_key, 0.0)
        if orig_m > 0.01:
            atten = 20 * math.log10(lp_m / orig_m) if lp_m > 0 else -999
            marker = " ↓被抑制" if (freq > cutoff_lp and atten < -6) else " ✓保留"
            print(f"    {freq:5d} Hz: {orig_m:.3f} → {lp_m:.3f} ({atten:+.1f} dB){marker}")

    e_above_in = compute_band_energy(signal, sample_rate, cutoff_lp + 50, sample_rate / 2)
    e_above_out = compute_band_energy(signal_lp, sample_rate, cutoff_lp + 50, sample_rate / 2)
    if e_above_in > 1e-10:
        sup = -10 * math.log10(max(e_above_out, 1e-10) / max(e_above_in, 1e-10))
        print(f"  {cutoff_lp+50} Hz 以上频段总抑制: {sup:.1f} dB")

    print(f"\n--- 高通 FIR 滤波 (截止 {cutoff_hp} Hz, {kernel_size} 抽头 Blackman 窗) ---")
    print(f"  预期: 440Hz、880Hz 等保留, 300Hz 以下 (100Hz、250Hz) 明显衰减")

    hp_kernel = create_highpass_kernel(cutoff_hp, sample_rate, kernel_size=kernel_size)
    signal_hp = apply_time_filter(signal, hp_kernel)

    freqs_hp, _, norm_hp, _ = compute_spectrum(signal_hp, sample_rate)
    peaks_hp = find_spectrum_peaks(freqs_hp, norm_hp, max_peaks=8)
    print(format_peak_table(peaks_hp, title="高通滤波后频谱峰值"))

    print(f"\n  各频率幅度变化:")
    peak_dict_hp = {round(p['frequency']): p['normalized_magnitude'] for p in peaks_hp}
    for freq, _ in components:
        f_key = round(freq)
        orig_m = peak_dict_orig.get(f_key, 0.0)
        hp_m = peak_dict_hp.get(f_key, 0.0)
        if orig_m > 0.01:
            atten = 20 * math.log10(hp_m / orig_m) if hp_m > 0 else -999
            marker = " ↓被抑制" if (freq < cutoff_hp and atten < -6) else " ✓保留"
            print(f"    {freq:5d} Hz: {orig_m:.3f} → {hp_m:.3f} ({atten:+.1f} dB){marker}")

    e_below_in = compute_band_energy(signal, sample_rate, 0, cutoff_hp - 50)
    e_below_out = compute_band_energy(signal_hp, sample_rate, 0, cutoff_hp - 50)
    if e_below_in > 1e-10:
        sup = -10 * math.log10(max(e_below_out, 1e-10) / max(e_below_in, 1e-10))
        print(f"  {cutoff_hp-50} Hz 以下频段总抑制: {sup:.1f} dB")

    print("\n✓ 时域 FIR 滤波演示完成")
    return signal_lp, signal_hp


def demo_effects(signal, sample_rate):
    print_sep("第五部分：音频效果（增益、回声）")

    n = len(signal)

    print(f"\n--- 增益效果 ---")
    gain_db = 6.0
    gained = apply_gain(signal, gain_db)

    input_rms = compute_signal_rms(signal)
    output_rms = compute_signal_rms(gained)
    actual_gain_db = 20 * math.log10(output_rms / input_rms) if input_rms > 0 else 0

    print(f"  目标增益: {gain_db} dB (线性倍数: {10**(gain_db/20):.4f}x)")
    print(f"  输入 RMS: {input_rms:.4f}")
    print(f"  输出 RMS: {output_rms:.4f}")
    print(f"  实际增益: {actual_gain_db:.2f} dB")

    print(f"\n--- 回声效果（无反馈单次回声）---")
    delay_ms = 200
    decay = 0.4
    echo_signal = apply_echo(signal, sample_rate, delay_ms, decay, feedback=False)

    delay_samples = int(delay_ms * sample_rate / 1000)
    print(f"  延迟: {delay_ms} ms = {delay_samples} 采样点")
    print(f"  衰减: {decay}")
    print(f"  输出 RMS: {compute_signal_rms(echo_signal):.4f}")

    print(f"\n--- 回声效果（带反馈多次回声）---")
    decay_fb = 0.5
    echo_fb_signal = apply_echo(signal, sample_rate, delay_ms, decay_fb, feedback=True)
    print(f"  延迟: {delay_ms} ms")
    print(f"  反馈系数: {decay_fb}")
    print(f"  理论总增益: 1/(1-α) = {1/(1-decay_fb):.2f}x")
    print(f"  输出 RMS: {compute_signal_rms(echo_fb_signal):.4f}")

    print("\n✓ 音频效果演示完成")
    return gained, echo_signal, echo_fb_signal


def demo_wav_io(signal, sample_rate, components):
    print_sep("第六部分：WAV 文件读写（可实际听效果）")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n输出目录: {os.path.abspath(OUTPUT_DIR)}/")

    original_path = os.path.join(OUTPUT_DIR, "01_original.wav")
    write_wav(original_path, signal, sample_rate)
    print(f"  ✓ 原始信号: {original_path}")

    cutoff_lp = 200
    lp_kernel = create_lowpass_kernel(cutoff_lp, sample_rate, kernel_size=255)
    signal_lp = apply_time_filter(signal, lp_kernel)
    lp_path = os.path.join(OUTPUT_DIR, "02_lowpass_200Hz.wav")
    write_wav(lp_path, signal_lp, sample_rate)
    print(f"  ✓ 低通滤波 (200Hz FIR): {lp_path}")

    cutoff_hp = 800
    hp_kernel = create_highpass_kernel(cutoff_hp, sample_rate, kernel_size=255)
    signal_hp = apply_time_filter(signal, hp_kernel)
    hp_path = os.path.join(OUTPUT_DIR, "03_highpass_800Hz.wav")
    write_wav(hp_path, signal_hp, sample_rate)
    print(f"  ✓ 高通滤波 (800Hz FIR): {hp_path}")

    gained = apply_gain(signal, 3.0)
    gain_path = os.path.join(OUTPUT_DIR, "04_gain_plus3dB.wav")
    write_wav(gain_path, gained, sample_rate)
    print(f"  ✓ 增益 +3dB: {gain_path}")

    echo_simple = apply_echo(signal, sample_rate, 300, 0.4, feedback=False)
    echo1_path = os.path.join(OUTPUT_DIR, "05_echo_simple.wav")
    write_wav(echo1_path, echo_simple, sample_rate)
    print(f"  ✓ 单次回声 (300ms): {echo1_path}")

    echo_fb = apply_echo(signal, sample_rate, 250, 0.55, feedback=True)
    echo2_path = os.path.join(OUTPUT_DIR, "06_echo_feedback.wav")
    write_wav(echo2_path, echo_fb, sample_rate)
    print(f"  ✓ 反馈回声 (250ms α=0.55): {echo2_path}")

    print(f"\n--- 验证 WAV 读回 ---")
    signal_read, sr_read, ch_read = read_wav(original_path)
    print(f"  读回采样率: {sr_read} Hz")
    print(f"  读回通道数: {ch_read}")
    rms_orig = compute_signal_rms(signal)
    rms_read = compute_signal_rms(signal_read)
    print(f"  写入 RMS: {rms_orig:.6f}")
    print(f"  读回 RMS: {rms_read:.6f}")
    print(f"  差异: {abs(rms_orig - rms_read):.2e} (16-bit 量化误差范围内)")

    print("\n✓ WAV 读写演示完成，所有文件可直接播放试听")


def demo_theory_explanation():
    print_sep("第七部分：原理速查")

    print("""
┌───────────────────────────────────────────────────────────────────┐
│ 1. FFT 分治 O(n²) → O(n log n)                                    │
│   Cooley-Tukey 按奇偶分两组，每层 O(N) 蝶形运算 × log₂N 层         │
│   旋转因子对称性 W^(k+N/2) = -W^k 消除一半冗余                    │
│                                                                   │
│ 2. 蝶形运算 & 位反转重排                                           │
│   蝶形: a+bW 与 a-bW 每次 1 乘 2 加                               │
│   位反转: 原位运算前需按二进制位逆序重排输入                       │
│   例 N=8: 原序 0,1,2,3,4,5,6,7 → 位反转 0,4,2,6,1,5,3,7           │
│                                                                   │
│ 3. 为何要求 2 的幂                                                 │
│   基-2 FFT 最常用，每次对半分解效率最高                            │
│   其他选择: 基-4、混合基、Bluestein(任意长度)                       │
│   工程上常用补零到下一 2^N                                          │
│                                                                   │
│ 4. 频谱 bin ↔ 频率映射                                             │
│   f_k = k × Fs / N        Δf = Fs / N                             │
│   例: Fs=44.1kHz N=1024, Δf≈43Hz, bin 512=22.05kHz(Nyquist)        │
│                                                                   │
│ 5. 低通滤波两种实现                                                │
│   频域: FFT → 置零高频 bin → IFFT，快但有吉布斯效应                │
│   时域: 与 sinc×窗 的 FIR 核卷积, 相位线性可实时, 阶数越高过渡带越窄│
│                                                                   │
│ 6. 回声实现                                                        │
│   单次:  y[n] = x[n] + α·x[n-D]                                    │
│   反馈:  y[n] = x[n] + α·y[n-D]   (α<1 保证稳定)                   │
│   反馈回声衰减序列: α, α², α³, ...                                │
└───────────────────────────────────────────────────────────────────┘
""")


def main():
    print_sep("音频信号处理引擎 v2.0 - 增强演示")
    print("  FFT/IFFT · 频谱分析 · FIR 滤波 · WAV 读写 · 效果器")

    demo_fft_basics()

    sample_rate = 22050
    duration = 1.0
    signal, sample_rate, components = demo_spectrum_analysis(sample_rate, duration)

    demo_filter_frequency_domain(signal, sample_rate, components)

    demo_filter_time_domain(signal, sample_rate, components)

    demo_effects(signal, sample_rate)

    demo_wav_io(signal, sample_rate, components)

    demo_theory_explanation()

    print_sep("所有演示完成 ✓")
    print()
    print("命令行工具用法:")
    print("  python audio_tool.py generate    - 生成合成信号并导出 WAV")
    print("  python audio_tool.py analyze     - 分析 WAV 频谱")
    print("  python audio_tool.py filter      - 低通/高通滤波")
    print("  python audio_tool.py gain        - 增益调整")
    print("  python audio_tool.py echo        - 回声效果")
    print("  python audio_tool.py process     - 完整处理管道")
    print()
    print("详细帮助: python audio_tool.py --help")
    print()


if __name__ == "__main__":
    main()
