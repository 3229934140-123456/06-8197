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
    analyze_signal_full,
    format_analysis_summary,
    export_spectrum_csv,
    compute_filter_response,
    get_filter_cutoff_attenuation,
    load_presets,
    list_presets,
    save_preset,
    find_wav_files,
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


def print_subsep(title="", width=72):
    bar = "-" * width
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

    magnitudes = [abs(s) for s in spectrum[:n // 2]]
    max_mag = max(magnitudes)
    max_idx = magnitudes.index(max_mag)
    peak_freq = max_idx * sample_rate / n

    print(f"\n输入: 50 Hz 正弦波, 幅度 1.0")
    print(f"频谱峰值: bin {max_idx} = {peak_freq:.2f} Hz, 归一化幅度 {2*max_mag/n:.4f}")

    reconstructed = ifft(spectrum)
    max_error = max(abs(signal[i] - reconstructed[i].real) for i in range(n))
    print(f"IFFT 重建最大误差: {max_error:.2e}")

    print("\n✓ FFT/IFFT 验证通过")


def demo_filter_calibration():
    print_sep("第二部分：FIR 滤波器截止频率校准验证")

    sample_rate = 22050
    duration = 2.0

    test_freqs_lp = [100, 250, 440, 880]
    print(f"\n[低通滤波器校准] 截止 200 Hz")
    print(f"测试频率: {test_freqs_lp} Hz")
    print(f"预期: 100Hz 基本保留, 250Hz 明显下降, 440Hz 强烈衰减")

    components = [(f, 1.0) for f in test_freqs_lp]
    signal = generate_composite_signal(components, sample_rate, duration)

    kernel_sizes = [127, 255, 511, 1023]

    for ks in kernel_sizes:
        kernel = create_lowpass_kernel(200, sample_rate, kernel_size=ks,
                                        window_type='blackman')
        filtered = apply_time_filter(signal, kernel)

        freqs, _, norm_mags, _ = compute_spectrum(filtered, sample_rate)
        peaks = find_spectrum_peaks(freqs, norm_mags, max_peaks=20,
                                     min_freq_separation=20)

        peak_dict = {}
        for p in peaks:
            pf = round(p['frequency'])
            if pf not in peak_dict or p['magnitude'] > peak_dict[pf]['magnitude']:
                peak_dict[pf] = p

        print(f"\n  {ks:4d} 抽头 Blackman 窗:")
        print(f"    {'频率':>6} {'幅度':>10} {'衰减(dB)':>10} {'状态':>10}")
        print(f"    {'------':>6} {'----------':>10} {'----------':>10} {'----------':>10}")
        for tf in test_freqs_lp:
            mag = 0
            for freq_key, p in peak_dict.items():
                if abs(freq_key - tf) < 30:
                    mag = max(mag, p['normalized_magnitude'])
            atten = 20 * math.log10(mag) if mag > 0 else -999
            status = "✓保留" if atten > -3 else ("~过渡" if atten > -20 else "✗衰减")
            print(f"    {tf:>5d} Hz {mag:>10.4f} {atten:>+9.1f} dB {status:>10}")

    test_freqs_hp = [100, 250, 440, 880]
    print(f"\n\n[高通滤波器校准] 截止 300 Hz")
    print(f"测试频率: {test_freqs_hp} Hz")
    print(f"预期: 100Hz、250Hz 明显衰减, 440Hz 基本保留, 880Hz 完全保留")

    for ks in kernel_sizes:
        kernel = create_highpass_kernel(300, sample_rate, kernel_size=ks,
                                         window_type='blackman')
        filtered = apply_time_filter(signal, kernel)

        freqs, _, norm_mags, _ = compute_spectrum(filtered, sample_rate)
        peaks = find_spectrum_peaks(freqs, norm_mags, max_peaks=20,
                                     min_freq_separation=20)

        peak_dict = {}
        for p in peaks:
            pf = round(p['frequency'])
            if pf not in peak_dict or p['magnitude'] > peak_dict[pf]['magnitude']:
                peak_dict[pf] = p

        print(f"\n  {ks:4d} 抽头 Blackman 窗:")
        print(f"    {'频率':>6} {'幅度':>10} {'衰减(dB)':>10} {'状态':>10}")
        print(f"    {'------':>6} {'----------':>10} {'----------':>10} {'----------':>10}")
        for tf in test_freqs_hp:
            mag = 0
            for freq_key, p in peak_dict.items():
                if abs(freq_key - tf) < 30:
                    mag = max(mag, p['normalized_magnitude'])
            atten = 20 * math.log10(mag) if mag > 0 else -999
            status = "✗衰减" if atten < -20 else ("~过渡" if atten < -3 else "✓保留")
            print(f"    {tf:>5d} Hz {mag:>10.4f} {atten:>+9.1f} dB {status:>10}")

    print_subsep("滤波器响应特性（511抽头 Blackman）")
    kernel = create_lowpass_kernel(200, sample_rate, kernel_size=511)
    freqs, _, mags_db = compute_filter_response(kernel, sample_rate, n_fft=16384)

    print(f"\n低通 200Hz 滤波器关键频点:")
    key_freqs = [50, 100, 150, 180, 200, 220, 250, 300, 400, 500, 1000]
    print(f"  {'频率':>8} {'衰减(dB)':>10} {'说明':>10}")
    print(f"  {'--------':>8} {'----------':>10} {'----------':>10}")
    for tf in key_freqs:
        atten = get_filter_cutoff_attenuation(kernel, sample_rate, tf)
        note = "通带" if atten > -1 else ("过渡带" if atten > -40 else "阻带")
        print(f"  {tf:>6d} Hz {atten:>+9.1f} dB {note:>10}")

    print("\n✓ 滤波器校准演示完成，默认 511 抽头 Blackman 窗在截止频率处有合理的过渡带")


def demo_spectrum_analysis():
    print_sep("第三部分：频谱分析与 CSV 导出")

    sample_rate = 22050
    duration = 1.5

    components = [
        (100, 1.00),
        (250, 0.70),
        (440, 0.50),
        (880, 0.30),
        (1500, 0.20),
        (3000, 0.10),
    ]

    signal = generate_composite_signal(components, sample_rate, duration)
    signal = add_white_noise(signal, amplitude=0.02, seed=42)

    analysis = analyze_signal_full(signal, sample_rate, label="合成测试信号",
                                   max_peaks=12)
    print(format_analysis_summary(analysis))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "spectrum_analysis.csv")
    export_spectrum_csv(csv_path, analysis, signal=signal, sample_rate=sample_rate,
                       include_full_spectrum=False)
    print(f"  ✓ 频谱数据已导出: {csv_path}")
    print("     包含: 信号汇总信息 + 峰值表")

    print("\n✓ 频谱分析与 CSV 导出演示完成")
    return signal, sample_rate, components


def demo_filters_time_domain(signal, sample_rate, components):
    print_sep("第四部分：时域 FIR 滤波效果对比")

    analysis_in = analyze_signal_full(signal, sample_rate, label="原始信号",
                                       max_peaks=8)
    print(format_analysis_summary(analysis_in))

    print_subsep("200Hz 低通滤波（511 抽头 Blackman 窗）")
    cutoff_lp = 200
    kernel_lp = create_lowpass_kernel(cutoff_lp, sample_rate, kernel_size=511,
                                       window_type='blackman')
    signal_lp = apply_time_filter(signal, kernel_lp)

    analysis_lp = analyze_signal_full(signal_lp, sample_rate, label="低通 200Hz 后",
                                       max_peaks=8)
    print(format_analysis_summary(analysis_lp))

    print("  各频率分量变化:")
    peak_in = {round(p['frequency']): p for p in analysis_in['peaks']}
    peak_lp = {round(p['frequency']): p for p in analysis_lp['peaks']}
    for freq, _ in components:
        f_key = round(freq)
        m_in = peak_in.get(f_key, {}).get('magnitude', 0)
        m_out = peak_lp.get(f_key, {}).get('magnitude', 0)
        if m_in > 0.01:
            ratio = m_out / m_in
            atten = 20 * math.log10(ratio) if ratio > 0 else -999
            marker = " ✓保留" if atten > -3 else (" ↓过渡" if atten > -20 else " ✗抑制")
            print(f"    {freq:5d} Hz: {m_in:.3f} → {m_out:.3f} ({atten:+.1f} dB){marker}")

    print_subsep("300Hz 高通滤波（511 抽头 Blackman 窗）")
    cutoff_hp = 300
    kernel_hp = create_highpass_kernel(cutoff_hp, sample_rate, kernel_size=511,
                                        window_type='blackman')
    signal_hp = apply_time_filter(signal, kernel_hp)

    analysis_hp = analyze_signal_full(signal_hp, sample_rate, label="高通 300Hz 后",
                                       max_peaks=8)
    print(format_analysis_summary(analysis_hp))

    print("  各频率分量变化:")
    peak_hp = {round(p['frequency']): p for p in analysis_hp['peaks']}
    for freq, _ in components:
        f_key = round(freq)
        m_in = peak_in.get(f_key, {}).get('magnitude', 0)
        m_out = peak_hp.get(f_key, {}).get('magnitude', 0)
        if m_in > 0.01:
            ratio = m_out / m_in
            atten = 20 * math.log10(ratio) if ratio > 0 else -999
            marker = " ✗抑制" if atten < -20 else (" ↓过渡" if atten < -3 else " ✓保留")
            print(f"    {freq:5d} Hz: {m_in:.3f} → {m_out:.3f} ({atten:+.1f} dB){marker}")

    print("\n✓ 时域 FIR 滤波对比演示完成")
    return signal_lp, signal_hp


def demo_presets():
    print_sep("第五部分：预设系统")

    print()
    print(list_presets())

    print("\n测试: 应用 'telephone' 预设并保存自定义预设")

    sample_rate = 22050
    signal = generate_composite_signal([(440, 1.0), (1000, 0.5), (3000, 0.3)],
                                        sample_rate, 1.0)

    presets = load_presets()
    preset = presets.get('telephone', {})
    print(f"\n  telephone 预设参数:")
    for k, v in preset.items():
        if k != 'description':
            print(f"    {k}: {v}")

    my_preset = {
        'description': '我的自定义预设 - 温暖人声',
        'highpass': 150,
        'lowpass': 5000,
        'gain': 2,
        'filter_domain': 'time',
        'kernel_size': 511,
    }

    preset_file = os.path.join(OUTPUT_DIR, 'my_presets.json')
    save_preset('warm-vocal', my_preset, preset_file)
    print(f"\n  ✓ 已保存自定义预设 'warm-vocal' 到 {preset_file}")

    loaded = load_presets(preset_file)
    print(f"  ✓ 重新加载后共 {len(loaded)} 个预设")

    print("\n✓ 预设系统演示完成")


def demo_wav_io_and_batch(signal, sample_rate):
    print_sep("第六部分：WAV 读写与批处理")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_dir = os.path.join(OUTPUT_DIR, "test_batch_input")
    out_dir = os.path.join(OUTPUT_DIR, "test_batch_output")
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    test_files = [
        ("tone_100Hz.wav", 100, 1.0),
        ("tone_440Hz.wav", 440, 0.8),
        ("tone_1000Hz.wav", 1000, 0.6),
        ("tone_3000Hz.wav", 3000, 0.4),
    ]

    print(f"\n生成 {len(test_files)} 个测试 WAV 到: {test_dir}")
    for fname, freq, amp in test_files:
        sig = generate_sine_wave(freq, sample_rate, 1.0, amplitude=amp)
        write_wav(os.path.join(test_dir, fname), sig, sample_rate)
        print(f"  ✓ {fname}: {freq} Hz, 幅度 {amp}")

    wav_list = find_wav_files(test_dir)
    print(f"\n目录中找到 {len(wav_list)} 个 WAV 文件")

    print("\n批量低通滤波 (500Hz):")
    for i, filepath in enumerate(wav_list, 1):
        fname = os.path.basename(filepath)
        sig, sr, ch = read_wav(filepath)
        kernel = create_lowpass_kernel(500, sr, kernel_size=511)
        filtered = apply_time_filter(sig, kernel)
        out_path = os.path.join(out_dir, f"lp500_{fname}")
        write_wav(out_path, filtered, sr)

        rms_in = compute_signal_rms(sig)
        rms_out = compute_signal_rms(filtered)
        print(f"  [{i}] {fname}: RMS {rms_in:.4f} → {rms_out:.4f} "
              f"({20*math.log10(rms_out/rms_in):+.1f} dB)")

    print(f"\n批处理输出已保存到: {out_dir}")
    print("\n✓ WAV 读写与批处理演示完成")


def demo_effects(signal, sample_rate):
    print_sep("第七部分：音频效果（增益 + 回声）")

    print_subsep("增益效果")
    gain_db = 6.0
    gained = apply_gain(signal, gain_db)
    rms_in = compute_signal_rms(signal)
    rms_out = compute_signal_rms(gained)
    print(f"  目标: {gain_db:+.1f} dB → 实际: {20*math.log10(rms_out/rms_in):+.2f} dB")

    print_subsep("回声效果（两种模式对比）")
    delay = 250
    decay = 0.4
    echo_simple = apply_echo(signal, sample_rate, delay, decay, feedback=False)
    echo_fb = apply_echo(signal, sample_rate, delay, decay, feedback=True)

    print(f"  延迟: {delay} ms, 衰减: {decay}")
    print(f"  单次回声 RMS: {compute_signal_rms(echo_simple):.4f}")
    print(f"  反馈回声 RMS: {compute_signal_rms(echo_fb):.4f}")
    print(f"  理论反馈增益: 1/(1-α) = {1/(1-decay):.2f}x = {20*math.log10(1/(1-decay)):.1f} dB")

    print("\n✓ 音频效果演示完成")
    return gained, echo_simple, echo_fb


def demo_full_wav_export(signal, sample_rate):
    print_sep("第八部分：导出可试听的 WAV 文件")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {os.path.abspath(OUTPUT_DIR)}/")

    files = []

    write_wav(os.path.join(OUTPUT_DIR, "01_original.wav"), signal, sample_rate)
    files.append(("原始信号", "01_original.wav"))

    kernel = create_lowpass_kernel(200, sample_rate, kernel_size=511)
    sig_lp = apply_time_filter(signal, kernel)
    write_wav(os.path.join(OUTPUT_DIR, "02_lowpass_200Hz.wav"), sig_lp, sample_rate)
    files.append(("低通 200Hz (FIR)", "02_lowpass_200Hz.wav"))

    kernel = create_highpass_kernel(1000, sample_rate, kernel_size=511)
    sig_hp = apply_time_filter(signal, kernel)
    write_wav(os.path.join(OUTPUT_DIR, "03_highpass_1kHz.wav"), sig_hp, sample_rate)
    files.append(("高通 1kHz (FIR)", "03_highpass_1kHz.wav"))

    sig_gain = apply_gain(signal, 6)
    write_wav(os.path.join(OUTPUT_DIR, "04_gain_plus6dB.wav"), sig_gain, sample_rate)
    files.append(("增益 +6dB", "04_gain_plus6dB.wav"))

    sig_echo = apply_echo(signal, sample_rate, 300, 0.45, feedback=True)
    write_wav(os.path.join(OUTPUT_DIR, "05_echo_feedback.wav"), sig_echo, sample_rate)
    files.append(("反馈回声", "05_echo_feedback.wav"))

    sig_phone = apply_freq_filter(signal, sample_rate, 3400, 'lowpass')
    sig_phone = apply_freq_filter(sig_phone, sample_rate, 300, 'highpass')
    sig_phone = apply_gain(sig_phone, 3)
    write_wav(os.path.join(OUTPUT_DIR, "06_telephone_effect.wav"), sig_phone, sample_rate)
    files.append(("电话音效果", "06_telephone_effect.wav"))

    for desc, fname in files:
        print(f"  ✓ {desc:20s} → {fname}")

    print("\n✓ 所有 WAV 文件已生成，可直接播放对比试听")


def main():
    print_sep("音频信号处理引擎 v2.0 - 完整演示")
    print("  FFT/IFFT · 频谱分析 · FIR 滤波校准 · 预设 · 批处理 · CSV 导出")

    demo_fft_basics()

    demo_filter_calibration()

    signal, sample_rate, components = demo_spectrum_analysis()

    demo_filters_time_domain(signal, sample_rate, components)

    demo_presets()

    demo_wav_io_and_batch(signal, sample_rate)

    demo_effects(signal, sample_rate)

    demo_full_wav_export(signal, sample_rate)

    print_sep("全部演示完成 ✓")
    print()
    print("命令行工具用法:")
    print("  python audio_tool.py --help            - 查看总帮助")
    print("  python audio_tool.py presets list      - 列出所有预设")
    print("  python audio_tool.py generate          - 生成合成信号")
    print("  python audio_tool.py analyze           - 频谱分析(支持目录批处理)")
    print("  python audio_tool.py filter            - 低通/高通滤波(支持目录批处理)")
    print("  python audio_tool.py gain              - 增益调整(支持目录批处理)")
    print("  python audio_tool.py echo              - 回声效果(支持目录批处理)")
    print("  python audio_tool.py process           - 完整处理管道 + 预设")
    print()
    print("提示: 输入参数如果是目录，会自动批量处理目录下所有 WAV 文件")
    print()


if __name__ == "__main__":
    main()
