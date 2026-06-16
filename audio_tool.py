import argparse
import sys
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
    read_wav,
    write_wav,
    compute_band_energy,
)


def print_sep(char="=", width=70):
    print(char * width)


def parse_freq_list(s):
    try:
        parts = s.split(",")
        result = []
        for p in parts:
            freq_amp = p.strip().split(":")
            if len(freq_amp) == 1:
                result.append((float(freq_amp[0]), 1.0))
            else:
                result.append((float(freq_amp[0]), float(freq_amp[1])))
        return result
    except Exception:
        raise argparse.ArgumentTypeError(
            "频率列表格式应为 'freq:amp,freq:amp,...'")


def analyze_signal(signal, sample_rate, label="信号", max_peaks=10,
                   min_freq_separation=None):
    print_sep("-")
    print(f"[{label}]")
    print_sep("-")

    n = len(signal)
    duration = n / sample_rate
    rms = compute_signal_rms(signal)
    peak = max(abs(s) for s in signal) if signal else 0

    print(f"  采样点数:     {n}")
    print(f"  时长:         {duration:.3f} 秒")
    print(f"  采样率:       {sample_rate} Hz")
    print(f"  RMS:          {rms:.6f}")
    print(f"  峰值幅度:     {peak:.6f}")
    if rms > 0:
        print(f"  峰值/RMS 比:  {peak / rms:.2f} ({20 * __import__('math').log10(peak / rms):.1f} dB)")

    freqs, mags, norm_mags, _ = compute_spectrum(signal, sample_rate)
    peaks = find_spectrum_peaks(freqs, norm_mags,
                            max_peaks=max_peaks,
                            min_freq_separation=min_freq_separation)
    print(format_peak_table(peaks, title=f"{label} 频谱峰值"))
    return peaks


def print_processing_summary(signal_in, signal_out, sample_rate, name):
    rms_in = compute_signal_rms(signal_in)
    rms_out = compute_signal_rms(signal_out)
    peak_in = max(abs(s) for s in signal_in) if signal_in else 0
    peak_out = max(abs(s) for s in signal_out) if signal_out else 0

    print(f"\n[{name}] 处理前后对比:")
    print(f"  输入 RMS:  {rms_in:.6f}  →  输出 RMS:  {rms_out:.6f}")
    if rms_in > 0:
        rms_change_db = 20 * __import__('math').log10(rms_out / rms_in)
        print(f"  RMS 变化: {rms_change_db:+.2f} dB")
    print(f"  输入峰值: {peak_in:.6f}  →  输出峰值: {peak_out:.6f}")


def run_generate(args):
    print_sep()
    print("  音频信号处理工具 - 生成合成信号")
    print_sep()

    components = parse_freq_list(args.frequencies)

    print(f"\n信号成分:")
    for freq, amp in components:
        print(f"  {freq:.1f} Hz, 幅度 {amp:.3f}")
    print(f"采样率: {args.sample_rate} Hz, 时长: {args.duration} s")

    signal = generate_composite_signal(components, args.sample_rate, args.duration)

    if args.noise > 0:
        signal = add_white_noise(signal, amplitude=args.noise, seed=42)
        print(f"已添加白噪声, 幅度 {args.noise}")

    analyze_signal(signal, args.sample_rate, "合成信号",
                  max_peaks=args.max_peaks)

    if args.output:
        write_wav(args.output, signal, args.sample_rate)
        print(f"\n已保存到: {args.output}")

    return signal, args.sample_rate


def run_analyze(args):
    print_sep()
    print("  音频信号处理工具 - 频谱分析")
    print_sep()

    signal, sample_rate, _ = read_wav(args.input)
    print(f"\n已读入: {args.input}")
    analyze_signal(signal, sample_rate, "输入信号",
                  max_peaks=args.max_peaks)
    return signal, sample_rate


def run_filter(args):
    import math

    print_sep()
    print(f"  音频信号处理工具 - {'低通' if args.type == 'lowpass' else '高通'}滤波")
    print_sep()

    signal, sample_rate, channels = read_wav(args.input)
    print(f"\n已读入: {args.input}")

    analyze_signal(signal, sample_rate, "滤波前", max_peaks=args.max_peaks)

    print(f"\n滤波器参数:")
    print(f"  类型:       {'低通' if args.type == 'lowpass' else '高通'}")
    print(f"  截止频率:    {args.cutoff} Hz")
    print(f"  实现方式:    {'频域(FFT)' if args.domain == 'freq' else '时域(FIR)'}")
    if args.domain == 'time':
        print(f"  滤波器阶数:  {args.kernel_size} 抽头")

    if args.type == 'lowpass':
        if args.domain == 'freq':
            filtered = apply_freq_filter(signal, sample_rate, args.cutoff, 'lowpass')
        else:
            kernel = create_lowpass_kernel(args.cutoff, sample_rate,
                                            kernel_size=args.kernel_size)
            filtered = apply_time_filter(signal, kernel)
    else:
        if args.domain == 'freq':
            filtered = apply_freq_filter(signal, sample_rate, args.cutoff, 'highpass')
        else:
            kernel = create_highpass_kernel(args.cutoff, sample_rate,
                                          kernel_size=args.kernel_size)
            filtered = apply_time_filter(signal, kernel)

    print_processing_summary(signal, filtered, sample_rate, "滤波")

    analyze_signal(filtered, sample_rate, "滤波后",
                  max_peaks=args.max_peaks)

    if args.cutoff > 0 and args.type == 'lowpass':
        e_low = compute_band_energy(signal, sample_rate, args.cutoff + 1, sample_rate / 2)
        e_low_out = compute_band_energy(filtered, sample_rate, args.cutoff + 1, sample_rate / 2)
        print(f"\n截止频率以上能量占比:")
        print(f"  输入: {e_low * 100:.2f}%  →  输出: {e_low_out * 100:.2f}%")
        if e_low > 0:
            print(f"  抑制: {-10 * math.log10(max(e_low_out, 1e-10) / max(e_low, 1e-10)):.1f} dB")

    if args.cutoff > 0 and args.type == 'highpass':
        e_high = compute_band_energy(signal, sample_rate, 0, args.cutoff)
        e_high_out = compute_band_energy(filtered, sample_rate, 0, args.cutoff)
        print(f"\n截止频率以下能量占比:")
        print(f"  输入: {e_high * 100:.2f}%  →  输出: {e_high_out * 100:.2f}%")
        if e_high > 0:
            print(f"  抑制: {-10 * math.log10(max(e_high_out, 1e-10) / max(e_high, 1e-10)):.1f} dB")

    if args.output:
        write_wav(args.output, filtered, sample_rate, n_channels=channels)
        print(f"\n已保存到: {args.output}")

    return filtered, sample_rate


def run_gain(args):
    print_sep()
    print("  音频信号处理工具 - 增益调整")
    print_sep()

    signal, sample_rate, channels = read_wav(args.input)
    print(f"\n已读入: {args.input}")

    analyze_signal(signal, sample_rate, "处理前", max_peaks=args.max_peaks)

    print(f"\n增益: {args.gain:+.2f} dB")

    gained = apply_gain(signal, args.gain)

    print_processing_summary(signal, gained, sample_rate, "增益")

    analyze_signal(gained, sample_rate, "处理后", max_peaks=args.max_peaks)

    if args.output:
        write_wav(args.output, gained, sample_rate, n_channels=channels)
        print(f"\n已保存到: {args.output}")

    return gained, sample_rate


def run_echo(args):
    print_sep()
    print("  音频信号处理工具 - 回声效果")
    print_sep()

    signal, sample_rate, channels = read_wav(args.input)
    print(f"\n已读入: {args.input}")

    analyze_signal(signal, sample_rate, "处理前", max_peaks=args.max_peaks)

    print(f"\n回声参数:")
    print(f"  延迟时间: {args.delay} ms")
    print(f"  衰减系数: {args.decay}")
    print(f"  反馈模式: {'开启' if args.feedback else '关闭'}")

    echoed = apply_echo(signal, sample_rate, args.delay, args.decay,
                          feedback=args.feedback)

    print_processing_summary(signal, echoed, sample_rate, "回声")

    analyze_signal(echoed, sample_rate, "处理后", max_peaks=args.max_peaks)

    if args.output:
        write_wav(args.output, echoed, sample_rate, n_channels=channels)
        print(f"\n已保存到: {args.output}")

    return echoed, sample_rate


def run_process(args):
    import math

    print_sep()
    print("  音频信号处理工具 - 完整处理流程")
    print_sep()

    sample_rate = args.sample_rate
    channels = 1
    if args.input:
        signal, sample_rate, channels = read_wav(args.input)
        print(f"\n已读入: {args.input}")
    else:
        components = parse_freq_list(args.frequencies)
        print(f"\n生成合成信号:")
        for freq, amp in components:
            print(f"  {freq:.1f} Hz, 幅度 {amp:.3f}")
        signal = generate_composite_signal(components, args.sample_rate, args.duration)
        if args.noise > 0:
            signal = add_white_noise(signal, amplitude=args.noise, seed=42)
            print(f"已添加白噪声, 幅度 {args.noise}")

    print()
    analyze_signal(signal, sample_rate, "原始信号", max_peaks=args.max_peaks)

    current_signal = signal

    if args.lowpass is not None:
        print(f"\n--- 应用低通滤波 (截止 {args.lowpass} Hz) ---")
        if args.filter_domain == 'freq':
            current_signal = apply_freq_filter(current_signal, sample_rate,
                                               args.lowpass, 'lowpass')
        else:
            kernel = create_lowpass_kernel(args.lowpass, sample_rate,
                                            kernel_size=args.kernel_size)
            current_signal = apply_time_filter(current_signal, kernel)
        analyze_signal(current_signal, sample_rate, "低通滤波后",
                      max_peaks=args.max_peaks)

    if args.highpass is not None:
        print(f"\n--- 应用高通滤波 (截止 {args.highpass} Hz) ---")
        if args.filter_domain == 'freq':
            current_signal = apply_freq_filter(current_signal, sample_rate,
                                               args.highpass, 'highpass')
        else:
            kernel = create_highpass_kernel(args.highpass, sample_rate,
                                          kernel_size=args.kernel_size)
            current_signal = apply_time_filter(current_signal, kernel)
        analyze_signal(current_signal, sample_rate, "高通滤波后",
                        max_peaks=args.max_peaks)

    if args.gain != 0:
        print(f"\n--- 应用增益 ({args.gain:+.2f} dB) ---")
        current_signal = apply_gain(current_signal, args.gain)
        analyze_signal(current_signal, sample_rate, "增益后",
                      max_peaks=args.max_peaks)

    if args.echo_delay is not None:
        print(f"\n--- 应用回声 (延迟 {args.echo_delay} ms, 衰减 {args.echo_decay}) ---")
        current_signal = apply_echo(current_signal, sample_rate,
                                  args.echo_delay, args.echo_decay,
                                  feedback=args.echo_feedback)
        analyze_signal(current_signal, sample_rate, "回声后",
                      max_peaks=args.max_peaks)

    print()
    print_processing_summary(signal, current_signal, sample_rate, "完整处理")

    if args.output:
        write_wav(args.output, current_signal, sample_rate, n_channels=channels)
        print(f"\n已保存到: {args.output}")

    return current_signal, sample_rate


def main():
    parser = argparse.ArgumentParser(
        description="音频信号处理工具 (Audio Signal Processor)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 生成合成信号并导出
  python audio_tool.py generate -f "50:1,120:0.7,250:0.5,500:0.3,800:0.2" -o test.wav

  # 分析 WAV 文件频谱
  python audio_tool.py analyze -i test.wav --max-peaks 8

  # 低通滤波 (频域)
  python audio_tool.py filter -i test.wav -t lowpass -c 200 -o lp_out.wav

  # 高通滤波 (时域 FIR)
  python audio_tool.py filter -i test.wav -t highpass -c 300 -d time -o hp_out.wav

  # 加 6dB 增益
  python audio_tool.py gain -i test.wav -g 6 -o gain_out.wav

  # 加回声
  python audio_tool.py echo -i test.wav --delay 200 --decay 0.5 --feedback -o echo_out.wav

  # 完整处理管道
  python audio_tool.py process -f "50:1,250:0.5,800:0.3" --lowpass 300 --gain 3 -o out.wav
""")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ============ generate ============
    p_gen = subparsers.add_parser("generate", help="生成合成信号")
    p_gen.add_argument("-f", "--frequencies", required=True,
                        help="频率列表 'freq:amp,freq:amp,... (amp 可省略, 默认 1.0")
    p_gen.add_argument("-s", "--sample-rate", type=int, default=22050,
                        help="采样率 (默认 22050 Hz)")
    p_gen.add_argument("-d", "--duration", type=float, default=2.0,
                        help="时长秒数 (默认 2.0)")
    p_gen.add_argument("-n", "--noise", type=float, default=0.0,
                        help="白噪声幅度 (默认 0)")
    p_gen.add_argument("-o", "--output", type=str, default=None,
                        help="输出 WAV 文件路径")
    p_gen.add_argument("--max-peaks", type=int, default=10,
                         help="显示峰值数量 (默认 10)")

    # ============ analyze ============
    p_ana = subparsers.add_parser("analyze", help="频谱分析")
    p_ana.add_argument("-i", "--input", required=True, help="输入 WAV 文件")
    p_ana.add_argument("--max-peaks", type=int, default=10,
                         help="显示峰值数量 (默认 10)")

    # ============ filter ============
    p_filt = subparsers.add_parser("filter", help="低通/高通滤波")
    p_filt.add_argument("-i", "--input", required=True, help="输入 WAV 文件")
    p_filt.add_argument("-t", "--type", choices=["lowpass", "highpass"],
                         required=True, help="滤波器类型")
    p_filt.add_argument("-c", "--cutoff", type=float, required=True,
                         help="截止频率 (Hz)")
    p_filt.add_argument("-d", "--domain", choices=["freq", "time"],
                         default="freq", help="实现方式 (默认 freq)")
    p_filt.add_argument("--kernel-size", type=int, default=127,
                         help="FIR 核大小 (默认 127)")
    p_filt.add_argument("-o", "--output", type=str, default=None,
                         help="输出 WAV 文件路径")
    p_filt.add_argument("--max-peaks", type=int, default=10,
                         help="显示峰值数量 (默认 10)")

    # ============ gain ============
    p_gain = subparsers.add_parser("gain", help="增益调整")
    p_gain.add_argument("-i", "--input", required=True, help="输入 WAV 文件")
    p_gain.add_argument("-g", "--gain", type=float, required=True,
                         help="增益 dB (dB 正数放大, 负数衰减)")
    p_gain.add_argument("-o", "--output", type=str, default=None,
                         help="输出 WAV 文件路径")
    p_gain.add_argument("--max-peaks", type=int, default=10,
                         help="显示峰值数量 (默认 10)")

    # ============ echo ============
    p_echo = subparsers.add_parser("echo", help="回声效果")
    p_echo.add_argument("-i", "--input", required=True, help="输入 WAV 文件")
    p_echo.add_argument("--delay", type=float, default=200, help="延迟时间 ms (默认 200)")
    p_echo.add_argument("--decay", type=float, default=0.5,
                          help="衰减系数 0-1 (默认 0.5)")
    p_echo.add_argument("--feedback", action="store_true", help="启用反馈回声")
    p_echo.add_argument("-o", "--output", type=str, default=None,
                         help="输出 WAV 文件路径")
    p_echo.add_argument("--max-peaks", type=int, default=10,
                         help="显示峰值数量 (默认 10)")

    # ============ process ============
    p_proc = subparsers.add_parser("process", help="完整处理流程")
    src_group = p_proc.add_mutually_exclusive_group(required=True)
    src_group.add_argument("-i", "--input", type=str, help="输入 WAV 文件")
    src_group.add_argument("-f", "--frequencies", type=str,
                            help="频率列表 (同 generate 子命令)")
    p_proc.add_argument("-s", "--sample-rate", type=int, default=22050,
                          help="采样率 (用于生成信号, 默认 22050)")
    p_proc.add_argument("-d", "--duration", type=float, default=2.0,
                          help="生成信号时长 (默认 2 秒)")
    p_proc.add_argument("-n", "--noise", type=float, default=0.0, help="白噪声幅度")
    p_proc.add_argument("--lowpass", type=float, default=None,
                          help="低通截止频率 Hz")
    p_proc.add_argument("--highpass", type=float, default=None,
                          help="高通截止频率 Hz")
    p_proc.add_argument("--filter-domain", choices=["freq", "time"],
                         default="freq", help="滤波实现方式")
    p_proc.add_argument("--kernel-size", type=int, default=127,
                         help="FIR 核大小")
    p_proc.add_argument("-g", "--gain", type=float, default=0,
                         help="增益 dB")
    p_proc.add_argument("--echo-delay", type=float, default=None,
                          help="回声延迟 ms")
    p_proc.add_argument("--echo-decay", type=float, default=0.5,
                         help="回声衰减系数")
    p_proc.add_argument("--echo-feedback", action="store_true",
                         help="回声反馈模式")
    p_proc.add_argument("-o", "--output", type=str, default=None,
                         help="输出 WAV 文件路径")
    p_proc.add_argument("--max-peaks", type=int, default=10,
                         help="显示峰值数量")

    args = parser.parse_args()

    if args.command:
        handlers = {
            "generate": run_generate,
            "analyze": run_analyze,
            "filter": run_filter,
            "gain": run_gain,
            "echo": run_echo,
            "process": run_process,
        }
        handler = handlers.get(args.command)
        if handler:
            try:
                handler(args)
            except FileNotFoundError as e:
                print(f"错误: 文件未找到: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
