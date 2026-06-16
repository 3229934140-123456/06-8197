import argparse
import sys
import os
import math
import csv

from audio_engine import (
    generate_composite_signal,
    add_white_noise,
    compute_spectrum,
    find_spectrum_peaks,
    format_peak_table,
    compute_signal_rms,
    compute_signal_peak,
    apply_gain,
    apply_echo,
    apply_freq_filter,
    apply_time_filter,
    create_lowpass_kernel,
    create_highpass_kernel,
    read_wav,
    write_wav,
    compute_band_energy,
    analyze_signal_full,
    format_analysis_summary,
    export_spectrum_csv,
    find_wav_files,
    load_presets,
    save_preset,
    list_presets,
    compute_filter_response,
    get_filter_cutoff_attenuation,
    show_preset_details,
    copy_preset,
    format_batch_summary,
    write_batch_manifest,
)


def print_sep(char="=", width=72):
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


def apply_preset_to_args(args, preset_name, presets, overwrite=False):
    if preset_name not in presets:
        print(f"错误: 预设 '{preset_name}' 不存在")
        print(list_presets(presets))
        sys.exit(1)

    preset = presets[preset_name]
    print(f"已应用预设: {preset_name} - {preset.get('description', '')}")
    if overwrite:
        print("  (模式: 完整覆盖默认参数)")
    else:
        print("  (模式: 仅填充未设置的参数)")

    applied_keys = []
    for key, value in preset.items():
        if key == 'description':
            continue
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if overwrite:
            if current is None or (isinstance(current, (int, float)) and (key == 'gain' and current == 0)):
                setattr(args, key, value)
                applied_keys.append(key)
            elif key in ('echo_decay', 'filter_domain', 'window'):
                setattr(args, key, value)
                applied_keys.append(key)
        else:
            is_default = (current is None)
            if key == 'gain' and current == 0:
                is_default = True
            if key == 'echo_decay' and current == 0.5:
                is_default = True
            if key == 'filter_domain' and current == 'time':
                is_default = True
            if key == 'window' and current == 'blackman':
                is_default = True
            if is_default:
                setattr(args, key, value)
                applied_keys.append(key)

    if applied_keys:
        print(f"  应用参数: {', '.join(applied_keys)}")

    return args


def describe_process_steps(args):
    steps = []
    if hasattr(args, 'type') and hasattr(args, 'cutoff'):
        prefix = '低通' if args.type == 'lowpass' else '高通'
        ks = getattr(args, 'kernel_size', None)
        ks_suffix = f"({ks}抽头)" if ks and getattr(args, 'domain', 'time') == 'time' else ''
        steps.append(f"{prefix}{int(args.cutoff)}Hz{ks_suffix}")
    else:
        if getattr(args, 'highpass', None) is not None:
            steps.append(f"高通{int(args.highpass)}Hz")
        if getattr(args, 'lowpass', None) is not None:
            steps.append(f"低通{int(args.lowpass)}Hz")
    if hasattr(args, 'gain') and getattr(args, 'gain', 0) != 0:
        steps.append(f"增益{args.gain:+.1f}dB")
    echo_delay = getattr(args, 'delay', None) if hasattr(args, 'delay') else getattr(args, 'echo_delay', None)
    if echo_delay is not None:
        echo_decay = getattr(args, 'decay', 0.5) if hasattr(args, 'decay') else getattr(args, 'echo_decay', 0.5)
        fb = getattr(args, 'feedback', False) if hasattr(args, 'feedback') else getattr(args, 'echo_feedback', False)
        steps.append(f"回声{int(echo_delay)}ms×{echo_decay:.2f}{'+FB' if fb else ''}")
    return steps


def print_dry_run_plan(wav_files, output_dir, steps_str, naming_pattern="{fname}"):
    print()
    print_sep("-")
    print("  Dry-Run 执行计划 (不实际写文件)")
    print_sep("-")
    print(f"  处理文件数: {len(wav_files)}")
    print(f"  输出目录:   {output_dir}")
    print(f"  处理步骤:   {steps_str if steps_str else '(无处理)'}")
    print(f"  命名模式:   {naming_pattern}")
    print()
    print(f"  {'#':>3}  {'原文件名':<30}  ->  输出文件名")
    print(f"  {'-':>3}  {'-':-<30}      {'-':-<30}")
    for i, fp in enumerate(wav_files, 1):
        fname = os.path.basename(fp)
        base, ext = os.path.splitext(fname)
        out_name = naming_pattern.format(fname=fname, base=base, ext=ext, ext_no_dot=ext[1:])
        print(f"  {i:>3}  {fname:<30}  ->  {out_name}")
    print()
    print("  如需执行，请移除 --dry-run 参数")
    print_sep("-")


def build_output_filename(fname, args, op_type=None):
    base, ext = os.path.splitext(fname)
    prefix_parts = []
    if op_type == 'filter':
        ftype_label = 'lp' if args.type == 'lowpass' else 'hp'
        prefix_parts.append(f"{ftype_label}_{int(args.cutoff)}Hz")
    elif op_type == 'gain':
        prefix_parts.append(f"g{args.gain:+.0f}dB")
    elif op_type == 'echo':
        fb = 'fb' if args.feedback else 's'
        prefix_parts.append(f"echo_{int(args.delay)}ms{fb}")
    elif op_type == 'process':
        if getattr(args, 'highpass', None):
            prefix_parts.append(f"hp{int(args.highpass)}")
        if getattr(args, 'lowpass', None):
            prefix_parts.append(f"lp{int(args.lowpass)}")
        if getattr(args, 'gain', 0) != 0:
            prefix_parts.append(f"g{args.gain:+.0f}dB")
        if getattr(args, 'echo_delay', None):
            prefix_parts.append(f"echo")
    if getattr(args, 'preserve_name', True):
        return fname
    if prefix_parts:
        return f"{'_'.join(prefix_parts)}_{fname}"
    return fname


def finalize_batch_results(results, output_dir, extra_info=None, manifest_name='manifest.csv'):
    print(format_batch_summary(results))
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, manifest_name)
    try:
        write_batch_manifest(manifest_path, results, extra_info=extra_info)
        print(f"\n处理清单已保存到: {manifest_path}")
    except Exception as e:
        print(f"\n警告: 保存清单失败: {e}")
    success_count = sum(1 for r in results if r.get('status') in ('成功', 'OK'))
    total_count = len(results)
    print(f"\n共成功处理 {success_count}/{total_count} 个文件")


def run_analysis_on_file(filepath, max_peaks=10, label=None):
    signal, sample_rate, channels = read_wav(filepath)
    fname = label if label else os.path.basename(filepath)
    analysis = analyze_signal_full(signal, sample_rate, label=fname,
                                   max_peaks=max_peaks)
    analysis['filename'] = os.path.basename(filepath)
    analysis['filepath'] = filepath
    analysis['channels'] = channels
    return analysis, signal, sample_rate, channels


def run_generate(args):
    print_sep()
    print("  音频信号处理工具 - 生成合成信号")
    print_sep()

    components = parse_freq_list(args.frequencies)

    print(f"\n信号成分:")
    for freq, amp in components:
        print(f"  {freq:.1f} Hz, 幅度 {amp:.3f}")
    print(f"采样率: {args.sample_rate} Hz, 时长: {args.duration} s")
    if args.noise > 0:
        print(f"白噪声幅度: {args.noise}")

    signal = generate_composite_signal(components, args.sample_rate, args.duration)

    if args.noise > 0:
        signal = add_white_noise(signal, amplitude=args.noise, seed=42)

    analysis = analyze_signal_full(signal, args.sample_rate, label="合成信号",
                                   max_peaks=args.max_peaks)
    analysis['filename'] = '(合成)'
    print(format_analysis_summary(analysis))

    if args.output:
        write_wav(args.output, signal, args.sample_rate)
        print(f"\n已保存到: {args.output}")

    if args.csv:
        export_spectrum_csv(args.csv, analysis, signal=signal,
                           sample_rate=args.sample_rate,
                           include_full_spectrum=args.full_csv)
        print(f"频谱数据已导出到: {args.csv}")

    return signal, args.sample_rate


def run_analyze(args):
    print_sep()
    print("  音频信号处理工具 - 频谱分析")
    print_sep()

    if os.path.isdir(args.input):
        return run_batch_analyze(args)

    print(f"\n分析文件: {args.input}")

    analysis, signal, sample_rate, channels = run_analysis_on_file(
        args.input, max_peaks=args.max_peaks)

    print(format_analysis_summary(analysis))

    if args.csv:
        export_spectrum_csv(args.csv, analysis, signal=signal,
                           sample_rate=sample_rate,
                           include_full_spectrum=args.full_csv)
        print(f"\n频谱数据已导出到: {args.csv}")

    return signal, sample_rate


def run_batch_analyze(args):
    input_dir = args.input
    output_dir = args.output if args.output else os.path.join(input_dir, 'analysis_out')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = find_wav_files(input_dir)
    if not wav_files:
        print(f"错误: 目录中没有找到 WAV 文件: {input_dir}")
        sys.exit(1)

    steps_str = "频谱分析"

    if getattr(args, 'dry_run', False):
        print_dry_run_plan(wav_files, output_dir, steps_str)
        return None, None

    print(f"\n批量分析目录: {input_dir}")
    print(f"找到 {len(wav_files)} 个 WAV 文件")
    print(f"输出目录: {output_dir}")

    results = []
    for i, filepath in enumerate(wav_files, 1):
        fname = os.path.basename(filepath)
        print(f"\n[{i}/{len(wav_files)}] 处理: {fname}")
        result = {
            'filename': fname,
            'input_path': os.path.abspath(filepath),
            'output_path': '',
            'steps': steps_str,
            'status': '处理中',
        }
        try:
            analysis, signal, sr, ch = run_analysis_on_file(
                filepath, max_peaks=args.max_peaks, label=fname)
            result.update({
                'sample_rate': sr,
                'duration': analysis['duration'],
                'channels': ch,
                'rms_in': analysis['rms'],
                'rms_out': analysis['rms'],
                'rms_change_db': 0.0,
                'peak_in': analysis['peak'],
                'peak_out': analysis['peak'],
                'dominant_in': round(analysis['dominant_freq'], 2),
                'dominant_out': round(analysis['dominant_freq'], 2),
            })

            csv_name = os.path.splitext(fname)[0] + '.csv'
            csv_path = os.path.join(output_dir, csv_name)
            result['output_path'] = os.path.abspath(csv_path)
            if getattr(args, 'csv_each', False):
                export_spectrum_csv(csv_path, analysis, signal=signal,
                                   sample_rate=sr,
                                   include_full_spectrum=getattr(args, 'full_csv', False))
            result['status'] = '成功'
        except Exception as e:
            result['status'] = '失败'
            result['error_msg'] = str(e)
            print(f"  跳过 (错误: {e})")
        results.append(result)

    if results:
        summary_csv = os.path.join(output_dir, 'summary.csv')
        with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '采样率', '采样点数', '时长(s)', 'RMS',
                           '峰值', '峰值(dB)', '峰值/RMS比', '主导频率(Hz)',
                           '低频能量%', '中频能量%', '高频能量%'])
            for i, r in enumerate(results):
                if r['status'] == '成功':
                    _, signal, sr, _ = run_analysis_on_file(wav_files[i], max_peaks=args.max_peaks)
                    a = analyze_signal_full(signal, sr)
                    writer.writerow([
                        r['filename'], r['sample_rate'], a['n_samples'],
                        f"{a['duration']:.4f}", f"{a['rms']:.6f}",
                        f"{a['peak']:.6f}", f"{a['peak_db']:.2f}",
                        f"{a['crest_factor']:.4f}", f"{a['dominant_freq']:.2f}",
                        f"{a['band_energy_low']*100:.2f}",
                        f"{a['band_energy_mid']*100:.2f}",
                        f"{a['band_energy_high']*100:.2f}",
                    ])
                else:
                    writer.writerow([r['filename']] + [''] * 11)
        print(f"\n汇总表已保存到: {summary_csv}")

    extra_info = {'任务类型': '频谱分析', '输入目录': input_dir, '输出目录': output_dir}
    finalize_batch_results(results, output_dir, extra_info=extra_info)
    return None, None


def run_filter(args):
    print_sep()
    print(f"  音频信号处理工具 - {'低通' if args.type == 'lowpass' else '高通'}滤波")
    print_sep()

    if os.path.isdir(args.input):
        return run_batch_filter(args)

    signal, sample_rate, channels = read_wav(args.input)
    print(f"\n输入文件: {args.input}")

    analysis_in = analyze_signal_full(signal, sample_rate, label="滤波前",
                                       max_peaks=args.max_peaks)
    print(format_analysis_summary(analysis_in))

    print(f"\n滤波器参数:")
    print(f"  类型:       {'低通' if args.type == 'lowpass' else '高通'}")
    print(f"  截止频率:    {args.cutoff:.1f} Hz")
    print(f"  实现方式:    {'频域(FFT)' if args.domain == 'freq' else '时域(FIR)'}")
    if args.domain == 'time':
        ks = args.kernel_size if args.kernel_size else 511
        print(f"  滤波器阶数:  {ks} 抽头")
        print(f"  窗函数:      {args.window}")

    if args.type == 'lowpass':
        if args.domain == 'freq':
            filtered = apply_freq_filter(signal, sample_rate, args.cutoff, 'lowpass')
        else:
            kernel = create_lowpass_kernel(args.cutoff, sample_rate,
                                            kernel_size=args.kernel_size,
                                            window_type=args.window)
            filtered = apply_time_filter(signal, kernel)
    else:
        if args.domain == 'freq':
            filtered = apply_freq_filter(signal, sample_rate, args.cutoff, 'highpass')
        else:
            kernel = create_highpass_kernel(args.cutoff, sample_rate,
                                          kernel_size=args.kernel_size,
                                          window_type=args.window)
            filtered = apply_time_filter(signal, kernel)

    analysis_out = analyze_signal_full(filtered, sample_rate, label="滤波后",
                                        max_peaks=args.max_peaks)
    print(format_analysis_summary(analysis_out))

    print()
    print_sep("-")
    print("  处理前后对比")
    print_sep("-")
    print(f"  RMS:     {analysis_in['rms']:.6f}  →  {analysis_out['rms']:.6f}")
    if analysis_in['rms'] > 0:
        rms_db = 20 * math.log10(analysis_out['rms'] / analysis_in['rms'])
        print(f"  RMS变化:  {rms_db:+.2f} dB")
    print(f"  峰值:    {analysis_in['peak']:.6f}  →  {analysis_out['peak']:.6f}")

    if args.type == 'lowpass':
        e_above_in = compute_band_energy(signal, sample_rate, args.cutoff * 1.2,
                                         sample_rate / 2)
        e_above_out = compute_band_energy(filtered, sample_rate, args.cutoff * 1.2,
                                          sample_rate / 2)
        if e_above_in > 0:
            sup = -10 * math.log10(max(e_above_out, 1e-20) / max(e_above_in, 1e-20))
            print(f"  截止频率以上 ({args.cutoff*1.2:.0f}Hz+) 抑制: {sup:.1f} dB")
    else:
        e_below_in = compute_band_energy(signal, sample_rate, 0, args.cutoff * 0.8)
        e_below_out = compute_band_energy(filtered, sample_rate, 0, args.cutoff * 0.8)
        if e_below_in > 0:
            sup = -10 * math.log10(max(e_below_out, 1e-20) / max(e_below_in, 1e-20))
            print(f"  截止频率以下 (0~{args.cutoff*0.8:.0f}Hz) 抑制: {sup:.1f} dB")

    if args.output:
        write_wav(args.output, filtered, sample_rate, n_channels=channels)
        print(f"\n已保存到: {args.output}")

    if args.csv:
        analysis_out['filename'] = os.path.basename(args.output) if args.output else '(输出)'
        export_spectrum_csv(args.csv, analysis_out, signal=filtered,
                           sample_rate=sample_rate,
                           include_full_spectrum=args.full_csv)
        print(f"频谱数据已导出到: {args.csv}")

    return filtered, sample_rate


def run_batch_filter(args):
    input_dir = args.input
    output_dir = args.output if args.output else os.path.join(input_dir, 'filtered_out')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = find_wav_files(input_dir)
    if not wav_files:
        print(f"错误: 目录中没有找到 WAV 文件: {input_dir}")
        sys.exit(1)

    ftype_label = '低通' if args.type == 'lowpass' else '高通'
    steps_list = describe_process_steps(args)
    steps_str = " → ".join(steps_list) if steps_list else f"{ftype_label}{int(args.cutoff)}Hz"

    if getattr(args, 'dry_run', False):
        print_dry_run_plan(wav_files, output_dir, steps_str)
        return None, None

    print(f"\n批量{ftype_label}滤波目录: {input_dir}")
    print(f"截止频率: {args.cutoff} Hz, 方式: {args.domain}")
    print(f"处理步骤: {steps_str}")
    print(f"找到 {len(wav_files)} 个 WAV 文件")
    print(f"输出目录: {output_dir}")

    results = []
    for i, filepath in enumerate(wav_files, 1):
        fname = os.path.basename(filepath)
        out_name = build_output_filename(fname, args, op_type='filter')
        out_path = os.path.join(output_dir, out_name)

        print(f"\n[{i}/{len(wav_files)}] 处理: {fname} → {out_name}")
        result = {
            'filename': fname,
            'input_path': os.path.abspath(filepath),
            'output_path': os.path.abspath(out_path),
            'steps': steps_str,
            'status': '处理中',
        }
        try:
            signal, sr, ch = read_wav(filepath)
            rms_in = compute_signal_rms(signal)
            a_in = analyze_signal_full(signal, sr, max_peaks=args.max_peaks)

            if args.type == 'lowpass':
                if args.domain == 'freq':
                    filtered = apply_freq_filter(signal, sr, args.cutoff, 'lowpass')
                else:
                    kernel = create_lowpass_kernel(args.cutoff, sr,
                                                    kernel_size=args.kernel_size,
                                                    window_type=args.window)
                    filtered = apply_time_filter(signal, kernel)
            else:
                if args.domain == 'freq':
                    filtered = apply_freq_filter(signal, sr, args.cutoff, 'highpass')
                else:
                    kernel = create_highpass_kernel(args.cutoff, sr,
                                                  kernel_size=args.kernel_size,
                                                  window_type=args.window)
                    filtered = apply_time_filter(signal, kernel)

            write_wav(out_path, filtered, sr, n_channels=ch)

            rms_out = compute_signal_rms(filtered)
            a_out = analyze_signal_full(filtered, sr, max_peaks=args.max_peaks)

            result.update({
                'sample_rate': sr,
                'duration': a_in['duration'],
                'channels': ch,
                'rms_in': rms_in,
                'rms_out': rms_out,
                'rms_change_db': 20 * math.log10(rms_out / rms_in) if rms_in > 0 else 0,
                'peak_in': a_in['peak'],
                'peak_out': a_out['peak'],
                'dominant_in': round(a_in['dominant_freq'], 2),
                'dominant_out': round(a_out['dominant_freq'], 2),
                'status': '成功',
            })
        except Exception as e:
            result['status'] = '失败'
            result['error_msg'] = str(e)
            print(f"  跳过 (错误: {e})")
        results.append(result)

    extra_info = {
        '任务类型': f'{ftype_label}滤波',
        '截止频率(Hz)': args.cutoff,
        '实现方式': args.domain,
        '输入目录': input_dir,
        '输出目录': output_dir,
    }
    finalize_batch_results(results, output_dir, extra_info=extra_info)
    return None, None


def run_gain(args):
    print_sep()
    print("  音频信号处理工具 - 增益调整")
    print_sep()

    if os.path.isdir(args.input):
        return run_batch_gain(args)

    signal, sample_rate, channels = read_wav(args.input)
    print(f"\n输入文件: {args.input}")

    analysis_in = analyze_signal_full(signal, sample_rate, label="处理前",
                                       max_peaks=args.max_peaks)
    print(format_analysis_summary(analysis_in))

    print(f"\n增益: {args.gain:+.2f} dB (线性: {10**(args.gain/20):.4f}x)")

    gained = apply_gain(signal, args.gain)

    analysis_out = analyze_signal_full(gained, sample_rate, label="处理后",
                                        max_peaks=args.max_peaks)
    print(format_analysis_summary(analysis_out))

    if args.output:
        write_wav(args.output, gained, sample_rate, n_channels=channels)
        print(f"\n已保存到: {args.output}")

    return gained, sample_rate


def run_batch_gain(args):
    input_dir = args.input
    output_dir = args.output if args.output else os.path.join(input_dir, 'gain_out')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = find_wav_files(input_dir)
    if not wav_files:
        print(f"错误: 目录中没有找到 WAV 文件: {input_dir}")
        sys.exit(1)

    steps_list = describe_process_steps(args)
    steps_str = " → ".join(steps_list)

    if getattr(args, 'dry_run', False):
        print_dry_run_plan(wav_files, output_dir, steps_str)
        return None, None

    print(f"\n批量增益调整: {args.gain:+.2f} dB")
    print(f"处理步骤: {steps_str}")
    print(f"输入目录: {input_dir}")
    print(f"找到 {len(wav_files)} 个 WAV 文件")
    print(f"输出目录: {output_dir}")

    results = []
    for i, filepath in enumerate(wav_files, 1):
        fname = os.path.basename(filepath)
        out_name = build_output_filename(fname, args, op_type='gain')
        out_path = os.path.join(output_dir, out_name)

        print(f"\n[{i}/{len(wav_files)}] 处理: {fname} → {out_name}")
        result = {
            'filename': fname,
            'input_path': os.path.abspath(filepath),
            'output_path': os.path.abspath(out_path),
            'steps': steps_str,
            'status': '处理中',
        }
        try:
            signal, sr, ch = read_wav(filepath)
            a_in = analyze_signal_full(signal, sr, max_peaks=args.max_peaks)
            gained = apply_gain(signal, args.gain)
            write_wav(out_path, gained, sr, n_channels=ch)
            a_out = analyze_signal_full(gained, sr, max_peaks=args.max_peaks)

            rms_in = a_in['rms']
            rms_out = a_out['rms']

            result.update({
                'sample_rate': sr,
                'duration': a_in['duration'],
                'channels': ch,
                'rms_in': rms_in,
                'rms_out': rms_out,
                'rms_change_db': 20 * math.log10(rms_out / rms_in) if rms_in > 0 else 0,
                'peak_in': a_in['peak'],
                'peak_out': a_out['peak'],
                'dominant_in': round(a_in['dominant_freq'], 2),
                'dominant_out': round(a_out['dominant_freq'], 2),
                'status': '成功',
            })
        except Exception as e:
            result['status'] = '失败'
            result['error_msg'] = str(e)
            print(f"  跳过 (错误: {e})")
        results.append(result)

    extra_info = {
        '任务类型': '增益调整',
        '增益(dB)': args.gain,
        '输入目录': input_dir,
        '输出目录': output_dir,
    }
    finalize_batch_results(results, output_dir, extra_info=extra_info)
    return None, None


def run_echo(args):
    print_sep()
    print("  音频信号处理工具 - 回声效果")
    print_sep()

    if os.path.isdir(args.input):
        return run_batch_echo(args)

    signal, sample_rate, channels = read_wav(args.input)
    print(f"\n输入文件: {args.input}")

    analysis_in = analyze_signal_full(signal, sample_rate, label="处理前",
                                       max_peaks=args.max_peaks)
    print(format_analysis_summary(analysis_in))

    print(f"\n回声参数:")
    print(f"  延迟时间: {args.delay} ms ({int(args.delay * sample_rate / 1000)} 采样点)")
    print(f"  衰减系数: {args.decay}")
    print(f"  反馈模式: {'开启' if args.feedback else '关闭'}")

    echoed = apply_echo(signal, sample_rate, args.delay, args.decay,
                          feedback=args.feedback)

    analysis_out = analyze_signal_full(echoed, sample_rate, label="处理后",
                                        max_peaks=args.max_peaks)
    print(format_analysis_summary(analysis_out))

    if args.output:
        write_wav(args.output, echoed, sample_rate, n_channels=channels)
        print(f"\n已保存到: {args.output}")

    return echoed, sample_rate


def run_batch_echo(args):
    input_dir = args.input
    output_dir = args.output if args.output else os.path.join(input_dir, 'echo_out')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = find_wav_files(input_dir)
    if not wav_files:
        print(f"错误: 目录中没有找到 WAV 文件: {input_dir}")
        sys.exit(1)

    steps_list = describe_process_steps(args)
    steps_str = " → ".join(steps_list)
    echo_type = "反馈式" if args.feedback else "单次"

    if getattr(args, 'dry_run', False):
        print_dry_run_plan(wav_files, output_dir, steps_str)
        return None, None

    print(f"\n批量回声效果: {echo_type} 延迟 {args.delay}ms, 衰减 {args.decay}")
    print(f"处理步骤: {steps_str}")
    print(f"输入目录: {input_dir}")
    print(f"找到 {len(wav_files)} 个 WAV 文件")
    print(f"输出目录: {output_dir}")

    results = []
    for i, filepath in enumerate(wav_files, 1):
        fname = os.path.basename(filepath)
        out_name = build_output_filename(fname, args, op_type='echo')
        out_path = os.path.join(output_dir, out_name)

        print(f"\n[{i}/{len(wav_files)}] 处理: {fname} → {out_name}")
        result = {
            'filename': fname,
            'input_path': os.path.abspath(filepath),
            'output_path': os.path.abspath(out_path),
            'steps': steps_str,
            'status': '处理中',
        }
        try:
            signal, sr, ch = read_wav(filepath)
            a_in = analyze_signal_full(signal, sr, max_peaks=args.max_peaks)
            echoed = apply_echo(signal, sr, args.delay, args.decay, feedback=args.feedback)
            write_wav(out_path, echoed, sr, n_channels=ch)
            a_out = analyze_signal_full(echoed, sr, max_peaks=args.max_peaks)

            rms_in = a_in['rms']
            rms_out = a_out['rms']

            result.update({
                'sample_rate': sr,
                'duration': a_in['duration'],
                'channels': ch,
                'rms_in': rms_in,
                'rms_out': rms_out,
                'rms_change_db': 20 * math.log10(rms_out / rms_in) if rms_in > 0 else 0,
                'peak_in': a_in['peak'],
                'peak_out': a_out['peak'],
                'dominant_in': round(a_in['dominant_freq'], 2),
                'dominant_out': round(a_out['dominant_freq'], 2),
                'status': '成功',
            })
        except Exception as e:
            result['status'] = '失败'
            result['error_msg'] = str(e)
            print(f"  跳过 (错误: {e})")
        results.append(result)

    extra_info = {
        '任务类型': '回声效果',
        '延迟(ms)': args.delay,
        '衰减系数': args.decay,
        '反馈模式': '开启' if args.feedback else '关闭',
        '输入目录': input_dir,
        '输出目录': output_dir,
    }
    finalize_batch_results(results, output_dir, extra_info=extra_info)
    return None, None


def run_process(args):
    print_sep()
    print("  音频信号处理工具 - 完整处理流程")
    print_sep()

    sample_rate = args.sample_rate
    channels = 1

    if args.input and os.path.isdir(args.input):
        return run_batch_process(args)

    if args.preset:
        presets = load_presets(args.preset_file)
        overwrite = getattr(args, 'preset_overwrite', True)
        apply_preset_to_args(args, args.preset, presets, overwrite=overwrite)

    if args.input:
        if os.path.isdir(args.input):
            return run_batch_process(args)
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
    analysis_in = analyze_signal_full(signal, sample_rate, label="原始信号",
                                       max_peaks=args.max_peaks)
    print(format_analysis_summary(analysis_in))

    current_signal = signal
    current_label = "原始信号"

    steps = []

    if args.lowpass is not None:
        print(f"\n--- 步骤 {len(steps)+1}: 低通滤波 (截止 {args.lowpass} Hz) ---")
        domain = args.filter_domain
        if domain == 'freq':
            current_signal = apply_freq_filter(current_signal, sample_rate,
                                               args.lowpass, 'lowpass')
        else:
            kernel = create_lowpass_kernel(args.lowpass, sample_rate,
                                            kernel_size=args.kernel_size,
                                            window_type=args.window)
            current_signal = apply_time_filter(current_signal, kernel)
        steps.append(f"低通{args.lowpass}Hz")
        analysis = analyze_signal_full(current_signal, sample_rate,
                                       label="低通滤波后", max_peaks=args.max_peaks)
        print(format_analysis_summary(analysis))

    if args.highpass is not None:
        print(f"\n--- 步骤 {len(steps)+1}: 高通滤波 (截止 {args.highpass} Hz) ---")
        domain = args.filter_domain
        if domain == 'freq':
            current_signal = apply_freq_filter(current_signal, sample_rate,
                                               args.highpass, 'highpass')
        else:
            kernel = create_highpass_kernel(args.highpass, sample_rate,
                                          kernel_size=args.kernel_size,
                                          window_type=args.window)
            current_signal = apply_time_filter(current_signal, kernel)
        steps.append(f"高通{args.highpass}Hz")
        analysis = analyze_signal_full(current_signal, sample_rate,
                                       label="高通滤波后", max_peaks=args.max_peaks)
        print(format_analysis_summary(analysis))

    if args.gain != 0:
        print(f"\n--- 步骤 {len(steps)+1}: 增益 {args.gain:+.2f} dB ---")
        current_signal = apply_gain(current_signal, args.gain)
        steps.append(f"增益{args.gain:+.0f}dB")
        analysis = analyze_signal_full(current_signal, sample_rate,
                                       label="增益后", max_peaks=args.max_peaks)
        print(format_analysis_summary(analysis))

    if args.echo_delay is not None:
        print(f"\n--- 步骤 {len(steps)+1}: 回声 (延迟 {args.echo_delay}ms, 衰减 {args.echo_decay}) ---")
        current_signal = apply_echo(current_signal, sample_rate,
                                  args.echo_delay, args.echo_decay,
                                  feedback=args.echo_feedback)
        steps.append(f"回声{args.echo_delay}ms")
        analysis = analyze_signal_full(current_signal, sample_rate,
                                       label="回声后", max_peaks=args.max_peaks)
        print(format_analysis_summary(analysis))

    print()
    print_sep("-")
    print("  完整处理对比")
    print_sep("-")
    rms_in = analysis_in['rms']
    rms_out = compute_signal_rms(current_signal)
    print(f"  处理步骤: {' → '.join(steps) if steps else '(无处理)'}")
    print(f"  RMS: {rms_in:.6f} → {rms_out:.6f}")
    if rms_in > 0:
        print(f"  总变化: {20*math.log10(rms_out/rms_in):+.2f} dB")

    if args.output:
        write_wav(args.output, current_signal, sample_rate, n_channels=channels)
        print(f"\n已保存到: {args.output}")

    if args.csv:
        analysis_out = analyze_signal_full(current_signal, sample_rate,
                                           label="输出", max_peaks=args.max_peaks)
        analysis_out['filename'] = os.path.basename(args.output) if args.output else '(输出)'
        export_spectrum_csv(args.csv, analysis_out, signal=current_signal,
                           sample_rate=sample_rate,
                           include_full_spectrum=args.full_csv)
        print(f"频谱数据已导出到: {args.csv}")

    if args.save_preset:
        config = {
            'description': args.save_preset_desc if args.save_preset_desc else '自定义预设',
            'lowpass': args.lowpass,
            'highpass': args.highpass,
            'gain': args.gain,
            'echo_delay': args.echo_delay,
            'echo_decay': args.echo_decay,
            'echo_feedback': args.echo_feedback,
            'filter_domain': args.filter_domain,
            'kernel_size': args.kernel_size,
            'window': args.window,
        }
        save_preset(args.save_preset, config, args.preset_file)
        print(f"\n预设已保存: {args.save_preset} → {args.preset_file}")

    return current_signal, sample_rate


def run_batch_process(args):
    input_dir = args.input
    output_dir = args.output if args.output else os.path.join(input_dir, 'processed_out')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = find_wav_files(input_dir)
    if not wav_files:
        print(f"错误: 目录中没有找到 WAV 文件: {input_dir}")
        sys.exit(1)

    if args.preset:
        presets = load_presets(args.preset_file)
        overwrite = getattr(args, 'preset_overwrite', True)
        apply_preset_to_args(args, args.preset, presets, overwrite=overwrite)

    steps_list = describe_process_steps(args)
    steps_str = " → ".join(steps_list) if steps_list else "(仅读写)"

    if getattr(args, 'dry_run', False):
        print_dry_run_plan(wav_files, output_dir, steps_str)
        return None, None

    print(f"\n批量处理目录: {input_dir}")
    print(f"处理步骤: {steps_str}")
    print(f"找到 {len(wav_files)} 个 WAV 文件")
    print(f"输出目录: {output_dir}")

    results = []
    for i, filepath in enumerate(wav_files, 1):
        fname = os.path.basename(filepath)
        out_name = build_output_filename(fname, args, op_type='process')
        out_path = os.path.join(output_dir, out_name)

        print(f"\n[{i}/{len(wav_files)}] 处理: {fname} → {out_name}")
        result = {
            'filename': fname,
            'input_path': os.path.abspath(filepath),
            'output_path': os.path.abspath(out_path),
            'steps': steps_str,
            'status': '处理中',
        }
        try:
            signal, sr, ch = read_wav(filepath)
            current = signal
            a_in = analyze_signal_full(signal, sr, max_peaks=args.max_peaks)

            if args.highpass is not None:
                if args.filter_domain == 'freq':
                    current = apply_freq_filter(current, sr, args.highpass, 'highpass')
                else:
                    kernel = create_highpass_kernel(args.highpass, sr,
                                                  kernel_size=args.kernel_size,
                                                  window_type=args.window)
                    current = apply_time_filter(current, kernel)

            if args.lowpass is not None:
                if args.filter_domain == 'freq':
                    current = apply_freq_filter(current, sr, args.lowpass, 'lowpass')
                else:
                    kernel = create_lowpass_kernel(args.lowpass, sr,
                                                   kernel_size=args.kernel_size,
                                                   window_type=args.window)
                    current = apply_time_filter(current, kernel)

            if args.gain != 0:
                current = apply_gain(current, args.gain)

            if args.echo_delay is not None:
                current = apply_echo(current, sr, args.echo_delay, args.echo_decay,
                                    feedback=args.echo_feedback)

            write_wav(out_path, current, sr, n_channels=ch)
            a_out = analyze_signal_full(current, sr, max_peaks=args.max_peaks)

            rms_in = a_in['rms']
            rms_out = a_out['rms']

            result.update({
                'sample_rate': sr,
                'duration': a_in['duration'],
                'channels': ch,
                'rms_in': rms_in,
                'rms_out': rms_out,
                'rms_change_db': 20 * math.log10(rms_out / rms_in) if rms_in > 0 else 0,
                'peak_in': a_in['peak'],
                'peak_out': a_out['peak'],
                'dominant_in': round(a_in['dominant_freq'], 2),
                'dominant_out': round(a_out['dominant_freq'], 2),
                'status': '成功',
            })
        except Exception as e:
            result['status'] = '失败'
            result['error_msg'] = str(e)
            print(f"  跳过 (错误: {e})")
        results.append(result)

    extra_info = {
        '任务类型': '完整处理流水线',
        '处理步骤': steps_str,
        '输入目录': input_dir,
        '输出目录': output_dir,
    }
    if args.preset:
        extra_info['使用预设'] = args.preset
    finalize_batch_results(results, output_dir, extra_info=extra_info)
    return None, None


def run_presets_list(args):
    presets = load_presets(args.preset_file)
    print_sep()
    print("  可用预设列表")
    print_sep()
    print()
    print(list_presets(presets))
    print()
    print(f"共 {len(presets)} 个预设 (含 {len(presets) - 8} 个自定义)")
    print()


def run_presets_show(args):
    presets = load_presets(args.preset_file)
    print_sep()
    print("  预设详情")
    print_sep()
    print()
    for name in args.names:
        print(show_preset_details(name, presets))
        print()
        print_sep("-")
        print()


def run_presets_copy(args):
    presets = load_presets(args.preset_file)
    print_sep()
    print("  复制预设")
    print_sep()
    if args.src not in presets:
        print(f"错误: 源预设 '{args.src}' 不存在")
        print(list_presets(presets))
        sys.exit(1)
    if args.dst in presets and not args.force:
        print(f"错误: 目标预设 '{args.dst}' 已存在, 使用 --force 覆盖")
        sys.exit(1)
    try:
        desc = args.description if args.description else f"复制自 '{args.src}'"
        cfg = copy_preset(args.src, args.dst, args.preset_file, description=desc, presets=presets)
        print(f"\n✓ 已复制预设:")
        print(f"  源:   {args.src}")
        print(f"  目标: {args.dst}")
        if args.description:
            print(f"  描述: {desc}")
        print(f"  文件: {args.preset_file}")
        print()
        print(show_preset_details(args.dst, load_presets(args.preset_file)))
    except Exception as e:
        print(f"复制失败: {e}")
        sys.exit(1)


def add_common_input_args(parser, add_output=True):
    parser.add_argument("-i", "--input", required=True, help="输入 WAV 文件或目录")
    if add_output:
        parser.add_argument("-o", "--output", type=str, default=None, help="输出路径")
    parser.add_argument("--max-peaks", type=int, default=10, help="显示峰值数量")
    parser.add_argument("--csv", type=str, default=None, help="导出频谱 CSV")
    parser.add_argument("--full-csv", action="store_true", help="CSV 包含完整频谱数据")
    parser.add_argument("--preset-file", type=str, default='custom_presets.json',
                        help="自定义预设文件")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示执行计划, 不实际写文件")
    parser.add_argument("--preserve-name", dest="preserve_name", action="store_true", default=True,
                        help="批量输出时保留原文件名 (默认)")
    parser.add_argument("--no-preserve-name", dest="preserve_name", action="store_false",
                        help="批量输出时在文件名前加处理类型前缀")


def main():
    parser = argparse.ArgumentParser(
        description="音频信号处理工具 (Audio Signal Processor v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 生成合成信号
  python audio_tool.py generate -f "100:1,440:0.5,1000:0.3" -o test.wav --csv test_spec.csv

  # 分析 WAV 频谱
  python audio_tool.py analyze -i test.wav --max-peaks 8 --csv analysis.csv

  # 低通滤波 (频域 FFT)
  python audio_tool.py filter -i test.wav -t lowpass -c 300 -o lp_out.wav

  # 高通滤波 (时域 FIR, 更陡峭)
  python audio_tool.py filter -i test.wav -t highpass -c 500 -d time --kernel-size 1023 -o hp_out.wav

  # 增益调整
  python audio_tool.py gain -i test.wav -g 6 -o louder.wav

  # 回声效果
  python audio_tool.py echo -i test.wav --delay 300 --decay 0.5 --feedback -o echo.wav

  # 使用预设
  python audio_tool.py process -i test.wav --preset vocal-clean -o clean.wav

  # 保存自定义预设
  python audio_tool.py process -i test.wav --highpass 200 --gain 3 --save-preset my-preset

  # 列出所有预设
  python audio_tool.py presets list

  # 批量处理目录
  python audio_tool.py analyze -i ./input_dir -o ./output_dir
  python audio_tool.py filter -i ./input_dir -t lowpass -c 500 -o ./lp_out
""")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ============ generate ============
    p_gen = subparsers.add_parser("generate", help="生成合成信号")
    p_gen.add_argument("-f", "--frequencies", required=True,
                        help="频率列表 'freq:amp,freq:amp,... (amp 可省略, 默认 1.0)")
    p_gen.add_argument("-s", "--sample-rate", type=int, default=22050,
                        help="采样率 (默认 22050 Hz)")
    p_gen.add_argument("-d", "--duration", type=float, default=2.0,
                        help="时长秒数 (默认 2.0)")
    p_gen.add_argument("-n", "--noise", type=float, default=0.0,
                        help="白噪声幅度 (默认 0)")
    p_gen.add_argument("-o", "--output", type=str, default=None,
                        help="输出 WAV 文件路径")
    p_gen.add_argument("--max-peaks", type=int, default=10, help="显示峰值数量")
    p_gen.add_argument("--csv", type=str, default=None, help="导出频谱 CSV")
    p_gen.add_argument("--full-csv", action="store_true", help="CSV 包含完整频谱")

    # ============ analyze ============
    p_ana = subparsers.add_parser("analyze", help="频谱分析")
    add_common_input_args(p_ana)
    p_ana.add_argument("--csv-each", action="store_true",
                        help="批量时为每个文件导出 CSV")

    # ============ filter ============
    p_filt = subparsers.add_parser("filter", help="低通/高通滤波")
    add_common_input_args(p_filt)
    p_filt.add_argument("-t", "--type", choices=["lowpass", "highpass"],
                         required=True, help="滤波器类型")
    p_filt.add_argument("-c", "--cutoff", type=float, required=True,
                         help="截止频率 (Hz)")
    p_filt.add_argument("-d", "--domain", choices=["freq", "time"],
                         default="time", help="实现方式 (默认 time/FIR)")
    p_filt.add_argument("--kernel-size", type=int, default=None,
                         help="FIR 核大小 (默认根据窗函数自动)")
    p_filt.add_argument("--window", choices=["blackman", "hamming", "hann", "rectangular"],
                         default="blackman", help="FIR 窗函数 (默认 blackman)")

    # ============ gain ============
    p_gain = subparsers.add_parser("gain", help="增益调整")
    add_common_input_args(p_gain)
    p_gain.add_argument("-g", "--gain", type=float, required=True,
                         help="增益 dB (正数放大, 负数衰减)")

    # ============ echo ============
    p_echo = subparsers.add_parser("echo", help="回声效果")
    add_common_input_args(p_echo)
    p_echo.add_argument("--delay", type=float, default=200, help="延迟时间 ms")
    p_echo.add_argument("--decay", type=float, default=0.5, help="衰减系数 0-1")
    p_echo.add_argument("--feedback", action="store_true", help="启用反馈回声")

    # ============ process ============
    p_proc = subparsers.add_parser("process", help="完整处理管道")
    src_group = p_proc.add_mutually_exclusive_group(required=True)
    src_group.add_argument("-i", "--input", type=str, help="输入 WAV 文件或目录")
    src_group.add_argument("-f", "--frequencies", type=str,
                            help="频率列表 (同 generate)")
    p_proc.add_argument("-s", "--sample-rate", type=int, default=22050,
                          help="采样率 (生成信号用)")
    p_proc.add_argument("--duration", type=float, default=2.0,
                          help="生成信号时长")
    p_proc.add_argument("--noise", type=float, default=0.0, help="白噪声幅度")
    p_proc.add_argument("--lowpass", type=float, default=None, help="低通截止 Hz")
    p_proc.add_argument("--highpass", type=float, default=None, help="高通截止 Hz")
    p_proc.add_argument("--filter-domain", choices=["freq", "time"],
                         default="time", help="滤波实现方式")
    p_proc.add_argument("--kernel-size", type=int, default=None, help="FIR 核大小")
    p_proc.add_argument("--window", choices=["blackman", "hamming", "hann", "rectangular"],
                         default="blackman", help="FIR 窗函数")
    p_proc.add_argument("-g", "--gain", type=float, default=0, help="增益 dB")
    p_proc.add_argument("--echo-delay", type=float, default=None, help="回声延迟 ms")
    p_proc.add_argument("--echo-decay", type=float, default=0.5, help="回声衰减")
    p_proc.add_argument("--echo-feedback", action="store_true", help="回声反馈")
    p_proc.add_argument("-o", "--output", type=str, default=None, help="输出文件")
    p_proc.add_argument("--max-peaks", type=int, default=10, help="峰值数")
    p_proc.add_argument("--csv", type=str, default=None, help="导出频谱 CSV")
    p_proc.add_argument("--full-csv", action="store_true", help="完整频谱 CSV")
    p_proc.add_argument("--preset", type=str, default=None, help="应用预设")
    p_proc.add_argument("--preset-overwrite", dest="preset_overwrite", action="store_true", default=True,
                        help="预设参数完整覆盖默认值 (默认)")
    p_proc.add_argument("--preset-no-overwrite", dest="preset_overwrite", action="store_false",
                        help="预设参数仅填充未设置的值")
    p_proc.add_argument("--preset-file", type=str, default='custom_presets.json',
                        help="自定义预设文件")
    p_proc.add_argument("--save-preset", type=str, default=None,
                        help="保存当前参数为预设")
    p_proc.add_argument("--save-preset-desc", type=str, default=None,
                        help="预设描述")
    p_proc.add_argument("--dry-run", action="store_true",
                        help="仅显示执行计划")
    p_proc.add_argument("--preserve-name", dest="preserve_name", action="store_true", default=True,
                        help="批量输出保留原文件名 (默认)")
    p_proc.add_argument("--no-preserve-name", dest="preserve_name", action="store_false",
                        help="批量输出加前缀")

    # ============ presets ============
    p_pre = subparsers.add_parser("presets", help="预设管理")
    p_pre_sub = p_pre.add_subparsers(dest="preset_cmd", help="预设命令")
    p_pre_list = p_pre_sub.add_parser("list", help="列出所有预设")
    p_pre_list.add_argument("--preset-file", type=str, default='custom_presets.json',
                           help="自定义预设文件")
    p_pre_show = p_pre_sub.add_parser("show", help="查看预设详细参数")
    p_pre_show.add_argument("names", nargs="+", help="预设名称 (可多个)")
    p_pre_show.add_argument("--preset-file", type=str, default='custom_presets.json',
                           help="自定义预设文件")
    p_pre_copy = p_pre_sub.add_parser("copy", help="复制预设并改名保存")
    p_pre_copy.add_argument("src", help="源预设名称")
    p_pre_copy.add_argument("dst", help="新预设名称")
    p_pre_copy.add_argument("-d", "--description", type=str, default=None,
                           help="新预设描述")
    p_pre_copy.add_argument("-f", "--force", action="store_true",
                           help="若目标预设已存在则覆盖")
    p_pre_copy.add_argument("--preset-file", type=str, default='custom_presets.json',
                           help="自定义预设文件 (保存位置)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    handlers = {
        "generate": run_generate,
        "analyze": run_analyze,
        "filter": run_filter,
        "gain": run_gain,
        "echo": run_echo,
        "process": run_process,
    }

    if args.command == "presets":
        if args.preset_cmd == "list":
            run_presets_list(args)
        elif args.preset_cmd == "show":
            run_presets_show(args)
        elif args.preset_cmd == "copy":
            run_presets_copy(args)
        else:
            p_pre.print_help()
        return

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


if __name__ == "__main__":
    main()
