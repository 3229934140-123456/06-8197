import math
from audio_engine import *


def print_separator(title=""):
    line = "=" * 70
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(f"\n{line}")


def demo_fft_basics():
    print_separator("第一部分：FFT 基础验证")

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

    magnitudes = [abs(s) for s in spectrum[:n//2]]
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


def demo_spectrum_analysis():
    print_separator("第二部分：合成信号频谱分析")

    sample_rate = 2000
    duration = 1.024
    n = int(sample_rate * duration)

    components = [
        (50, 1.0),
        (120, 0.7),
        (250, 0.5),
        (500, 0.3),
        (800, 0.2),
    ]

    print(f"\n合成信号成分:")
    for freq, amp in components:
        print(f"  {freq:4d} Hz, 幅度 {amp:.2f}")

    signal = generate_composite_signal(components, sample_rate, duration)
    signal = add_white_noise(signal, amplitude=0.05)

    frequencies, magnitudes, full_spectrum = compute_spectrum(signal, sample_rate)

    print(f"\n频谱分析结果:")
    print(f"  采样率: {sample_rate} Hz")
    print(f"  FFT 长度: {n} 点")
    print(f"  频率分辨率: {sample_rate/n:.3f} Hz/bin")
    print(f"  Nyquist 频率: {sample_rate/2} Hz")

    print(f"\n主要频谱峰值 (前 8 个):")
    peak_data = []
    for k in range(1, n//2 - 1):
        if magnitudes[k] > magnitudes[k-1] and magnitudes[k] > magnitudes[k+1]:
            freq = frequencies[k]
            mag_norm = 2 * magnitudes[k] / n
            if mag_norm > 0.05:
                peak_data.append((freq, mag_norm))

    peak_data.sort(key=lambda x: x[1], reverse=True)
    for i, (freq, mag) in enumerate(peak_data[:8]):
        print(f"  峰值 {i+1}: {freq:7.2f} Hz, 归一化幅度 {mag:.4f}")

    print("\n✓ 频谱分析完成")
    return signal, sample_rate, frequencies, magnitudes


def demo_filter_freq_domain(signal, sample_rate):
    print_separator("第三部分：频域滤波（FFT 方法）")

    n = len(signal)

    print(f"\n原始信号 RMS: {math.sqrt(sum(s**2 for s in signal)/n):.4f}")

    lowpass_signal = apply_freq_filter(signal, sample_rate, 200, 'lowpass')
    highpass_signal = apply_freq_filter(signal, sample_rate, 300, 'highpass')

    print(f"\n低通滤波 (截止 200 Hz):")
    print(f"  输出 RMS: {math.sqrt(sum(s**2 for s in lowpass_signal[:n])/n):.4f}")

    _, mag_low, _ = compute_spectrum(lowpass_signal, sample_rate)
    print(f"  200 Hz 以上分量 (归一化幅度和): {sum(mag_low[40:])*2/n:.4f}")

    print(f"\n高通滤波 (截止 300 Hz):")
    print(f"  输出 RMS: {math.sqrt(sum(s**2 for s in highpass_signal[:n])/n):.4f}")

    _, mag_high, _ = compute_spectrum(highpass_signal, sample_rate)
    print(f"  300 Hz 以下分量 (归一化幅度和): {sum(mag_high[:60])*2/n:.4f}")

    print("\n✓ 频域滤波演示完成")
    return lowpass_signal, highpass_signal


def demo_filter_time_domain(signal, sample_rate):
    print_separator("第四部分：时域滤波（卷积核方法）")

    n = len(signal)

    kernel_size = 63
    lp_kernel = create_lowpass_kernel(200, sample_rate, kernel_size)
    hp_kernel = create_highpass_kernel(300, sample_rate, kernel_size)

    print(f"\n低通滤波器核:")
    print(f"  长度: {len(lp_kernel)} 抽头")
    print(f"  截止频率: 200 Hz")
    print(f"  窗函数: Blackman")
    print(f"  核系数和: {sum(lp_kernel):.6f}")

    lp_filtered = apply_time_filter(signal, lp_kernel)

    print(f"\n低通滤波结果:")
    print(f"  输入 RMS: {math.sqrt(sum(s**2 for s in signal)/n):.4f}")
    print(f"  输出 RMS: {math.sqrt(sum(s**2 for s in lp_filtered)/n):.4f}")

    hp_filtered = apply_time_filter(signal, hp_kernel)

    print(f"\n高通滤波结果:")
    print(f"  输出 RMS: {math.sqrt(sum(s**2 for s in hp_filtered)/n):.4f}")

    print("\n✓ 时域滤波演示完成")
    return lp_filtered, hp_filtered


def demo_effects(signal, sample_rate):
    print_separator("第五部分：音频效果（增益、回声）")

    n = len(signal)

    print(f"\n--- 增益效果 ---")
    gain_db = 6.0
    gained_signal = apply_gain(signal, gain_db)

    input_rms = math.sqrt(sum(s**2 for s in signal)/n)
    output_rms = math.sqrt(sum(s**2 for s in gained_signal)/n)
    actual_gain_db = 20 * math.log10(output_rms / input_rms)

    print(f"  目标增益: {gain_db} dB")
    print(f"  实际增益: {actual_gain_db:.2f} dB")
    print(f"  线性增益倍数: {10**(gain_db/20):.4f}")

    print(f"\n--- 回声效果（无反馈）---")
    delay_ms = 200
    decay = 0.5
    echo_signal = apply_echo(signal, sample_rate, delay_ms, decay, feedback=False)

    delay_samples = int(delay_ms * sample_rate / 1000)
    print(f"  延迟时间: {delay_ms} ms ({delay_samples} 采样点)")
    print(f"  衰减系数: {decay}")
    print(f"  输入 RMS: {input_rms:.4f}")
    print(f"  输出 RMS: {math.sqrt(sum(s**2 for s in echo_signal)/n):.4f}")

    print(f"\n--- 回声效果（带反馈）---")
    decay_fb = 0.6
    echo_fb_signal = apply_echo(signal, sample_rate, delay_ms, decay_fb, feedback=True)

    print(f"  延迟时间: {delay_ms} ms")
    print(f"  反馈系数: {decay_fb}")
    print(f"  输出 RMS: {math.sqrt(sum(s**2 for s in echo_fb_signal)/n):.4f}")

    print("\n✓ 音频效果演示完成")
    return gained_signal, echo_signal, echo_fb_signal


def demo_theory_explanation():
    print_separator("第六部分：原理说明")

    print("""
1. FFT 如何用分治把 O(n²) 降到 O(n log n)
   ─────────────────────────────────────
   朴素 DFT 公式: X[k] = Σ x[n]·e^(-j2πkn/N)
   每个频率分量需要 N 次复数乘法 → O(N²)

   Cooley-Tukey FFT 分治思想:
   - 将 x[n] 按奇偶下标分成两组: x[偶数], x[奇数]
   - 利用旋转因子对称性: W_N^(k+N/2) = -W_N^k
   - N 点 DFT = 两个 N/2 点 DFT 的组合
   - 递归分解直到 2 点 DFT

   复杂度分析:
   - 分解层数: log₂N 层
   - 每层运算量: O(N) 次蝶形运算
   - 总复杂度: O(N log N)

   例: N=1024
   - 朴素 DFT: 1,048,576 次运算
   - FFT: 10,240 次运算 → 快 100 倍


2. 蝶形运算和位反转重排的作用
   ────────────────────────────
   蝶形运算 (Butterfly):
   - FFT 的基本运算单元
   - 输入: 两个复数 a, b 和旋转因子 W
   - 输出: a + W·b 和 a - W·b
   - 形似蝴蝶,故名"蝶形"
   - 每次蝶形运算包含 1 次复数乘法 + 2 次复数加法

   位反转重排 (Bit-Reversal):
   - 因为分治是按二进制位的奇偶分组
   - 输入顺序经过 log₂N 次分组后变成"位反转"顺序
   - 例如 N=8 (3位):
     原索引:  000 001 010 011 100 101 110 111
     位反转:  000 100 010 110 001 101 011 111
             =  0   4   2   6   1   5   3   7
   - 作用: 保证原位运算 (in-place) 的正确性
   - 方式一: 开始时把输入按位反转重排
   - 方式二: 结束时把输出按位反转重排


3. 为何输入长度通常要求是 2 的幂
   ────────────────────────────────
   - 基-2 FFT (最常用) 要求 N = 2^m
   - 每次都可以精确对半分解
   - 算法实现简单,效率高
   - 其他选择:
     * 基-4 FFT: N = 4^m, 乘法更少但更复杂
     * 混合基 FFT: N 可分解为小素数乘积
     * Bluestein 算法: 任意长度 FFT
   - 工程中常用"补零"(zero-padding)到下一个 2 的幂


4. 频谱 bin 对应的频率如何确定
   ──────────────────────────────
   第 k 个 bin 对应的频率:
     f_k = k × Fs / N

   其中:
   - Fs: 采样率 (Hz)
   - N: FFT 变换长度 (点数)
   - k: bin 索引 (0, 1, ..., N/2)

   频率分辨率 (bin 间距):
     Δf = Fs / N

   例子: Fs=44100 Hz, N=1024
   - Δf = 44100/1024 ≈ 43.07 Hz
   - bin 0: 0 Hz (直流分量)
   - bin 1: ~43 Hz
   - bin 512: 22050 Hz (Nyquist 频率)

   提高频率分辨率的方法:
   - 增加 FFT 长度 N
   - 补零可以让频谱更"密",但不能提高真实分辨率


5. 低通滤波的两种实现
   ──────────────────────
   频域方法 (FFT 滤波):
   原理: 频谱 × 滤波器频率响应
   步骤:
     1. FFT 将信号变换到频域
     2. 把截止频率以上的 bin 设为 0
     3. IFFT 变换回时域
   优点: 计算快 (O(N log N)), 概念直观
   缺点: 有"吉布斯现象", 边缘效应, 需要 2 的幂
   适用: 离线处理、频谱分析

   时域方法 (卷积/ FIR 滤波器):
   原理: 信号与滤波器冲激响应卷积
     y[n] = Σ h[k] · x[n-k]
   设计方法:
     - 窗函数法: 理想 sinc × 窗函数
     - 等波纹法: Parks-McClellan
   优点: 相位线性, 稳定, 实时处理
   缺点: 直接卷积 O(N·M), M 是核长度
   适用: 实时音频、嵌入式系统


6. 回声效果的实现原理
   ──────────────────────
   基本回声 (无反馈):
     y[n] = x[n] + α · x[n - D]
   
   带反馈回声 (多次回声):
     y[n] = x[n] + α · y[n - D]

   其中:
   - D: 延迟采样数 = 延迟时间(ms) × Fs / 1000
   - α: 衰减/反馈系数 (0 < α < 1)

   无反馈 vs 带反馈:
   - 无反馈: 只有一次回声, 简单叠加
   - 带反馈: 无限递减的回声序列
     第 k 次回声幅度: α^k
     总增益: 1/(1-α)  (需注意 α<1 保证稳定)

   实际应用中:
   - 延迟时间: 通常 30-500 ms
   - 衰减系数: 0.3-0.8
   - 可加入低通滤波模拟空气吸收
   - 可使用多抽头延迟线模拟房间反射
""")

    print("✓ 原理说明完成")


def main():
    print_separator("音频信号处理引擎 - 完整演示")
    print("  时域与频域处理 · FFT · 滤波 · 效果器")

    demo_fft_basics()

    signal, sample_rate, frequencies, magnitudes = demo_spectrum_analysis()

    demo_filter_freq_domain(signal, sample_rate)

    demo_filter_time_domain(signal, sample_rate)

    demo_effects(signal, sample_rate)

    demo_theory_explanation()

    print_separator("演示全部完成")
    print("\n所有功能模块:")
    print("  ✓ FFT/IFFT (基-2, 位反转, 蝶形运算)")
    print("  ✓ 频谱分析 (频率映射, 峰值检测)")
    print("  ✓ 频域滤波 (低通/高通, FFT 方法)")
    print("  ✓ 时域滤波 (FIR, 窗函数法, 卷积)")
    print("  ✓ 增益效果 (dB 与线性转换)")
    print("  ✓ 回声效果 (有无反馈两种模式)")
    print("  ✓ 信号生成 (正弦波, 复合信号, 噪声)")
    print()


if __name__ == "__main__":
    main()
