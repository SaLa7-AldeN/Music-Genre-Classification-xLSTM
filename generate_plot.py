import numpy as np
import matplotlib.pyplot as plt

# إعدادات الشكل
plt.figure(figsize=(10, 6))

# 1. إنشاء إشارة أنالوج (موجة متصلة)
t_analog = np.linspace(0, 1, 1000)  # وقت متصل
frequency = 5  # تردد الموجة 5 هرتز
amplitude = 1
analog_signal = amplitude * np.sin(2 * np.pi * frequency * t_analog)

# 2. إنشاء إشارة ديجital (نقاط العينة - Sampling)
sampling_rate = 15  # معدل أخذ العينات (أكثر من ضعف التردد حسب نايكويست)
t_digital = np.linspace(0, 1, sampling_rate)
digital_samples = amplitude * np.sin(2 * np.pi * frequency * t_digital)

# 3. الرسم
plt.plot(t_analog, analog_signal, label='Analog Signal (Continuous)', color='cornflowerblue', linewidth=2, alpha=0.7)
plt.stem(t_digital, digital_samples, linefmt='r--', markerfmt='ro', basefmt=' ', label='Digital Samples (Discrete)')

# 4. التزيين والكتابة
plt.title('Analog to Digital Conversion: Sampling Process', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 5. حفظ الصورة
output_filename = "nyquist_sampling.png"
plt.savefig(output_filename, dpi=300)
print(f"✅ Image saved as: {output_filename}")
plt.show()