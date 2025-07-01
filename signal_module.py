import numpy as np
import matplotlib.pyplot as plt


class Signal:
    """
        Класс для создания и хранения сигнала.

        Атрибуты:
            duration (float): Длительность сигнала в секундах.
            sampling_rate (int): Частота дискретизации в Гц.
            t (np.ndarray): Массив точек времени.
            data (np.ndarray): Массив значений сигнала.

        Методы:
            add_harmonic(amplitude, frequency): Добавляет гармоническую составляющую к сигналу.
            add_aperiodic(noise_amplitude): Добавляет апериодический шум к сигналу.
        """
    def __init__(self, duration: float, sampling_rate: int):
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        self.data = np.zeros_like(self.t)

    def add_harmonic(self, amplitude: float, frequency: float, phase: float = 0):
        self.data += amplitude * np.sin(2 * np.pi * frequency * self.t + phase)

    def add_aperiodic(self, noise_amplitude: float):
        noise = noise_amplitude * np.random.normal(size=self.t.shape)
        self.data += noise


class FourierTransform:
    """
        Класс для выполнения преобразования Фурье сигнала.

        Атрибуты:
            signal (Signal): Объект сигнала для анализа.
            fft_result (np.ndarray): Результат преобразования Фурье.

        Методы:
            compute_fft(): Выполняет вычисление FFT.
            get_magnitude(): Возвращает амплитудный спектр.
            get_frequency_spectrum(): Возвращает массив частот.
        """
    def __init__(self, signal: Signal):
        self.signal = signal
        self.fft_result = None
        self.freqs = None

    def compute_fft(self):
        self.fft_result = np.fft.fft(self.signal.data)
        self.freqs = np.fft.fftfreq(len(self.signal.data), 1 / self.signal.sampling_rate)

    def get_magnitude(self):
        if self.fft_result is None:
            self.compute_fft()
        return np.abs(self.fft_result)

    def get_frequency_spectrum(self):
        if self.freqs is None:
            self.compute_fft()
        return self.freqs


class SignalVisualizer:
    """
        Класс для визуализации сигнала и его спектра.

        Атрибуты:
            signal (Signal): Объект сигнала для визуализации.
            fourier_transform (FourierTransform): Объект преобразования Фурье.

        Методы:
            plot_time_domain(): Строит график сигнала во временной области.
            plot_frequency_domain(): Строит график амплитудного спектра.
            save_plots(output_path): Сохраняет оба графика (временной и частотный) в один файл.
        """
    def __init__(self, signal: Signal, fourier_transform: FourierTransform):
        self.signal = signal
        self.fourier_transform = fourier_transform

    def plot_time_domain(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.signal.t, self.signal.data)
        plt.title('Исходный сигнал во временной области')
        plt.xlabel('Время [с]')
        plt.ylabel('Амплитуда')
        plt.grid(True)
        plt.tight_layout()

    def plot_frequency_domain(self):
        magnitude = self.fourier_transform.get_magnitude()
        freqs = self.fourier_transform.get_frequency_spectrum()
        half = len(freqs) // 2

        plt.figure(figsize=(10, 4))
        plt.stem(freqs[:half], magnitude[:half])
        plt.title('Амплитудный спектр сигнала')
        plt.xlabel('Частота [Гц]')
        plt.ylabel('Амплитуда')
        plt.grid(True)
        plt.tight_layout()

    def save_plots(self, output_path: str):
        """
        Сохраняет графики временного сигнала и спектра в один файл с двумя подграфиками.

        Args:
            output_path (str): Путь для сохранения файла с графиками.
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Временной сигнал
        axs[0].plot(self.signal.t, self.signal.data)
        axs[0].set_title('Исходный сигнал во временной области')
        axs[0].set_xlabel('Время [с]')
        axs[0].set_ylabel('Амплитуда')
        axs[0].grid(True)

        # Амплитудный спектр
        magnitude = self.fourier_transform.get_magnitude()
        freqs = self.fourier_transform.get_frequency_spectrum()
        half = len(freqs) // 2
        axs[1].stem(freqs[:half], magnitude[:half])
        axs[1].set_title('Амплитудный спектр сигнала')
        axs[1].set_xlabel('Частота [Гц]')
        axs[1].set_ylabel('Амплитуда')
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


