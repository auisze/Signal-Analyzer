from signal_module import Signal, FourierTransform, SignalVisualizer

def main():
    # Создаем сигнал
    signal = Signal(duration=1.0, sampling_rate=1000)
    signal.add_harmonic(amplitude=1.0, frequency=50)
    signal.add_harmonic(amplitude=0.5, frequency=120)
    signal.add_aperiodic(noise_amplitude=0.2)

    # Выполняем преобразование Фурье
    ft = FourierTransform(signal)
    ft.compute_fft()

    # Визуализируем и сохраняем графики в один файл
    visualizer = SignalVisualizer(signal, ft)
    visualizer.save_plots('signal_analysis.png')

if __name__ == '__main__':
    main()
