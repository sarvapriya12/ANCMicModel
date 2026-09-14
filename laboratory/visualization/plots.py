import numpy as np
import pyqtgraph as pg


class WaveformPlot(pg.PlotWidget):
    def __init__(self, title="Waveform"):
        super().__init__(title=title)
        self.setYRange(-1, 1)
        self.setLabel('bottom', 'Samples')
        self.setLabel('left', 'Amplitude')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve = self.plot(pen='g')
        
    def update_data(self, data):
        self.plot_curve.setData(data)

class DualWaveformPlot(pg.PlotWidget):
    def __init__(self, title="GRAPH 1: RAW INPUT (Mic A vs Mic B)"):
        super().__init__(title=title)
        self.setYRange(-1, 1)
        self.setLabel('bottom', 'Time (ms)')
        self.setLabel('left', 'Amplitude')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.curve_a = self.plot(pen=pg.mkPen(color='#00ffff', width=1.5), name="Mic A")
        self.curve_b = self.plot(pen=pg.mkPen(color='#ff0000', width=1.5), name="Mic B")
        self.addLegend()
        self.sr = 48000
        
    def update_data(self, data_a, data_b):
        time_axis = np.arange(len(data_a)) * 1000 / self.sr
        self.curve_a.setData(time_axis, data_a)
        self.curve_b.setData(time_axis, data_b)

class SpectrumPlot(pg.PlotWidget):
    def __init__(self, title="GRAPH 2: SPECTRUM (Raw vs Enhanced)", sr=48000):
        super().__init__(title=title)
        self.sr = sr
        self.setLogMode(x=True, y=False)
        self.setYRange(-100, 0)
        self.setLabel('bottom', 'Frequency (Hz)')
        self.setLabel('left', 'Magnitude (dB)')
        self.showGrid(x=True, y=True, alpha=0.5)
        self.curve_raw = self.plot(pen=pg.mkPen(color='#ff5500', width=1, style=pg.QtCore.Qt.PenStyle.DashLine), name="Mic A Raw")
        self.curve_enh = self.plot(pen=pg.mkPen(color='#00ff00', width=1.5), name="Enhanced")
        self.addLegend()
        
    def update_data(self, raw_audio, enhanced_audio):
        if len(raw_audio) > 0:
            window = np.hanning(len(raw_audio))
            spec_raw = np.fft.rfft(raw_audio * window)
            mag_raw = 20 * np.log10(np.abs(spec_raw) + 1e-10)
            freqs = np.fft.rfftfreq(len(raw_audio), 1/self.sr)
            freqs[0] = 1 # avoid log(0)
            self.curve_raw.setData(freqs, mag_raw)
            
        if len(enhanced_audio) > 0:
            window_enh = np.hanning(len(enhanced_audio))
            spec_enh = np.fft.rfft(enhanced_audio * window_enh)
            mag_enh = 20 * np.log10(np.abs(spec_enh) + 1e-10)
            freqs_enh = np.fft.rfftfreq(len(enhanced_audio), 1/self.sr)
            freqs_enh[0] = 1
            self.curve_enh.setData(freqs_enh, mag_enh)

class CoherencePlot(pg.PlotWidget):
    def __init__(self, title="GRAPH 3: FREQUENCY COHERENCE"):
        super().__init__(title=title)
        self.setYRange(0, 1)
        self.setLogMode(x=True, y=False)
        self.setLabel('bottom', 'Frequency (Hz)')
        self.setLabel('left', 'Coherence')
        self.showGrid(x=True, y=True, alpha=0.5)
        self.plot_curve = self.plot(pen=pg.mkPen(color='#ff00ff', width=1.5))
        self.sr = 48000
        
    def update_data(self, mic_a, mic_b):
        if len(mic_a) == 0: return
        window = np.hanning(len(mic_a))
        spec_a = np.fft.rfft(mic_a * window)
        spec_b = np.fft.rfft(mic_b * window)
        
        cross_spec = spec_a * np.conj(spec_b)
        auto_a = np.abs(spec_a)**2
        auto_b = np.abs(spec_b)**2
        
        coherence = np.abs(cross_spec)**2 / (auto_a * auto_b + 1e-10)
        freqs = np.fft.rfftfreq(len(mic_a), 1/self.sr)
        freqs[0] = 1
        
        self.plot_curve.setData(freqs, coherence)

class LatencyBarChart(pg.PlotWidget):
    def __init__(self, title="GRAPH 4: LATENCY BREAKDOWN"):
        super().__init__(title=title)
        self.setYRange(0, 30)
        self.setXRange(-0.5, 2.5)
        self.setLabel('left', 'Latency (ms)')
        self.showGrid(y=True, alpha=0.5)
        
        ax = self.getAxis('bottom')
        ax.setTicks([[(0, 'DSP'), (1, 'AI'), (2, 'Total')]])
        
        self.bar_dsp = pg.BarGraphItem(x=[0], height=[0], width=0.6, brush='b')
        self.bar_ai = pg.BarGraphItem(x=[1], height=[0], width=0.6, brush='m')
        self.bar_tot = pg.BarGraphItem(x=[2], height=[0], width=0.6, brush='r')
        
        self.addItem(self.bar_dsp)
        self.addItem(self.bar_ai)
        self.addItem(self.bar_tot)
        
    def update_data(self, dsp_lat, ai_lat):
        tot = dsp_lat + ai_lat
        self.bar_dsp.setOpts(height=[dsp_lat])
        self.bar_ai.setOpts(height=[ai_lat])
        self.bar_tot.setOpts(height=[tot])
        max_lat = max(tot, 10)
        if max_lat > self.getViewBox().viewRange()[1][1] * 0.9:
            self.setYRange(0, max_lat * 1.5)
