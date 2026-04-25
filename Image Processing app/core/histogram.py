import cv2
from matplotlib.figure import Figure

class Histogram:

    @staticmethod
    def computeHistoColored(image):
        histB = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        histG = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
        histR = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
        return histB, histG, histR

    @staticmethod
    def computeHistoGray(gray_image):
        return cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()

    @staticmethod
    def compute_cdf_colored(histB, histG, histR):
        normalize = lambda h: h.cumsum() / h.cumsum().max()
        return normalize(histB), normalize(histG), normalize(histR)

    @staticmethod
    def compute_cdf_gray(hist):
        cdf = hist.cumsum()
        return cdf / cdf.max()

    @staticmethod
    def plot_colored_histogram(histB, histG, histR, show_blue=True, show_green=True, show_red=True):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        if show_blue:  ax.bar(range(256), histB, color='blue',  alpha=0.3, label='Blue')
        if show_green: ax.bar(range(256), histG, color='green', alpha=0.3, label='Green')
        if show_red:   ax.bar(range(256), histR, color='red',   alpha=0.3, label='Red')
        ax.set(xlabel='Pixel Intensity', ylabel='Frequency', title='RGB Histogram')
        ax.legend(); ax.grid(True)
        return fig

    @staticmethod
    def plot_gray_histogram(hist):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.bar(range(256), hist, color='black')
        ax.set(xlabel='Pixel Intensity', ylabel='Frequency', title='Grayscale Histogram')
        ax.grid(True)
        return fig

    @staticmethod
    def plot_cdf_colored(cdfB, cdfG, cdfR, show_blue=True, show_green=True, show_red=True):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        if show_blue:  ax.plot(cdfB, color='blue',  label='Blue')
        if show_green: ax.plot(cdfG, color='green', label='Green')
        if show_red:   ax.plot(cdfR, color='red',   label='Red')
        ax.set(xlabel='Pixel Intensity', ylabel='Cumulative Probability', title='RGB CDF')
        ax.legend(); ax.grid(True)
        return fig

    @staticmethod
    def plot_cdf_gray(cdf):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(cdf, color='black')
        ax.set(xlabel='Pixel Intensity', ylabel='Cumulative Probability', title='Grayscale CDF')
        ax.grid(True)
        return fig

    @staticmethod
    def equalize_gray(gray_image):
        """Self-contained: computes hist and cdf internally."""
        hist = Histogram.computeHistoGray(gray_image)
        cdf = hist.cumsum()
        lookup_table = ((cdf / hist.sum()) * 255).astype('uint8')
        return lookup_table[gray_image]