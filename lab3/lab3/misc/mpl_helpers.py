import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class PdfPlotter:

    def __init__(self, filepath, closefigs=True):
        if closefigs:
            for fignum in plt.get_fignums():
                plt.close(fignum)

        self.plot_number = 0
        self.filepath = filepath
        self.figs = []
        if '_pdf_plotter' not in plt.__dict__.keys():
            plt._figure = plt.figure
            plt._show = plt.show
        plt.figure = self.figure
        plt.show = self.save
        plt._pdf_plotter = self

    def save(self):
        pp = PdfPages(self.filepath)
        for figure in self.figs:
            plt._figure(figure.number)
            plt.savefig(pp, format='pdf')
        pp.close()

    def figure(self, num=None, **kwargs):
        if num is not None:
            return plt._figure(num=num, **kwargs)

        fig = plt._figure(**kwargs)
        self.figs.append(fig)
        self.plot_number = 0

        return fig
