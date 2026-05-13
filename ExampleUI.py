from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from relaxation_metric import compute_relaxation_metric_from_bands
from focus import compute_bandpowers, compute_focus_metric_from_bands
from volume_control import map_metric_to_volume, set_system_volume_percent, TonePlayer
from relaxation_metric import RelaxationMetric

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800,600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        # Layout
        self.layout = QtWidgets.QVBoxLayout(self.centralwidget)

        # Checkbox
        self.checkBox = QtWidgets.QCheckBox("Plot Sine Wave")
        self.layout.addWidget(self.checkBox)

        # Matplotlib Figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.layout.addWidget(self.canvas)

        # Button to trigger plot
        self.button = QtWidgets.QPushButton("Plot")
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.plotonCanvas)

    def plotonCanvas(self):

        x = np.linspace(0,10,100)
        y = np.sin(x)

        self.figure.clear()

        ax = self.figure.add_subplot(111)
        ax.plot(x,y)

        ax.set_title("Example Signal")

        self.canvas.draw()


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()

    sys.exit(app.exec_())