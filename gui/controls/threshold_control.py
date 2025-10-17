from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider

from gui.controls.control_panel import ControlPanel


class ThresholdControl(ControlPanel):
    def __init__(self, ctx, key="threshold",parent = None):
        super().__init__(parent)
        self.ctx = ctx
        self.key = key

        layout = QVBoxLayout(self)



        self.label = QLabel("Threshold")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0,255)
        self.slider.setValue(self.ctx.params.get(self.key,128))



        layout.addWidget(self.label)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connet(lambda v: self.ctx.params.update({self.key: v}))