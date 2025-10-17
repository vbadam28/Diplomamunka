from PySide6.QtWidgets import QWidget, QVBoxLayout


class ControlWindow(QWidget):
    def __init__(self, ctx, controls, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        layout = QVBoxLayout(self)
        self.controls = []
        for ctrl_cls in controls:
            ctrl = ctrl_cls(self.ctx,parent = self)
            self.controls.append(ctrl)
            layout.addWidget(ctrl)
        self.setLayout(layout)