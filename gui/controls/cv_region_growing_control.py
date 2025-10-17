from PySide6.QtWidgets import QVBoxLayout, QLabel

from gui.controls.control_panel import ControlPanel


class CVRegionGrowingControl(ControlPanel):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)

        self.ctx = ctx

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("OpenCv region growing"))

        self.addIntRange(layout,"low diff",[0,255],self.ctx.params.get("cvrg:lo_diff",75),"cvrg:lo_diff","up diff",[0,255],self.ctx.params.get("cvrg:up_diff",75),"cvrg:up_diff")