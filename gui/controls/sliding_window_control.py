from PySide6.QtWidgets import QVBoxLayout, QLabel

from gui.controls.control_panel import ControlPanel


class SlidingWindowControl(ControlPanel):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Sliding Window Control"))

        self.addIntControl(layout, "Step",[1,240],self.ctx.params.get("sw:step",4),"sw:step")
        self.addIntRange(layout,"Window Height",[1,120],self.ctx.params.get("sw:w_height",8),"sw:w_height","Window Width",[1,120],self.ctx.params.get("sw:w_width",8),"sw:w_width")

        self.addIntControl(layout, "Top K Window ",[1,100],self.ctx.params.get("sw:top_k",5),"sw:top_k")

        self.addComboBox(layout, "Mode",["mean","max","std","blob","entropy"], self.ctx.params.get("sw:mode","mean"),"sw:mode")