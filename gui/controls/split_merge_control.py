from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QSpinBox, QHBoxLayout, QDoubleSpinBox

from gui.controls.control_panel import ControlPanel


class SplitMergeControl(ControlPanel):
    def __init__(self, ctx, key="quadtree_depth", parent = None):
        super().__init__(parent)
        self.ctx = ctx
        self.key = key

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel('QuadTree'))




        self.addIntControl(layout, "Depth",[1,8],self.ctx.params.get("qt:depth", 3),"qt:depth","spinBox")

        #self.addFloatControl(layout, "Hyperintense Min",[0,1],self.ctx.get("qt:hyperMin", 0.8),"qt:hyperMin")
        #self.addFloatControl(layout, "Hyperintense Max",[0,1],self.ctx.get("qt:hyperMax", 1.0),"qt:hyperMax")

        self.addFloatRange(layout,"Hyperintense Min",[0,1],self.ctx.params.get("qt:hyperMin", 0.82),"qt:hyperMin", "Hyperintense Max",[0,1],self.ctx.params.get("qt:hyperMax", 1.0),"qt:hyperMax")

        #self.addFloatControl(layout, "Hypointense Min", [0, 1], self.ctx.get("qt:hypoMin", 0.8), "qt:hypoMin")
        #self.addFloatControl(layout, "Hypointense Max", [0, 1], self.ctx.get("qt:hypoMax", 1.0), "qt:hypoMax")

        self.addFloatRange(layout, "Hypointense Min", [0, 1], self.ctx.params.get("qt:hypoMin", 0.05), "qt:hypoMin", "Hypointense Max", [0, 1], self.ctx.params.get("qt:hypoMax", 0.14), "qt:hypoMax")

        self.addFloatControl(layout, "Mean Hyper Threshold", [0, 1], self.ctx.params.get("qt:meanHyperTresh", 0.842), "qt:meanHyperTresh")
        self.addFloatControl(layout, "Mean Hypo Threshold", [0, 1], self.ctx.params.get("qt:meanHypoTresh", 0.09), "qt:meanHypoTresh")

        self.addIntControl(layout, "Sum Hyper Thres", [1, 1000], self.ctx.params.get("qt:sumHyperThresh", 300), "qt:sumHyperThresh")
        self.addIntControl(layout, "Sum Hypo Thres", [1, 200], self.ctx.params.get("qt:sumHypoThresh", 100), "qt:sumHypoThresh")

