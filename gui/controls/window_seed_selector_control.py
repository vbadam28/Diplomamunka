from PySide6.QtWidgets import QVBoxLayout

from gui.controls.control_panel import ControlPanel


class WindowSeedSelectorControl(ControlPanel):
    def __init__(self, ctx, parent = None):
        super().__init__(parent)

        self.ctx = ctx

        layout = QVBoxLayout(self)

        self.addComboBox(layout, "Mode",["center", "max"],ctx.params.get("wss:mode","center"),"wss:mode")