from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QSpinBox, QHBoxLayout, QDoubleSpinBox, QComboBox


class ControlPanel(QWidget):
    def __init__(self,parent = None):
        super().__init__(parent)
        self.ctx = None
    def addIntControl(self,layout, label,range,value, key, type="both" ):
        ctrLayout = QHBoxLayout()

        label = QLabel(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(range[0],range[1])
        slider.setValue(value)

        spinBox = QSpinBox()
        spinBox.setRange(range[0],range[1])
        spinBox.setValue(value)

        if type == "both":
            slider.valueChanged.connect(lambda v: spinBox.setValue(v))
            spinBox.valueChanged.connect(lambda  v: slider.setValue(v))


        if type=="both" or type=="slider":
            ctrLayout.addWidget(slider)
        if type=="both" or type=="spinBox":
            ctrLayout.addWidget(spinBox)
            spinBox.valueChanged.connect(lambda v: self.ctx.params.update({key: v}))
        else:
            slider.valueChanged.connect(lambda v: self.ctx.params.update({key: v}))

        layout.addWidget(label)
        layout.addLayout(ctrLayout)
        return label, slider, spinBox

    def addFloatControl(self, layout, label, range, value, key, type="both"):
        ctrLayout = QHBoxLayout()

        label = QLabel(label)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(range[0]*1000), int(range[1]*1000))
        slider.setValue(int(value*1000))

        spinBox = QDoubleSpinBox()
        spinBox.setRange(range[0], range[1])
        spinBox.setValue(value)


        slider.valueChanged.connect(lambda v: spinBox.setValue(v/1000))
        spinBox.valueChanged.connect(lambda v: slider.setValue(int(v*1000)))

        spinBox.valueChanged.connect(lambda v: self.ctx.params.update({key: v}))

        if type == "both" or type == "slider":
            ctrLayout.addWidget(slider)
        if type == "both" or type == "spinBox":
            ctrLayout.addWidget(spinBox)

        layout.addWidget(label)
        layout.addLayout(ctrLayout)
        return label, slider, spinBox

    def addFloatRange(self,  layout, minLabel, minRange, minValue, minKey,maxLabel, maxRange, maxValue, maxKey, type="both"):
        localLayout = QHBoxLayout()

        minLabel, minSlider, minSpinBox = self.addFloatControl(localLayout, minLabel, minRange, minValue, minKey, type="both")
        maxLabel, maxSlider, maxSpinBox = self.addFloatControl(localLayout, maxLabel, maxRange, maxValue, maxKey, type="both")

        layout.addLayout(localLayout)
        return (minLabel, maxLabel), (minSlider, maxSlider), (minSpinBox, maxSpinBox)

    def addIntRange(self,  layout, minLabel, minRange, minValue, minKey,maxLabel, maxRange, maxValue, maxKey, type="both"):
        localLayout = QHBoxLayout()

        minLabel, minSlider, minSpinBox = self.addIntControl(localLayout, minLabel, minRange, minValue, minKey, type="both")
        maxLabel, maxSlider, maxSpinBox = self.addIntControl(localLayout, maxLabel, maxRange, maxValue, maxKey, type="both")

        layout.addLayout(localLayout)
        return (minLabel, maxLabel), (minSlider, maxSlider), (minSpinBox, maxSpinBox)

    def addComboBox(self,layout, label, range, value, key):

        ctrLayout = QHBoxLayout()

        label = QLabel(label)

        comboBox = QComboBox()
        comboBox.addItems(range)
        comboBox.setCurrentText(value)
        #comboBox.setCurrentIndex(range.items().index(value))


        ctrLayout.addWidget(comboBox)
        comboBox.currentIndexChanged.connect(lambda i: self.ctx.params.update({key: range[i]}))


        layout.addWidget(label)
        layout.addLayout(ctrLayout)
        return label, comboBox
