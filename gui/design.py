# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(2184, 1261)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("icon.ico")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(1000, 800))
        self.centralwidget.setStyleSheet(_fromUtf8(""))
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.verticalFrame_controls = QtGui.QFrame(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.verticalFrame_controls.sizePolicy().hasHeightForWidth())
        self.verticalFrame_controls.setSizePolicy(sizePolicy)
        self.verticalFrame_controls.setMinimumSize(QtCore.QSize(500, 800))
        self.verticalFrame_controls.setStyleSheet(_fromUtf8(""))
        self.verticalFrame_controls.setObjectName(_fromUtf8("verticalFrame_controls"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.verticalFrame_controls)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_2 = QtGui.QLabel(self.verticalFrame_controls)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_5.addWidget(self.label_2)
        self.lineEdit_outputDir = QtGui.QLineEdit(self.verticalFrame_controls)
        self.lineEdit_outputDir.setEnabled(False)
        self.lineEdit_outputDir.setObjectName(_fromUtf8("lineEdit_outputDir"))
        self.horizontalLayout_5.addWidget(self.lineEdit_outputDir)
        self.pushButton_outputDir = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_outputDir.setObjectName(_fromUtf8("pushButton_outputDir"))
        self.horizontalLayout_5.addWidget(self.pushButton_outputDir)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.pushButton_loadFMRI = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_loadFMRI.setObjectName(_fromUtf8("pushButton_loadFMRI"))
        self.horizontalLayout_9.addWidget(self.pushButton_loadFMRI)
        self.pushButton_loadStrucMRI = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_loadStrucMRI.setObjectName(_fromUtf8("pushButton_loadStrucMRI"))
        self.horizontalLayout_9.addWidget(self.pushButton_loadStrucMRI)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        self.gridLayout_5 = QtGui.QGridLayout()
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.listWidget_RSN = QtGui.QListWidget(self.verticalFrame_controls)
        self.listWidget_RSN.setObjectName(_fromUtf8("listWidget_RSN"))
        self.gridLayout_5.addWidget(self.listWidget_RSN, 4, 4, 1, 1)
        self.pushButton_rsnload = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_rsnload.setObjectName(_fromUtf8("pushButton_rsnload"))
        self.gridLayout_5.addWidget(self.pushButton_rsnload, 3, 4, 1, 1)
        self.pushButton_icaload = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_icaload.setObjectName(_fromUtf8("pushButton_icaload"))
        self.gridLayout_5.addWidget(self.pushButton_icaload, 3, 0, 1, 1)
        self.label_6 = QtGui.QLabel(self.verticalFrame_controls)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout_5.addWidget(self.label_6, 2, 4, 1, 1)
        self.label_5 = QtGui.QLabel(self.verticalFrame_controls)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_5.addWidget(self.label_5, 4, 2, 1, 1)
        self.listWidget_ICAComponents = QtGui.QListWidget(self.verticalFrame_controls)
        self.listWidget_ICAComponents.setObjectName(_fromUtf8("listWidget_ICAComponents"))
        self.gridLayout_5.addWidget(self.listWidget_ICAComponents, 4, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.verticalFrame_controls)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_5.addWidget(self.label_4, 2, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_5)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pushButton_runAnalysis = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_runAnalysis.setEnabled(True)
        self.pushButton_runAnalysis.setObjectName(_fromUtf8("pushButton_runAnalysis"))
        self.horizontalLayout.addWidget(self.pushButton_runAnalysis)
        self.pushButton_Plot = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_Plot.setObjectName(_fromUtf8("pushButton_Plot"))
        self.horizontalLayout.addWidget(self.pushButton_Plot)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.label_9 = QtGui.QLabel(self.verticalFrame_controls)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.verticalLayout_2.addWidget(self.label_9)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.lineEdit_ICANetwork = QtGui.QLineEdit(self.verticalFrame_controls)
        self.lineEdit_ICANetwork.setObjectName(_fromUtf8("lineEdit_ICANetwork"))
        self.horizontalLayout_2.addWidget(self.lineEdit_ICANetwork)
        self.label_8 = QtGui.QLabel(self.verticalFrame_controls)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.horizontalLayout_2.addWidget(self.label_8)
        self.lineEdit_mappedICANetwork = QtGui.QLineEdit(self.verticalFrame_controls)
        self.lineEdit_mappedICANetwork.setObjectName(_fromUtf8("lineEdit_mappedICANetwork"))
        self.horizontalLayout_2.addWidget(self.lineEdit_mappedICANetwork)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.pushButton_addNetwork = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_addNetwork.setObjectName(_fromUtf8("pushButton_addNetwork"))
        self.horizontalLayout_8.addWidget(self.pushButton_addNetwork)
        self.pushButton_rmNetwork = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_rmNetwork.setObjectName(_fromUtf8("pushButton_rmNetwork"))
        self.horizontalLayout_8.addWidget(self.pushButton_rmNetwork)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.listWidget_mappedICANetworks = QtGui.QListWidget(self.verticalFrame_controls)
        self.listWidget_mappedICANetworks.setObjectName(_fromUtf8("listWidget_mappedICANetworks"))
        self.verticalLayout_2.addWidget(self.listWidget_mappedICANetworks)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.pushButton_reset = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_reset.setObjectName(_fromUtf8("pushButton_reset"))
        self.horizontalLayout_7.addWidget(self.pushButton_reset)
        self.pushButton_createReport = QtGui.QPushButton(self.verticalFrame_controls)
        self.pushButton_createReport.setEnabled(True)
        self.pushButton_createReport.setObjectName(_fromUtf8("pushButton_createReport"))
        self.horizontalLayout_7.addWidget(self.pushButton_createReport)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_3.addWidget(self.verticalFrame_controls)
        self.scrollArea = QtGui.QScrollArea(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(9)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1653, 1146))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.slice_frame = QtGui.QFrame(self.scrollAreaWidgetContents)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(9)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slice_frame.sizePolicy().hasHeightForWidth())
        self.slice_frame.setSizePolicy(sizePolicy)
        self.slice_frame.setStyleSheet(_fromUtf8(""))
        self.slice_frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.slice_frame.setFrameShadow(QtGui.QFrame.Raised)
        self.slice_frame.setObjectName(_fromUtf8("slice_frame"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.slice_frame)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.groupBox_2 = QtGui.QGroupBox(self.slice_frame)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.pushButton_zslice = QtGui.QPushButton(self.groupBox_2)
        self.pushButton_zslice.setEnabled(False)
        self.pushButton_zslice.setMinimumSize(QtCore.QSize(200, 0))
        self.pushButton_zslice.setMaximumSize(QtCore.QSize(200, 16777215))
        self.pushButton_zslice.setObjectName(_fromUtf8("pushButton_zslice"))
        self.verticalLayout.addWidget(self.pushButton_zslice)
        self.radioButton_ortho = QtGui.QRadioButton(self.groupBox_2)
        self.radioButton_ortho.setChecked(True)
        self.radioButton_ortho.setObjectName(_fromUtf8("radioButton_ortho"))
        self.buttonGroup_xview = QtGui.QButtonGroup(MainWindow)
        self.buttonGroup_xview.setObjectName(_fromUtf8("buttonGroup_xview"))
        self.buttonGroup_xview.addButton(self.radioButton_ortho)
        self.verticalLayout.addWidget(self.radioButton_ortho)
        self.radioButton_axial = QtGui.QRadioButton(self.groupBox_2)
        self.radioButton_axial.setObjectName(_fromUtf8("radioButton_axial"))
        self.buttonGroup_xview.addButton(self.radioButton_axial)
        self.verticalLayout.addWidget(self.radioButton_axial)
        self.radioButton_sagittal = QtGui.QRadioButton(self.groupBox_2)
        self.radioButton_sagittal.setObjectName(_fromUtf8("radioButton_sagittal"))
        self.buttonGroup_xview.addButton(self.radioButton_sagittal)
        self.verticalLayout.addWidget(self.radioButton_sagittal)
        self.radioButton_coronal = QtGui.QRadioButton(self.groupBox_2)
        self.radioButton_coronal.setObjectName(_fromUtf8("radioButton_coronal"))
        self.buttonGroup_xview.addButton(self.radioButton_coronal)
        self.verticalLayout.addWidget(self.radioButton_coronal)
        self.horizontalLayout_14 = QtGui.QHBoxLayout()
        self.horizontalLayout_14.setObjectName(_fromUtf8("horizontalLayout_14"))
        self.label = QtGui.QLabel(self.groupBox_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(125, 0))
        self.label.setMaximumSize(QtCore.QSize(125, 16777215))
        self.label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_14.addWidget(self.label)
        self.spinBox_numSlices = QtGui.QSpinBox(self.groupBox_2)
        self.spinBox_numSlices.setMinimumSize(QtCore.QSize(70, 0))
        self.spinBox_numSlices.setMaximumSize(QtCore.QSize(70, 16777215))
        self.spinBox_numSlices.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_numSlices.setProperty("value", 5)
        self.spinBox_numSlices.setObjectName(_fromUtf8("spinBox_numSlices"))
        self.horizontalLayout_14.addWidget(self.spinBox_numSlices)
        self.verticalLayout.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.frame = QtGui.QFrame(self.groupBox_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 75))
        self.frame.setObjectName(_fromUtf8("frame"))
        self.horizontalLayout_xcontrols = QtGui.QHBoxLayout(self.frame)
        self.horizontalLayout_xcontrols.setObjectName(_fromUtf8("horizontalLayout_xcontrols"))
        self.verticalLayout_6 = QtGui.QVBoxLayout()
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.horizontalSlider_Xslice = QtGui.QSlider(self.frame)
        self.horizontalSlider_Xslice.setMinimum(-78)
        self.horizontalSlider_Xslice.setMaximum(78)
        self.horizontalSlider_Xslice.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Xslice.setObjectName(_fromUtf8("horizontalSlider_Xslice"))
        self.verticalLayout_6.addWidget(self.horizontalSlider_Xslice)
        self.label_Xslice = QtGui.QLabel(self.frame)
        self.label_Xslice.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Xslice.setObjectName(_fromUtf8("label_Xslice"))
        self.verticalLayout_6.addWidget(self.label_Xslice)
        self.horizontalLayout_xcontrols.addLayout(self.verticalLayout_6)
        self.verticalLayout_7 = QtGui.QVBoxLayout()
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.horizontalSlider_Yslice = QtGui.QSlider(self.frame)
        self.horizontalSlider_Yslice.setMinimum(-112)
        self.horizontalSlider_Yslice.setMaximum(76)
        self.horizontalSlider_Yslice.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Yslice.setObjectName(_fromUtf8("horizontalSlider_Yslice"))
        self.verticalLayout_7.addWidget(self.horizontalSlider_Yslice)
        self.label_Yslice = QtGui.QLabel(self.frame)
        self.label_Yslice.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Yslice.setObjectName(_fromUtf8("label_Yslice"))
        self.verticalLayout_7.addWidget(self.label_Yslice)
        self.horizontalLayout_xcontrols.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtGui.QVBoxLayout()
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.horizontalSlider_Zslice = QtGui.QSlider(self.frame)
        self.horizontalSlider_Zslice.setMinimum(-70)
        self.horizontalSlider_Zslice.setMaximum(85)
        self.horizontalSlider_Zslice.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Zslice.setObjectName(_fromUtf8("horizontalSlider_Zslice"))
        self.verticalLayout_8.addWidget(self.horizontalSlider_Zslice)
        self.label_Zslice = QtGui.QLabel(self.frame)
        self.label_Zslice.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Zslice.setObjectName(_fromUtf8("label_Zslice"))
        self.verticalLayout_8.addWidget(self.label_Zslice)
        self.horizontalLayout_xcontrols.addLayout(self.verticalLayout_8)
        self.verticalLayout_4.addWidget(self.frame)
        self.groupBox = QtGui.QGroupBox(self.groupBox_2)
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 108))
        self.groupBox.setTitle(_fromUtf8(""))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.checkBox_showWM = QtGui.QCheckBox(self.groupBox)
        self.checkBox_showWM.setEnabled(True)
        self.checkBox_showWM.setObjectName(_fromUtf8("checkBox_showWM"))
        self.horizontalLayout_6.addWidget(self.checkBox_showWM)
        self.pushButton_loadWM = QtGui.QPushButton(self.groupBox)
        self.pushButton_loadWM.setMinimumSize(QtCore.QSize(80, 0))
        self.pushButton_loadWM.setMaximumSize(QtCore.QSize(80, 16777215))
        self.pushButton_loadWM.setObjectName(_fromUtf8("pushButton_loadWM"))
        self.horizontalLayout_6.addWidget(self.pushButton_loadWM)
        self.gridLayout_2.addLayout(self.horizontalLayout_6, 0, 0, 1, 1)
        self.checkBox_tindplot = QtGui.QCheckBox(self.groupBox)
        self.checkBox_tindplot.setObjectName(_fromUtf8("checkBox_tindplot"))
        self.gridLayout_2.addWidget(self.checkBox_tindplot, 1, 1, 1, 1)
        self.checkBox_tgroupplot = QtGui.QCheckBox(self.groupBox)
        self.checkBox_tgroupplot.setEnabled(False)
        self.checkBox_tgroupplot.setObjectName(_fromUtf8("checkBox_tgroupplot"))
        self.gridLayout_2.addWidget(self.checkBox_tgroupplot, 1, 0, 1, 1)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.checkBox_showSegmentations = QtGui.QCheckBox(self.groupBox)
        self.checkBox_showSegmentations.setEnabled(False)
        self.checkBox_showSegmentations.setObjectName(_fromUtf8("checkBox_showSegmentations"))
        self.horizontalLayout_11.addWidget(self.checkBox_showSegmentations)
        self.pushButton_loadSegmentation = QtGui.QPushButton(self.groupBox)
        self.pushButton_loadSegmentation.setEnabled(False)
        self.pushButton_loadSegmentation.setMinimumSize(QtCore.QSize(80, 0))
        self.pushButton_loadSegmentation.setMaximumSize(QtCore.QSize(80, 16777215))
        self.pushButton_loadSegmentation.setObjectName(_fromUtf8("pushButton_loadSegmentation"))
        self.horizontalLayout_11.addWidget(self.pushButton_loadSegmentation)
        self.gridLayout_2.addLayout(self.horizontalLayout_11, 0, 4, 1, 1)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.checkBox_showGM = QtGui.QCheckBox(self.groupBox)
        self.checkBox_showGM.setEnabled(True)
        self.checkBox_showGM.setObjectName(_fromUtf8("checkBox_showGM"))
        self.horizontalLayout_10.addWidget(self.checkBox_showGM)
        self.pushButton_loadGM = QtGui.QPushButton(self.groupBox)
        self.pushButton_loadGM.setMinimumSize(QtCore.QSize(80, 0))
        self.pushButton_loadGM.setMaximumSize(QtCore.QSize(80, 16777215))
        self.pushButton_loadGM.setObjectName(_fromUtf8("pushButton_loadGM"))
        self.horizontalLayout_10.addWidget(self.pushButton_loadGM)
        self.gridLayout_2.addLayout(self.horizontalLayout_10, 0, 1, 1, 1)
        self.horizontalLayout_12 = QtGui.QHBoxLayout()
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.checkBox_showBrain = QtGui.QCheckBox(self.groupBox)
        self.checkBox_showBrain.setEnabled(True)
        self.checkBox_showBrain.setObjectName(_fromUtf8("checkBox_showBrain"))
        self.horizontalLayout_12.addWidget(self.checkBox_showBrain)
        self.pushButton_loadBrain = QtGui.QPushButton(self.groupBox)
        self.pushButton_loadBrain.setMinimumSize(QtCore.QSize(80, 0))
        self.pushButton_loadBrain.setMaximumSize(QtCore.QSize(80, 16777215))
        self.pushButton_loadBrain.setObjectName(_fromUtf8("pushButton_loadBrain"))
        self.horizontalLayout_12.addWidget(self.pushButton_loadBrain)
        self.gridLayout_2.addLayout(self.horizontalLayout_12, 0, 2, 1, 1)
        self.checkBox_tspectrum = QtGui.QCheckBox(self.groupBox)
        self.checkBox_tspectrum.setEnabled(False)
        self.checkBox_tspectrum.setObjectName(_fromUtf8("checkBox_tspectrum"))
        self.gridLayout_2.addWidget(self.checkBox_tspectrum, 1, 2, 1, 1)
        self.checkBox_taveplot = QtGui.QCheckBox(self.groupBox)
        self.checkBox_taveplot.setObjectName(_fromUtf8("checkBox_taveplot"))
        self.gridLayout_2.addWidget(self.checkBox_taveplot, 1, 3, 1, 2)
        self.horizontalLayout_13 = QtGui.QHBoxLayout()
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.checkBox_showCSF = QtGui.QCheckBox(self.groupBox)
        self.checkBox_showCSF.setObjectName(_fromUtf8("checkBox_showCSF"))
        self.horizontalLayout_13.addWidget(self.checkBox_showCSF)
        self.pushButton_loadCSF = QtGui.QPushButton(self.groupBox)
        self.pushButton_loadCSF.setMinimumSize(QtCore.QSize(80, 0))
        self.pushButton_loadCSF.setMaximumSize(QtCore.QSize(80, 16777215))
        self.pushButton_loadCSF.setObjectName(_fromUtf8("pushButton_loadCSF"))
        self.horizontalLayout_13.addWidget(self.pushButton_loadCSF)
        self.gridLayout_2.addLayout(self.horizontalLayout_13, 0, 3, 1, 1)
        self.verticalLayout_4.addWidget(self.groupBox)
        self.horizontalLayout_4.addLayout(self.verticalLayout_4)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.frame_plot = QtGui.QFrame(self.slice_frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(9)
        sizePolicy.setHeightForWidth(self.frame_plot.sizePolicy().hasHeightForWidth())
        self.frame_plot.setSizePolicy(sizePolicy)
        self.frame_plot.setMinimumSize(QtCore.QSize(800, 800))
        self.frame_plot.setObjectName(_fromUtf8("frame_plot"))
        self.verticalLayout_xplot = QtGui.QVBoxLayout(self.frame_plot)
        self.verticalLayout_xplot.setObjectName(_fromUtf8("verticalLayout_xplot"))
        self.verticalLayout_plot = QtGui.QVBoxLayout()
        self.verticalLayout_plot.setObjectName(_fromUtf8("verticalLayout_plot"))
        self.verticalLayout_xplot.addLayout(self.verticalLayout_plot)
        self.verticalLayout_3.addWidget(self.frame_plot)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout_3.addLayout(self.gridLayout)
        self.verticalLayout_5.addWidget(self.slice_frame)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout_3.addWidget(self.scrollArea)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 2184, 31))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuLoad_from_Script = QtGui.QMenu(self.menubar)
        self.menuLoad_from_Script.setObjectName(_fromUtf8("menuLoad_from_Script"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.action_LoadScript = QtGui.QAction(MainWindow)
        self.action_LoadScript.setObjectName(_fromUtf8("action_LoadScript"))
        self.actionICA_Files = QtGui.QAction(MainWindow)
        self.actionICA_Files.setObjectName(_fromUtf8("actionICA_Files"))
        self.actionRSN_Files = QtGui.QAction(MainWindow)
        self.actionRSN_Files.setObjectName(_fromUtf8("actionRSN_Files"))
        self.actionFMRI_File = QtGui.QAction(MainWindow)
        self.actionFMRI_File.setObjectName(_fromUtf8("actionFMRI_File"))
        self.actionStructural_fMRI = QtGui.QAction(MainWindow)
        self.actionStructural_fMRI.setObjectName(_fromUtf8("actionStructural_fMRI"))
        self.actionWhite_Matter_Mask = QtGui.QAction(MainWindow)
        self.actionWhite_Matter_Mask.setObjectName(_fromUtf8("actionWhite_Matter_Mask"))
        self.actionGray_Matter_Mask = QtGui.QAction(MainWindow)
        self.actionGray_Matter_Mask.setObjectName(_fromUtf8("actionGray_Matter_Mask"))
        self.actionSegmentations = QtGui.QAction(MainWindow)
        self.actionSegmentations.setObjectName(_fromUtf8("actionSegmentations"))
        self.actionBrain_Mask = QtGui.QAction(MainWindow)
        self.actionBrain_Mask.setObjectName(_fromUtf8("actionBrain_Mask"))
        self.actionBrain_Mask_2 = QtGui.QAction(MainWindow)
        self.actionBrain_Mask_2.setObjectName(_fromUtf8("actionBrain_Mask_2"))
        self.actionWhite_Matter = QtGui.QAction(MainWindow)
        self.actionWhite_Matter.setObjectName(_fromUtf8("actionWhite_Matter"))
        self.actionGray_Matter = QtGui.QAction(MainWindow)
        self.actionGray_Matter.setObjectName(_fromUtf8("actionGray_Matter"))
        self.menuLoad_from_Script.addAction(self.action_LoadScript)
        self.menuLoad_from_Script.addSeparator()
        self.menubar.addAction(self.menuLoad_from_Script.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "ICA to Network Mapping", None))
        self.label_2.setText(_translate("MainWindow", "Output Directory", None))
        self.pushButton_outputDir.setText(_translate("MainWindow", "Set", None))
        self.pushButton_loadFMRI.setText(_translate("MainWindow", "Load fMRI Data", None))
        self.pushButton_loadStrucMRI.setText(_translate("MainWindow", "Load structural MRI Data", None))
        self.pushButton_rsnload.setText(_translate("MainWindow", "Load RSNs", None))
        self.pushButton_icaload.setText(_translate("MainWindow", "Load ICA Components", None))
        self.label_6.setText(_translate("MainWindow", "Resting State Networks", None))
        self.label_5.setText(_translate("MainWindow", "< >", None))
        self.label_4.setText(_translate("MainWindow", "ICA Components", None))
        self.pushButton_runAnalysis.setText(_translate("MainWindow", "Run Analysis", None))
        self.pushButton_Plot.setText(_translate("MainWindow", "Plot", None))
        self.label_9.setText(_translate("MainWindow", "Mapped Networks", None))
        self.label_8.setText(_translate("MainWindow", ">", None))
        self.pushButton_addNetwork.setText(_translate("MainWindow", "Set Mapped Network", None))
        self.pushButton_rmNetwork.setText(_translate("MainWindow", "Delete Mapped Network", None))
        self.pushButton_reset.setText(_translate("MainWindow", "Reset", None))
        self.pushButton_createReport.setText(_translate("MainWindow", "Generate Report", None))
        self.groupBox_2.setTitle(_translate("MainWindow", "Plot Options", None))
        self.pushButton_zslice.setText(_translate("MainWindow", "Show Largest Overlapping", None))
        self.radioButton_ortho.setText(_translate("MainWindow", "Ortho", None))
        self.radioButton_axial.setText(_translate("MainWindow", "Axial", None))
        self.radioButton_sagittal.setText(_translate("MainWindow", "Sagittal", None))
        self.radioButton_coronal.setText(_translate("MainWindow", "Coronal", None))
        self.label.setText(_translate("MainWindow", "Number of Slices", None))
        self.label_Xslice.setText(_translate("MainWindow", "X", None))
        self.label_Yslice.setText(_translate("MainWindow", "Y", None))
        self.label_Zslice.setText(_translate("MainWindow", "Z", None))
        self.checkBox_showWM.setText(_translate("MainWindow", "Display White Matter", None))
        self.pushButton_loadWM.setText(_translate("MainWindow", "Load", None))
        self.checkBox_tindplot.setText(_translate("MainWindow", "Show Individual Time Series Data", None))
        self.checkBox_tgroupplot.setText(_translate("MainWindow", "Show Compoosite Time-Series", None))
        self.checkBox_showSegmentations.setText(_translate("MainWindow", "Display Segmentations", None))
        self.pushButton_loadSegmentation.setText(_translate("MainWindow", "Load", None))
        self.checkBox_showGM.setText(_translate("MainWindow", "Display Gray Matter", None))
        self.pushButton_loadGM.setText(_translate("MainWindow", "Load", None))
        self.checkBox_showBrain.setText(_translate("MainWindow", "Brain Boundary", None))
        self.pushButton_loadBrain.setText(_translate("MainWindow", "Load", None))
        self.checkBox_tspectrum.setText(_translate("MainWindow", "Show Power Spectrum", None))
        self.checkBox_taveplot.setText(_translate("MainWindow", "Show Spatially-Averagd Masked Time Series Data", None))
        self.checkBox_showCSF.setText(_translate("MainWindow", "Show CSF", None))
        self.pushButton_loadCSF.setText(_translate("MainWindow", "Load", None))
        self.menuLoad_from_Script.setTitle(_translate("MainWindow", "Load", None))
        self.action_LoadScript.setText(_translate("MainWindow", "From settings file", None))
        self.actionICA_Files.setText(_translate("MainWindow", "ICA Files", None))
        self.actionRSN_Files.setText(_translate("MainWindow", "RSN Files", None))
        self.actionFMRI_File.setText(_translate("MainWindow", "Functional MRI", None))
        self.actionStructural_fMRI.setText(_translate("MainWindow", "Structural fMRI", None))
        self.actionWhite_Matter_Mask.setText(_translate("MainWindow", "White Matter Mask", None))
        self.actionGray_Matter_Mask.setText(_translate("MainWindow", "Gray Matter Mask", None))
        self.actionSegmentations.setText(_translate("MainWindow", "Segmentations", None))
        self.actionBrain_Mask.setText(_translate("MainWindow", "Brain Mask", None))
        self.actionBrain_Mask_2.setText(_translate("MainWindow", "Brain Mask", None))
        self.actionWhite_Matter.setText(_translate("MainWindow", "White Matter", None))
        self.actionGray_Matter.setText(_translate("MainWindow", "Gray Matter", None))

