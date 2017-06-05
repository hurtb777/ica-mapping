import os  # For listing directory methods
from os.path import join as opj  # method to join strings of file paths
import sys  # We need sys so that we can pass argv to QApplication
import json
import numpy as np  # Library to for all mathematical operations
from nilearn import plotting, image, input_data  # library for neuroimaging
from PyQt4 import QtGui, QtCore, Qt  # Import QT
from functools import partial
import re  # library for regular expressions
from nibabel.nifti1 import Nifti1Image

# import sip
# sip.setapi('QVariant', 2)
import nipype.interfaces.io as nio

import matplotlib.pyplot as plt  # Plotting library
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from settings import mri_plots as mp, time_plots as tp  # use items in settings file
import design  # This file holds our MainWindow and all design related things
import mapper as map
from reports import create_html

ANATOMICAL_TO_TIMESERIES_PLOT_RATIO = 5
CONFIGURATION_FILE = '../config.json'

class MapperGUI(QtGui.QMainWindow, design.Ui_MainWindow):
    """
    Mapping GUI
    """
    def __init__(self, configuration_file=None):
        super(self.__class__, self).__init__()  # Runs the initialization of the base classes (.QMainWindow and design.UI_MainWindow)
        self.setupUi(self)  # This is defined in design.py file automatically; created in QT Designer
        self.gd = {}  # gui data; dict of dict where gd[class][unique_name][file_path, nilearn image object]
        self.computed_analysis = False
        cfile = configuration_file if isinstance(configuration_file, str) else CONFIGURATION_FILE

        anat_sp = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        anat_sp.setVerticalStretch(ANATOMICAL_TO_TIMESERIES_PLOT_RATIO)
        ts_sp = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        ts_sp.setVerticalStretch(1)

        # figure instance to view spatial data
        self.figure_x = plt.figure()
        self.canvas_x = FigureCanvas(self.figure_x)
        self.verticalLayout_plot.addWidget(self.canvas_x)
        self.canvas_x.setSizePolicy(anat_sp)

        # fig
        self.figure_t = plt.figure()
        self.canvas_t = FigureCanvas(self.figure_t)
        self.verticalLayout_plot.addWidget(self.canvas_t)
        self.canvas_t.setSizePolicy(ts_sp)

        self.reset_gui()
        self.load_configuration_file(cfile)

        # Connections
        self.action_LoadScript.triggered.connect(self._load_configuration)
        self.pushButton_outputDir.clicked.connect(self.browse_output_directory)
        self.pushButton_loadFMRI.clicked.connect(partial(self.browse_file, 'fmri'))
        self.pushButton_loadStrucMRI.clicked.connect(partial(self.browse_file, 'smri'))
        self.pushButton_icaload.clicked.connect(partial(self.browse_folder, self.listWidget_ICAComponents,
                                                        search_pattern=self.config['ica']['search_pattern'],
                                                        title="Select ICA Component Directory",
                                                        list_name="ica",
                                                       )
                                               )  # When the button is pressed
        self.pushButton_rsnload.clicked.connect(partial(self.browse_folder, self.listWidget_RSN,
                                                        search_pattern=self.config['rsn']['search_pattern'],
                                                        title="Select RSN Directory",
                                                        list_name="rsn",    
                                                       )
                                               )  # When the button is pressed
        self.pushButton_loadGM.clicked.connect(partial(self.browse_file, 'gm_mask'))
        self.pushButton_loadWM.clicked.connect(partial(self.browse_file, 'wm_mask'))
        self.pushButton_loadCSF.clicked.connect(partial(self.browse_file, 'csf_mask'))
        self.pushButton_loadBrain.clicked.connect(partial(self.browse_file, 'brain_mask'))
        self.pushButton_loadSegmentation.clicked.connect(partial(self.browse_file, 'segmentation'))
        self.pushButton_Plot.clicked.connect(self.update_plots)

        self.listWidget_ICAComponents.itemClicked.connect(self.update_gui)
        self.listWidget_RSN.itemClicked.connect(self.update_gui)
        self.listWidget_mappedICANetworks.itemClicked.connect(self.update_mapping)
        self.listWidget_mappedICANetworks.currentItemChanged.connect(self.update_mapping)

        self.pushButton_addNetwork.clicked.connect(self.add_mapped_network)
        self.pushButton_rmNetwork.clicked.connect(self.delete_mapped_network)
        self.pushButton_reset.clicked.connect(self.reset_gui)
        self.pushButton_runAnalysis.clicked.connect(self.run_analysis)
        self.pushButton_createReport.clicked.connect(self.generate_report)
        
        self.horizontalSlider_Xslice.sliderReleased.connect(self.update_plots)
        self.horizontalSlider_Yslice.sliderReleased.connect(self.update_plots)
        self.horizontalSlider_Zslice.sliderReleased.connect(self.update_plots)
        self.spinBox_numSlices.valueChanged.connect(self.update_plots)

        self.listWidget_ICAComponents.setCurrentRow(0)
        self.listWidget_RSN.setCurrentRow(0)
        self.mapper = map.Mapper(map_files=self.get_imgobjects('rsn'), in_files=self.get_imgobjects('ica'))

    def update_gui(self):
        ica_name = str(self.listWidget_ICAComponents.currentItem().text())
        rsn_name = str(self.listWidget_RSN.currentItem().text())
        # print self.mapper.corr
        if ica_name in self.mapper.corr.keys():
            new_order = np.argsort(self.mapper.corr[ica_name])[::-1]
            for i in range(self.listWidget_RSN.count()):
                item = self.listWidget_RSN.item(i)
                lookup = str(item.data(QtCore.Qt.UserRole).toString())

                item.setText("%s (%0.2f)" %(lookup, self.mapper.corr[ica_name][i]))
                # self.listWidget_RSN.setIt
        else:
            for i in range(self.listWidget_RSN.count()):
                item = self.listWidget_RSN.item(i)
                lookup = str(item.data(QtCore.Qt.UserRole).toString())
                item.setText(lookup)

        # if rsn_name in self.
        if self.computed_analysis:
            # cfname = self.gd['rsn']['filepath']
            pass
        
        self.lineEdit_ICANetwork.setText(ica_name)
        self.lineEdit_mappedICANetwork.setText(rsn_name)
        
    def update_mapping(self):
        ica_lookup, rsn_lookup = self.get_current_networks()
        self.listWidget_ICAComponents.setCurrentItem(self.gd['mapped'][ica_lookup]['ica_item'])
        self.listWidget_RSN.setCurrentItem(self.gd['mapped'][ica_lookup]['rsn_item'])
        # self.listWidget_ICAComponents.setCurrentItem(self.gd['mapped']['ica_name']['ica_item'])
        
        self.lineEdit_ICANetwork.setText(self.gd['mapped'][ica_lookup]['ica_custom_name'])
        self.lineEdit_mappedICANetwork.setText(self.gd['mapped'][ica_lookup]['rsn_custom_name'])
        
    def reset_gui(self):
        self.listWidget_mappedICANetworks.clear()
        self.gd = {'fmri': {}, 'smri': {},'ica': {}, 'rsn': {}, 'mapped': {}}

    def add_mapped_network(self):
        ica_lookup, rsn_lookup = self.get_current_networks()
        ica_custom_name = self.lineEdit_ICANetwork.text()
        rsn_custom_name = self.lineEdit_mappedICANetwork.text()
        name = "%s > %s" %(ica_custom_name, rsn_custom_name)
        if ica_lookup not in self.gd['mapped'].keys():  # not yet mapped by user
            map_itemWidget = QtGui.QListWidgetItem(name)
            self.listWidget_mappedICANetworks.addItem(map_itemWidget)  
        else:  # overwrite existing mapping; get row in mapped listWidget, overwrite the label
            map_itemWidget = self.gd['mapped'][ica_lookup]['mapped_item']
            map_itemWidget.setText(name)
            self.listWidget_mappedICANetworks.setCurrentItem(map_itemWidget)
        self.gd['mapped'].update(
             {ica_lookup:  # 1 Mapped Network per ICA (string linked to the listWidgetItem)
              {'rsn_lookup': rsn_lookup, #  RSN name (string linked to the listWidgetItem)
               'custom_name': name,  # Composite user's custom Name
               'ica_custom_name': ica_custom_name,  # User's Name
               'rsn_custom_name': rsn_custom_name,  # User's Name
               'ica_item': self.listWidget_ICAComponents.currentItem(),  # Index of the ICA listWidget 
               'rsn_item' : self.listWidget_RSN.currentItem(),  # Index of the RSN listWidget
               'mapped_item' : map_itemWidget,  # QListWidgetItem in the mapped listWidget
              }
             }
        )
        self.update_mapping()
        
    def delete_mapped_network(self):
        for item in self.listWidget_mappedICANetworks.selectedItems():
            ica_name =self.listWidget_ICAComponents.currentItem().text()
            self.listWidget_mappedICANetworks.takeItem(self.listWidget_mappedICANetworks.row(item))
            del self.gd['mapped'][ica_name]
            
    def browse_file(self, file_type, pushButton=None):
        fname = QtGui.QFileDialog.getOpenFileName(self, caption="Open Image", directory=".",
                                                  filter="Image Files (*.nii.gz *.nii)")
        self.load_base_file(fname, file_type=file_type)

    def load_base_file(self, file_name, file_type='fmri'):
        if file_name:
            self.gd.update({file_type: {'full_path': file_name, 'img': image.load_img(str(file_name))}})

    def browse_folder(self, listWidget=None, search_pattern='\w+', title="Select Directory", list_name='ica'):
        directory = QtGui.QFileDialog.getExistingDirectory(self, title)
        if search_pattern is not None:
            self.find_files(directory, "*", search_pattern, listWidget, list_name)
        return directory

    def browse_output_directory(self):
        directory = QtGui.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.lineEdit_outputDir.setText(directory)

    def find_files(self, directory, template, search_pattern, listWidget, list_name, extra_items=None):
        if directory: # if user didn't pick a directory don't continue
            ds = nio.DataGrabber(base_directory=directory, template=template, sort_filelist=True)
            all_nifti_files = ds.run().outputs.outfiles
            listWidget.clear() # In case there are any existing elements in the list
            r = re.compile(search_pattern)
            list_items = []
            for file_name in filter(r.search, all_nifti_files):
                match = re.search(r, file_name)
                lookup_key = match.groups()[0]
                item = QtGui.QListWidgetItem(lookup_key)
                listWidget.addItem(item)
                item.setData(QtCore.Qt.UserRole, lookup_key)
                self.gd[list_name][lookup_key] = {'img': image.load_img(opj(directory, file_name)),
                                                  'filepath': opj(directory, file_name),
                                                  'name': lookup_key,
                                                  'widget': item}
            if extra_items:
                for extra in extra_items:
                    item = QtGui.QListWidgetItem(extra)
                    listWidget.addItem(item)
                    item.setData(QtCore.Qt.UserRole, extra)
                    self.gd[list_name][extra] = {'img': None, 'filepath': None, 'name': extra, 'widget': item}

    def _load_configuration(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, caption="Open Input File", directory=".",
                                                  filter="Configuration File (*.json)")
        self.load_configuration_file(fname)

    def load_configuration_file(self, fname):
        """Load from configuration (.ini) file."""
        with open(fname) as json_config:
            data = json.load(json_config)

        if os.path.exists(data['fmri_file']):
            self.load_base_file(data['fmri_file'], file_type='fmri')
        if os.path.exists(data['smri_file']):
            self.load_base_file(data['smri_file'], file_type='smri')
        if os.path.exists(data['masks']['gm']):
            self.load_base_file(data['masks']['gm'], file_type='gm_mask')
        if os.path.exists(data['masks']['wm']):
            self.load_base_file(data['masks']['wm'], file_type='wm_mask')
        if os.path.exists(data['masks']['csf']):
            self.load_base_file(data['masks']['csf'], file_type='csf_mask')
        if os.path.exists(data['masks']['brain']):
            self.load_base_file(data['masks']['brain'], file_type='brain_mask')
        if os.path.exists(data['ica']['directory']):
            self.find_files(data['ica']['directory'], data['ica']['template'], data['ica']['search_pattern'],
                            self.listWidget_ICAComponents, list_name='ica')
        if os.path.exists(data['rsn']['directory']):
            self.find_files(data['rsn']['directory'], data['rsn']['template'], data['rsn']['search_pattern'],
                            self.listWidget_RSN, list_name='rsn', extra_items=data['rsn']['extra_items'])
        if os.path.exists(data['output_directory']):
            self.lineEdit_outputDir.setText(os.path.abspath(data['output_directory']))
        self.config = data

    def run_analysis(self):
        btn_txt = self.pushButton_runAnalysis.text()
        self.pushButton_runAnalysis.setText("Creating")
        lookup_val = str(self.listWidget_ICAComponents.currentItem().data(QtCore.Qt.UserRole).toString())
        corr = self.mapper.run_one(self.gd['ica'][lookup_val]['img'], label=lookup_val) \
            if lookup_val not in self.mapper.corr.keys() else self.mapper.corr[lookup_val]
        self.update_gui()
        self.computed_analysis = True
        self.pushButton_runAnalysis.setText(btn_txt)

    def generate_report(self):
        btn_txt = self.pushButton_createReport.text()
        self.pushButton_createReport.setText("Creating")
        fx = plt.figure(figsize=(10,4))
        ft = plt.figure(figsize=(10, 2))
        directory = str(self.lineEdit_outputDir.text())
        if not os.path.exists(directory):
            os.makedirs(directory)
        for u, v in self.gd['mapped'].iteritems():
            options = self.get_plot_options(ica_lookup=u, rsn_lookup=v['rsn_lookup'])
            self.plot_x(fx, **options)
            self.plot_t(ft, **options)
            context = {'mri_view': fx, 'time_series': ft, 'title': v['custom_name']}
            fname = opj(directory, '%s_out.html' % u)
            create_html(context, fname)
        self.pushButton_createReport.setText(btn_txt)

    def plot_x(self, fig, ica_lookup, rsn_lookup, display='ortho', coords=(0,0,0), show_rsn=True, show_wm=False,
               show_csf=False, show_gm=False, show_brain=False, show_segmentation=False, *args, **kwargs):
        anat_img = self.gd['smri']['img']
        stat_img = self.gd['ica'][ica_lookup]['img']
        fig.clear()
        ax1 = fig.add_subplot(111)
        ax1.hold(False)  # discards the old graph

        d = plotting.plot_stat_map(stat_map_img=stat_img, bg_img=anat_img, axes=ax1,
                                   title='ICA Component %s' % ica_lookup,
                                   cut_coords=coords, display_mode=display, annotate=True,
                                   draw_cross=True, colorbar=True)
        if show_rsn:
            d.add_contours(self.gd['rsn'][rsn_lookup]['img'], filled=mp['rsn']['filled'], alpha=mp['rsn']['alpha'],
                           levels=[mp['rsn']['levels']], colors=mp['rsn']['colors'])
        if show_wm:
            d.add_contours(self.gd['wm_mask']['img'], filled=mp['wm']['filled'], alpha=mp['wm']['alpha'],
                           levels=[mp['wm']['levels']], colors=mp['wm']['colors'])
        if show_csf:
            d.add_contours(self.gd['csf_mask']['img'], filled=mp['csf']['filled'], alpha=mp['csf']['alpha'],
                           levels=[mp['csf']['levels']], colors=mp['csf']['colors'])
        if show_gm:
            d.add_contours(self.gd['gm_mask']['img'], filled=mp['gm']['filled'], alpha=mp['gm']['alpha'],
                           levels=[mp['gm']['levels']], colors=mp['gm']['colors'])
        if show_brain:
            d.add_contours(self.gd['brain_mask']['img'], filled=mp['brain']['filled'], alpha=mp['brain']['alpha'],
                           levels=[mp['brain']['levels']], colors=mp['brain']['colors'])
        if show_segmentation:
            # TODO: Implement Segmentations
            # d.add_contours(self.gd['segmentations']['img'], filled=True, alpha=0.2, levels=[0.5], colors='w')
            pass
        ax1.set_axis_off()
        fig.tight_layout(pad=0.01)

    def plot_t(self, fig, show_time_individual=False, show_time_average=False, ica_lookup=None, show_spectrum=False, show_time_group=False,
               coords=(0,0,0), significance_threshold=0.5, *args, **kwargs):
        # Determine Axes Layout
        if not (show_time_individual or show_time_average) and not show_time_group and not show_spectrum:
            return  # no plot to render
        fig.clear()
        gs = gridspec.GridSpec(2, 5)
        if not (show_time_individual or show_time_average) and show_time_group and not show_spectrum:
            axgr = plt.subplot(gs[:, :])
        if not (show_time_individual or show_time_average) and show_time_group and show_spectrum:
            axgr, axps = plt.subplot(gs[:, 3:]), plt.subplot(gs[:, :3])
        if (show_time_individual or show_time_average) and not show_time_group and not show_spectrum:
            axts = plt.subplot(gs[:, :])
        if (show_time_individual or show_time_average) and show_time_group and not show_spectrum:
            axts, axgr = plt.subplot(gs[:, 3:]), plt.subplot(gs[:, :3])
        if (show_time_individual or show_time_average) and show_time_group and show_spectrum:
            axts, axgr, axps = plt.subplot(gs[:, 3:]), plt.subplot(gs[0, :3]), plt.subplot(gs[1, :3])

        # Process Data & Plot
        dat = np.abs(self.gd['ica'][ica_lookup]['img'].get_data().astype(np.float)) > significance_threshold
        masked = image.new_img_like(self.gd['smri']['img'], dat.astype(np.int))
        if show_time_individual:
            try:
                seed_masker = input_data.NiftiSpheresMasker(mask_img=masked, seeds=[coords], radius=0, detrend=False,
                                                            standardize=False, t_r=4., memory='nilearn_cache',
                                                            memory_level=1, verbose=0)
                ind_ts = seed_masker.fit_transform(self.gd['fmri']['img'])
            except:
                ind_ts = []
            plt.plot(ind_ts, axes=axts, label='Voxel (%d, %d, %d) Time-Series' % (coords[0], coords[1], coords[2]))
            plt.xlabel('Time (s)')
            plt.ylabel('fMRI signal')
            axts.hold(True)
        if show_time_average:
            brain_masker = input_data.NiftiMasker(mask_img=masked,
                                                  t_r=4.,memory='nilearn_cache', memory_level=1, verbose=0)

            ts = brain_masker.fit_transform(self.gd['fmri']['img'])
            ave_ts = np.mean(ts, axis=1)
            plt.plot(ave_ts, axes=axts, label="Average Signal")
            plt.xlabel('Time (s)')
            plt.ylabel('fMRI signal')
            axts.hold(False)
        if show_time_group:
            # TODO: Plot Group Logic
            pass
        if show_spectrum:
            # TODO: Plot Power Spectrum Logic
            pass
        plt.tight_layout(pad=0.1)

    def update_plots(self):
        ica_lookup, rsn_lookup = self.get_current_networks()
        options = self.get_plot_options(ica_lookup, rsn_lookup)
        self.plot_x(self.figure_x, **options)
        self.canvas_x.draw()
        self.plot_t(self.figure_t, **options)
        self.canvas_t.draw()

    def apply_slice_views(self):
        x, y, z = self.get_and_set_slice_coordinates()
        num_slices = int(self.spinBox_numSlices.text())
        if self.buttonGroup_xview.checkedButton() == self.radioButton_ortho:
            display = 'ortho'
            coords = (x, y, z)
        elif self.buttonGroup_xview.checkedButton() == self.radioButton_axial:
            display = 'z'
            coords = np.linspace(-50, 50, num_slices)
        elif self.buttonGroup_xview.checkedButton() == self.radioButton_coronal:
            display = 'y'
            coords = np.linspace(-50, 50, num_slices)
        else:  # Sagittal
            display = 'x'
            coords = np.linspace(-50, 50, num_slices)

        return display, coords

    def get_and_set_slice_coordinates(self):
        x, y, z = self.horizontalSlider_Xslice.value(), self.horizontalSlider_Yslice.value(), self.horizontalSlider_Zslice.value()
        self.label_Xslice.setText("X: %d" % x)
        self.label_Yslice.setText("Y: %d" % y)
        self.label_Zslice.setText("Z: %d" % z)
        return x, y, z

    def get_current_networks(self):
        ica_lookup = str(self.listWidget_ICAComponents.currentItem().data(QtCore.Qt.UserRole).toString())
        rsn_lookup = str(self.listWidget_RSN.currentItem().data(QtCore.Qt.UserRole).toString())
        return ica_lookup, rsn_lookup

    def get_plot_options(self, ica_lookup, rsn_lookup):
        display, coords = self.apply_slice_views()
        options = {'ica_lookup': ica_lookup, 'rsn_lookup': rsn_lookup, 'display': display, 'coords': coords}
        if isinstance(self.gd['rsn'][rsn_lookup]['img'], Nifti1Image):
            options.update({'show_rsn': True})
        if self.checkBox_showWM.isChecked() and 'wm_mask' in self.gd.keys():
            options.update({'show_wm': True})
        if self.checkBox_showCSF.isChecked() and 'csf_mask' in self.gd.keys():
            options.update({'show_csf': True})
        if self.checkBox_showGM.isChecked() and 'gm_mask' in self.gd.keys():
            options.update({'show_gm': True})
        if self.checkBox_showBrain.isChecked() and 'brain_mask' in self.gd.keys():
            options.update({'show_brain': True})
        if self.checkBox_showSegmentations.isChecked() and 'segmentations' in self.gd.keys():
            options.update({'show_segmentations': True})
        if self.checkBox_taveplot.isChecked():
            options.update({'show_time_average': True})
        if self.checkBox_tgroupplot.isChecked():
            options.update({'show_time_group': True})
        if self.checkBox_tindplot.isChecked():
            options.update({'show_time_individual': True})
        if self.checkBox_tspectrum.isChecked():
            options.update(({'show_spectrum': True}))
        return options

    def get_filepaths(self, list_name):
        return self.get_guiitem(list_name, 'filepath')

    def get_imgobjects(self, list_name):
        return [img for img in self.get_guiitem(list_name, 'img') if isinstance(img, Nifti1Image)]

    def get_guiitem(self, list_name, prop):
        return [v[prop] for v in self.gd[list_name].itervalues()]


def main():
    app = QtGui.QApplication(sys.argv)  # A new instance of QApplication
    form = MapperGUI()  # We set the form to be our ExampleApp (design)
    form.show()  # Show the form
    app.exec_()  # and execute the app


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function