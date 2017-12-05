"""
ica_mapping_gui.py
"""

# Python Libraries and QT
from os.path import join as opj  # method to join strings of file paths
import os, sys, re, getopt, json
from functools import partial
#from PyQt4 import QtGui, QtCore, Qt  # Import QT
from PyQt4 import QtGui, QtCore  # Import QT  #--kw-- 11/30/2017: Qt appears to be unused
mypath = os.getcwd()
sys.path.append(opj(mypath, '..'))  #--kw-- 10/16/2017: changed for compatibility w/ POSIX
sys.path.append(opj(mypath))   #--kw-- 10/16/2017: sys.path requires both root dir. ica-mapping-master & subdir. ica-mapping-master/gui

# Mathematical/Neuroimaging/Plotting Libraries
import numpy as np  # Library to for all mathematical operations
from nilearn import plotting, image, input_data  # library for neuroimaging
from nibabel.nifti1 import Nifti1Image, Nifti1Pair #--kw-- 11/9/2017: added class for .img/.hdr pair data format
import nipype.interfaces.io as nio
import matplotlib.pyplot as plt  # Plotting library
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# Internal imports
from settings import mri_plots as mp, time_plots as tp  # use items in settings file
import design  # This file holds our MainWindow and all design related things
from reports import create_html
import mapper as map

ANATOMICAL_TO_TIMESERIES_PLOT_RATIO = 5
CONFIGURATION_FILE = 'config.json' #--kw-- 12/4/2017: needs to be changed to '../config.json' to run program in spyder, change to 'config.json' for command line

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
        self.pushButton_icaload.clicked.connect(self.browse_ica_files) #--kw-- 11/29/2017: need new fn. for w/ GIFT v4.0, ica components saved as 4D nifti files, etc.
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
        self.pushButton_Plot.clicked.connect(self.plot_max_overlap) #--kw-- 11/17/2017: default to plot max. overlap between ica & rsn for new plots

        self.listWidget_ICAComponents.itemClicked.connect(self.update_gui)
        self.listWidget_RSN.itemClicked.connect(self.update_gui)
        self.listWidget_mappedICANetworks.itemClicked.connect(self.update_mapping)
#        self.listWidget_mappedICANetworks.currentItemChanged.connect(self.update_mapping) #--kw-- 11/16/2017: skip update w/ change, only update w/ click on mappedICANetworks list

        self.pushButton_addNetwork.clicked.connect(self.add_mapped_network)
        self.pushButton_rmNetwork.clicked.connect(self.delete_mapped_network)
        self.pushButton_reset.clicked.connect(self.reset_gui)
        self.pushButton_runAnalysis.clicked.connect(self.run_analysis)
        self.pushButton_createReport.clicked.connect(self.generate_report)
        
        self.horizontalSlider_Xslice.sliderReleased.connect(self.update_plots)
        self.horizontalSlider_Yslice.sliderReleased.connect(self.update_plots)
        self.horizontalSlider_Zslice.sliderReleased.connect(self.update_plots)
        self.spinBox_numSlices.valueChanged.connect(self.update_plots)
        
        self.pushButton_zslice.clicked.connect(self.plot_max_overlap) #--kw-- 11/17/2017: plot max. overlap between ica & rsn 

        self.listWidget_ICAComponents.setCurrentRow(0)
        self.listWidget_RSN.setCurrentRow(0)
#        self.mapper = map.Mapper(map_files=self.get_imgobjects('rsn'), map_filenames=self.get_imgobjNames('rsn'),
#                                 in_files=self.get_imgobjects('ica'), in_filenames=self.get_imgobjNames('ica'))#--kw-- 11/29/2017: need to reset Mapper if values have changed, #--kw-- 11/10/2017: added filenames input, for use in creating dictionary of correlations 
        
        self.plot_max_overlaps = False #--kw-- 11/20/2017: after 1st call to plotting fn., update display of plots w/ GUI if true

    def update_gui(self, update_lineEdits=True): #--kw-- 11/16/2017: option to skip lineEdit updates used by update_mapping fn.
        ica_name = str(self.listWidget_ICAComponents.currentItem().data(QtCore.Qt.UserRole).toString()) #--kw-- 11/16/2017: use original item text, not current text w/ prefix below for corr., etc.
        rsn_name = str(self.listWidget_RSN.currentItem().data(QtCore.Qt.UserRole).toString())
        if ica_name in self.mapper.corr.keys():
            for i in range(self.listWidget_RSN.count()):
                item = self.listWidget_RSN.item(i)
                lookup = str(item.data(QtCore.Qt.UserRole).toString())

                if lookup in self.config['rsn']['extra_items']: #--kw-- 11/16/2017: skip extra items on rsn list w/o spatial maps
                    item.setTextColor(QtGui.QColor(0,0,0,100))  #--kw-- 11/15/2017: highlight top 3 matches w/ text transparency
                elif lookup in self.gd['rsn_duplicates'].keys(): #--kw-- 11/14/2017: mark RSNs w/ duplicate matches with an asteriks
                    item.setText("%s (%0.2f)***" %(lookup, self.mapper.corr[ica_name][lookup])) #--kw--11/14/2017: added indexing for self.mapper.corr by lookup
                else:
                    item.setText("%s (%0.2f)" %(lookup, self.mapper.corr[ica_name][lookup])) #--kw--11/14/2017: added indexing for self.mapper.corr by lookup
                if lookup not in self.mapper.matches_top3[ica_name].keys(): #--kw-- 11/15/2017: highlight top 3 matches w/ text transparency
                    item.setTextColor(QtGui.QColor(0,0,0,100))
                else:
                    item.setTextColor(QtGui.QColor(0,0,0,255))
        else:
            for i in range(self.listWidget_RSN.count()):
                item = self.listWidget_RSN.item(i)
                lookup = str(item.data(QtCore.Qt.UserRole).toString())
                item.setText(lookup)

        if self.computed_analysis:
#            pass
            self.listWidget_mappedICANetworks.sortItems()
            for i in range(self.listWidget_ICAComponents.count()): #--kw-- mark ICA comps requiring more analysis
                item = self.listWidget_ICAComponents.item(i)
                lookup = str(item.data(QtCore.Qt.UserRole).toString())
                if lookup in self.gd['mapped'].keys() and lookup not in self.gd['ica_to_RSNduplicates'].keys():
                    item.setTextColor(QtGui.QColor(0,0,0,70)) #--kw-- display partially transparent text if ica is mapped onto unique rsn_custom_name
                else:
                    item.setTextColor(QtGui.QColor(0,0,0,255))
            for i in range(self.listWidget_mappedICANetworks.count()): #--kw-- similarly, highlight mappings requiring more analysis w/ transparency
                item = self.listWidget_mappedICANetworks.item(i)
                ica_lookup, rsn_lookup = item.text().split(' > ')
                if ica_lookup in self.gd['mapped'].keys() and ica_lookup not in self.gd['ica_to_RSNduplicates'].keys():
                    item.setTextColor(QtGui.QColor(0,0,0,70)) #--kw-- display partially transparent text if ica is mapped onto unique rsn_custom_name
                else:
                    item.setTextColor(QtGui.QColor(0,0,0,255))
            if ica_name in self.gd['mapped'].keys():
                if self.gd['mapped'][ica_name]['rsn_lookup']==rsn_name:
                    self.listWidget_mappedICANetworks.setCurrentItem(self.gd['mapped'][ica_name]['mapped_item'])
                else:
                    self.listWidget_mappedICANetworks.setCurrentRow(-1)
            else:
                self.listWidget_mappedICANetworks.setCurrentRow(-1)
            self.plot_max_overlap() #--kw-- automatically update display of plots when updating networks
                
        if update_lineEdits: #--kw-- 11/16/2017: added option to skip lineEdit updates used by update_mapping fn.
           self.lineEdit_ICANetwork.setText(ica_name)
           self.lineEdit_mappedICANetwork.setText(rsn_name)

    def update_mapping(self, mapped_item=None): #--kw-- 11/16/2017: mapped_item input added to allow change in focus of ica & rsn list of items in response to clicking on map item
#        ica_lookup, rsn_lookup = self.get_current_networks() #--kw-- 11/16/2017: skip if updating in response to map item list
        if mapped_item:
            ica_lookup, rsn_lookup = str(mapped_item.text()).split(' > ')
        else:
            mapped_item = str(self.listWidget_mappedICANetworks.currentItem().text())
            ica_lookup, rsn_lookup = mapped_item.split(' > ')

        self.listWidget_ICAComponents.setCurrentItem(self.gd['mapped'][ica_lookup]['ica_item'])
        self.listWidget_RSN.setCurrentItem(self.gd['mapped'][ica_lookup]['rsn_item'])
        
        self.lineEdit_ICANetwork.setText(self.gd['mapped'][ica_lookup]['ica_custom_name'])
        self.lineEdit_mappedICANetwork.setText(self.gd['mapped'][ica_lookup]['rsn_custom_name'])
        self.update_gui(update_lineEdits=False) #--kw-- update corrs. for rsn text after shifting focus, skip lineEdit updates in favor of above
        
    def reset_gui(self):
        self.listWidget_mappedICANetworks.clear()
        self.gd = {'fmri': {}, 'smri': {},'ica': {}, 'rsn': {}, 'mapped': {}, 'rsn_duplicates': {}, 'ica_to_RSNduplicates': {}} #--kw-- 11/16/2017: added dict for rsn's w/ duplicate matches

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
        self.update_duplicate_mappings() #--kw-- 11/16/2017: new fn. used to mark duplicates for rsn display
        self.update_mapping(map_itemWidget) #--kw-- modified fn. updates foci of lists using input
        ica_ind = self.listWidget_ICAComponents.currentRow() #--kw-- 11/30/2017: added code to move to next unmapped network
        ind_max = self.listWidget_ICAComponents.count() - 1
        mapped_keys = [k for k in self.gd['mapped'].keys() if k not in self.gd['ica_to_RSNduplicates'].keys()]
        while (ica_ind <= ind_max):
            if str(self.listWidget_ICAComponents.item(ica_ind).data(QtCore.Qt.UserRole).toString()) in mapped_keys:
                ica_ind += 1
            else:
                break
        if ica_ind > ind_max:
            ica_ind = ica_ind % ind_max - 1 #--kw-- 11/30/2017: loop through ica indices from beginning
            while (ica_ind <= ind_max) and (str(self.listWidget_ICAComponents.item(ica_ind).data(QtCore.Qt.UserRole).toString()) in mapped_keys):
                ica_ind += 1
        if (ica_ind <= ind_max) and (str(self.listWidget_ICAComponents.item(ica_ind).data(QtCore.Qt.UserRole).toString()) not in mapped_keys):
            self.listWidget_ICAComponents.setCurrentRow(ica_ind)
            ica_lookup = str(self.listWidget_ICAComponents.item(ica_ind).data(QtCore.Qt.UserRole).toString())
            rsn_lookup = [k for k in self.mapper.matches_top3[ica_lookup].keys() if self.mapper.matches_top3[ica_lookup][k]==max(self.mapper.matches_top3[ica_lookup].values())][0] #--kw-- 11/30/2017: find name of top rsn match & set rsn list appropriately
            self.listWidget_RSN.setCurrentItem(self.gd['rsn'][rsn_lookup]['widget'])
        else:
            self.listWidget_ICAComponents.setCurrentRow(0)
        self.update_gui()#--kw-- 11/16/2017: changing mappings may change display
        
    def delete_mapped_network(self):
        for item in self.listWidget_mappedICANetworks.selectedItems():
            ica_name =self.listWidget_ICAComponents.currentItem().text()
            self.listWidget_mappedICANetworks.takeItem(self.listWidget_mappedICANetworks.row(item))
            del self.gd['mapped'][ica_name]
            
    def update_duplicate_mappings(self): #--kw-- 11/16/2017: new fn., counts duplicated rsn matches
        rsn_matched = []
        ica_to_RSNduplicates = {}
        for ica in self.gd['mapped'].keys():
            rsn_matched.append(self.gd['mapped'][ica]['rsn_custom_name'])
        ex = re.compile('Noise*')
        rsn_noise = filter(ex.search, rsn_matched)
        ex = re.compile('Other*')
        rsn_other = filter(ex.search, rsn_matched)
        rsn_matched = [k for k in rsn_matched if k not in rsn_noise not in rsn_other]
        rsn_duplicated = {rsn: rsn_matched.count(rsn) for rsn in rsn_matched if rsn_matched.count(rsn) > 1}
        for ica in self.gd['mapped'].keys():
            rsn = self.gd['mapped'][ica]['rsn_custom_name']
            if rsn in rsn_duplicated.keys():
                ica_to_RSNduplicates[ica] = rsn
        self.gd['rsn_duplicates'] = rsn_duplicated
        self.gd['ica_to_RSNduplicates'] = ica_to_RSNduplicates
            
    def browse_file(self, file_type, pushButton=None):
        fname = QtGui.QFileDialog.getOpenFileName(self, caption="Open Image", directory=".",
                                                  filter="Image Files (*.nii.gz *.nii *.img)")#--kw-- 10/17/2017: changed to load images saved in paired .img/.hdr format
        self.load_base_file(fname, file_type=file_type)

    def load_base_file(self, file_name, file_type='fmri'):
        if file_name:
            self.gd.update({file_type: {'full_path': file_name, 'img': image.load_img(str(file_name))}})

    def browse_folder(self, listWidget=None, search_pattern='\w+', title="Select Directory", list_name='ica'):
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, title)) #--kw-- 12/4/2017: added output as string
        if search_pattern is not None:
            self.find_files(directory, "*", search_pattern, listWidget, list_name)
        return directory
    
    def browse_ica_files(self): #--kw-- 11/29/2017: new fn., includes functionality to load 4D nifti files
        listWidget=self.listWidget_ICAComponents
        
        search_pattern=self.config['ica']['search_pattern']
        title='Select all ICA Component Files'
        list_name='ica'
        selected_files = QtGui.QFileDialog.getOpenFileNames(self, caption=title, directory=".", filter="Image Files (*.nii.gz *.nii *.img)")
        selected_files = [str(f) for f in selected_files if isinstance(selected_files, QtCore.QStringList)]  
        
        listWidget.clear() # In case there are any existing elements in the list
        self.gd['ica'].clear()
        
        r = re.compile(search_pattern)
        filtered_files = filter(r.search, selected_files)

        if len(filtered_files) is 1:
            t_dim = image.load_img(filtered_files).shape[3]
            if t_dim > 1: #if 4D nifti vol...
                for t in range(t_dim):
                    match = re.search(r, filtered_files[0])
                    lookup_key = match.groups()[0]
                    lookup_key = lookup_key + ',%d' %(t+1) #--kw-- 11/30/2017: add suffix to display name
                    item = QtGui.QListWidgetItem(lookup_key)
                    listWidget.addItem(item)
                    item.setData(QtCore.Qt.UserRole, lookup_key)
                    self.gd[list_name][lookup_key] = {'img': image.index_img(filtered_files[0], t),
                                                      'filepath': filtered_files,
                                                      'name': lookup_key,
                                                      'widget': item}
            else:
                file_name = filtered_files[0]
                match = re.search(r, file_name)
                lookup_key = match.groups()[0]
                item = QtGui.QListWidgetItem(lookup_key)
                listWidget.addItem(item)
                item.setData(QtCore.Qt.UserRole, lookup_key)
                self.gd[list_name][lookup_key] = {'img': image.load_img(filtered_files[0]),
                                                  'filepath': filtered_files[0],
                                                  'name': lookup_key,
                                                  'widget': item}
        else:
            for file_name in filtered_files:
                match = re.search(r, file_name)
                lookup_key = match.groups()[0]
                item = QtGui.QListWidgetItem(lookup_key)
                listWidget.addItem(item)
                item.setData(QtCore.Qt.UserRole, lookup_key)
                self.gd[list_name][lookup_key] = {'img': image.load_img(file_name),
                                                  'filepath': file_name,
                                                  'name': lookup_key,
                                                  'widget': item}
                


    def browse_output_directory(self):
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Output Directory")) #--kw-- 12/4/2017: if needed, convert output to string
        if directory:
            self.lineEdit_outputDir.setText(directory)
        return directory #--kw-- 12/4/2017: added return for fn.

    def find_files(self, directory, template, search_pattern, listWidget, list_name, extra_items=None):
        if directory: # if user didn't pick a directory don't continue
            ds = nio.DataGrabber(base_directory=directory, template=template, sort_filelist=True)
            all_nifti_files = ds.run().outputs.outfiles
            listWidget.clear() # In case there are any existing elements in the list
            r = re.compile(search_pattern)     
#            list_items = [] #--kw-- 11/30/2017: unused variable
            filtered_files = filter(r.search, all_nifti_files) #--kw-- 11/30/2017: added functionality to handle 4d Nifti files
            if len(filtered_files) is 1:
                t_dim = image.load_img(filtered_files).shape[3]
                if t_dim > 1: #if 4D nifti vol...
                    for t in range(t_dim):
                        match = re.search(r, filtered_files[0])
                        lookup_key = match.groups()[0]
                        lookup_key = lookup_key + ',%d' %(t+1) #--kw-- 11/30/2017: add suffix to display name
                        item = QtGui.QListWidgetItem(lookup_key)
                        listWidget.addItem(item)
                        item.setData(QtCore.Qt.UserRole, lookup_key)
                        self.gd[list_name][lookup_key] = {'img': image.index_img(filtered_files[0], t),
                                                          'filepath': filtered_files,
                                                          'name': lookup_key,
                                                          'widget': item}
                else:
                    file_name = filtered_files[0]
                    match = re.search(r, file_name)
                    lookup_key = match.groups()[0]
                    item = QtGui.QListWidgetItem(lookup_key)
                    listWidget.addItem(item)
                    item.setData(QtCore.Qt.UserRole, lookup_key)
                    self.gd[list_name][lookup_key] = {'img': image.load_img(filtered_files[0]),
                                                      'filepath': filtered_files[0],
                                                      'name': lookup_key,
                                                      'widget': item}
            else:
                for file_name in filtered_files:
                    match = re.search(r, file_name)
                    lookup_key = match.groups()[0]
                    item = QtGui.QListWidgetItem(lookup_key)
                    listWidget.addItem(item)
                    item.setData(QtCore.Qt.UserRole, lookup_key)
                    self.gd[list_name][lookup_key] = {'img': image.load_img(file_name),
                                                      'filepath': file_name,
                                                      'name': lookup_key,
                                                      'widget': item}
#            for file_name in filter(r.search, all_nifti_files): #--kw-- 11/30/2017: added functionality to handle 4d Nifti files above 
#                match = re.search(r, file_name)
#                lookup_key = match.groups()[0]
#                item = QtGui.QListWidgetItem(lookup_key)
#                listWidget.addItem(item)
#                item.setData(QtCore.Qt.UserRole, lookup_key)
#                self.gd[list_name][lookup_key] = {'img': image.load_img(opj(directory, file_name)),
#                                                  'filepath': opj(directory, file_name),
#                                                  'name': lookup_key,
#                                                  'widget': item}
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
        self.pushButton_runAnalysis.setText("Creating...") #--kw-- 11/10/2017: tiny & possibly irrelevant asthetic tweak
        if self.listWidget_ICAComponents.count() == 0: self.browse_ica_files()
        if self.listWidget_RSN.count() == 0:
            self.browse_folder(listWidget=self.listWidget_RSN,
                               search_pattern=self.config['rsn']['search_pattern'],
                               title="Select RSN Directory",
                               list_name="rsn")
        self.mapper = map.Mapper(map_files=self.get_imgobjects('rsn'), map_filenames=self.get_imgobjNames('rsn'),
                                 in_files=self.get_imgobjects('ica'), in_filenames=self.get_imgobjNames('ica')) #--kw-- 11/29/2017: need to reset Mapper if values have changed

        self.mapper.run()  #--kw-- 11/9/2017: changed to automatically run entire analysis
#        lookup_val = str(self.listWidget_ICAComponents.currentItem().data(QtCore.Qt.UserRole).toString())
#        corr = self.mapper.run_one(self.gd['ica'][lookup_val]['img'], label=lookup_val) \
#            if lookup_val not in self.mapper.corr.keys() else self.mapper.corr[lookup_val]
        for ica_lookup, rsn_lookup in self.mapper.matches.iteritems(): #--kw-- 11/13/2017: added.  Loop over current matches & add results to mapping list
            if ica_lookup not in self.gd['mapped'].keys() and rsn_lookup is not None:
                name = "%s > %s" %(ica_lookup, rsn_lookup)
                map_itemWidget = QtGui.QListWidgetItem(name)
                self.listWidget_mappedICANetworks.addItem(map_itemWidget)  
                ica_custom_name = ica_lookup
                rsn_custom_name = rsn_lookup
                ica_item = self.listWidget_ICAComponents.findItems(ica_lookup, QtCore.Qt.MatchContains)[0]
                rsn_item = self.listWidget_RSN.findItems(rsn_lookup, QtCore.Qt.MatchContains)[0]
                self.gd['mapped'].update(
                   {ica_lookup:  # 1 Mapped Network per ICA 
                      {'rsn_lookup': rsn_lookup, #  RSN name 
                       'custom_name': name,  # Composite user's custom Name
                       'ica_custom_name': ica_custom_name,  # User's Custom Name
                       'rsn_custom_name': rsn_custom_name,  # User's Custom Name
                       'ica_item': ica_item,  # Index of the ICA listWidget 
                       'rsn_item' : rsn_item,  # Index of the RSN listWidget
                       'mapped_item' : map_itemWidget,  # QListWidgetItem in the mapped listWidget
                       }
                    }
                )
        self.update_duplicate_mappings() #--kw-- 11/16/2017: new fn., for rsn display
        self.computed_analysis = True #--kw-- 11/13/2017: moved line from below, computed_analysis info used for display in update_gui
        if self.listWidget_ICAComponents.currentItem() is None: self.listWidget_ICAComponents.setCurrentRow(0)  #--kw-- 11/30/2017: call to self.update_gui() requires current items specified for both lists
        if self.listWidget_RSN.currentItem() is None: self.listWidget_RSN.setCurrentRow(0)  #--kw-- 11/30/2017: call to self.update_gui() requires current items specified for both lists
        ica_ind = self.listWidget_ICAComponents.currentRow() #--kw-- 11/30/2017: added code to move to next unmapped network
        ind_max = self.listWidget_ICAComponents.count() - 1
        mapped_keys = [k for k in self.gd['mapped'].keys() if k not in self.gd['ica_to_RSNduplicates'].keys()]
        while (ica_ind <= ind_max):
            if str(self.listWidget_ICAComponents.item(ica_ind).data(QtCore.Qt.UserRole).toString()) in mapped_keys:
                ica_ind += 1
            else:
                break
        if ica_ind > ind_max:
            ica_ind = ica_ind % ind_max - 1 #--kw-- 11/30/2017: loop through ica indices from beginning
            while (ica_ind <= ind_max) and (str(self.listWidget_ICAComponents.item(ica_ind).data(QtCore.Qt.UserRole).toString()) in mapped_keys):
                ica_ind += 1
        if (ica_ind <= ind_max) and (str(self.listWidget_ICAComponents.item(ica_ind).data(QtCore.Qt.UserRole).toString()) not in mapped_keys):
            self.listWidget_ICAComponents.setCurrentRow(ica_ind)
            ica_lookup = str(self.listWidget_ICAComponents.item(ica_ind).data(QtCore.Qt.UserRole).toString())
            rsn_lookup = [k for k in self.mapper.matches_top3[ica_lookup].keys() if self.mapper.matches_top3[ica_lookup][k]==max(self.mapper.matches_top3[ica_lookup].values())][0] #--kw-- 11/30/2017: find name of top rsn match & set rsn list appropriately
            self.listWidget_RSN.setCurrentItem(self.gd['rsn'][rsn_lookup]['widget'])
        else:
            self.listWidget_ICAComponents.setCurrentRow(0)
        self.update_gui()
#        self.computed_analysis = True#--kw-- 11/13/2017: moved to above
        self.pushButton_runAnalysis.setText(btn_txt)

    def generate_report(self):
        btn_txt = self.pushButton_createReport.text()
        self.pushButton_createReport.setText("Creating")
        fx = plt.figure(figsize=(10,4))
        ft = plt.figure(figsize=(10, 2))
        if str(self.lineEdit_outputDir.text()) is '': #--kw-- 11/30/2017: ensure output dir. is specified
            directory = self.browse_output_directory()
        else:
            directory = str(self.lineEdit_outputDir.text())
        if not os.path.exists(directory):
            os.makedirs(directory)
        for u, v in self.gd['mapped'].iteritems():
            if self.plot_max_overlaps:   #--kw-- 11/27/2017: plot max. overlap in report
                self.plot_max_overlap(ica_lookup=u, rsn_lookup=v['rsn_lookup']) 
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
#        if show_rsn:
        if show_rsn and self.gd['rsn'][rsn_lookup]['img'] is not None: #--kw-- 11/27/2017: only plot rsns w/ associated spatial maps, skip for Noise_artifact & Other_nonnoise_rsn
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
    
    def plot_max_overlap(self, ica_lookup=None, rsn_lookup=None): #--kw-- 11/17/2017: fn. for 'Show Largest Overlap' pushbutton
        if self.computed_analysis:
            if ica_lookup is None or rsn_lookup is None:
                ica_lookup, rsn_lookup = self.get_current_networks()
            if rsn_lookup in self.mapper.matches_max_coords[ica_lookup]:
                x,y,z = self.mapper.matches_max_coords[ica_lookup][rsn_lookup]
            else:
                ica_img = [self.gd['ica'][ica_lookup]['img']]
                map_img = [self.gd['rsn'][rsn_lookup]['img']]
                ica_name = [ica_lookup]
                rsn_name = [rsn_lookup]
                self.mapper.matches_max_coords[ica_lookup].update(self.mapper.find_max_coords(ica_img, map_img, ica_name, rsn_name)[ica_lookup])
                x,y,z = self.mapper.matches_max_coords[ica_lookup][rsn_lookup]
                
            self.horizontalSlider_Xslice.setValue(x)
            self.horizontalSlider_Yslice.setValue(y)
            self.horizontalSlider_Zslice.setValue(z)
            self.update_plots()
        else:
            pass
        self.plot_max_overlaps = True #--kw-- plot pos. of max overlap during subsequent calls to update_gui fn.
        

    def get_current_networks(self):
        ica_lookup = str(self.listWidget_ICAComponents.currentItem().data(QtCore.Qt.UserRole).toString())
        rsn_lookup = str(self.listWidget_RSN.currentItem().data(QtCore.Qt.UserRole).toString())
        return ica_lookup, rsn_lookup

    def get_plot_options(self, ica_lookup, rsn_lookup):
        display, coords = self.apply_slice_views()
        options = {'ica_lookup': ica_lookup, 'rsn_lookup': rsn_lookup, 'display': display, 'coords': coords}
        if isinstance(self.gd['rsn'][rsn_lookup]['img'], (Nifti1Image, Nifti1Pair)): #--kw-- 11/9/2017: added class for .img/.hdr pair data format
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
        return [img for img in self.get_guiitem(list_name, 'img') if isinstance(img, (Nifti1Image, Nifti1Pair))] #--kw-- 11/9/2017: added class for .img/.hdr pair data format
    
    def get_imgobjNames(self, list_name): #--kw-- 11/13/2017: added fn. to retrieve names associated w/ RSNs, filtering list items w/o templates
        imgs = [True if isinstance(img, (Nifti1Image, Nifti1Pair)) else False for img in self.get_guiitem(list_name, 'img')]
        names_all = [name for name in self.get_guiitem(list_name, 'name')]
        names = []
        for i, name in enumerate(names_all):
            if imgs[i]:
                names.append(name)
        return(names)

    def get_guiitem(self, list_name, prop):
        return [v[prop] for v in self.gd[list_name].itervalues()]


def main(argv):
    config_file = None
    try:
        opts, args = getopt.getopt(argv, "hi:", ["config_file="])
    except getopt.GetoptError:
#        print 'ica_mapping_gui.py -i <config_file> '
        print('ica_mapping_gui.py -i <config_file> ')  #--kw-- 10/16/2017: changed for future compatibility w/ python3
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
#            print 'ica_mapping_gui.py -i <config_file>'
            print('ica_mapping_gui.py -i <config_file>')  #--kw-- 10/16/2017: changed for future compatibility w/ python3
            sys.exit()
        elif opt in ("-i", "--config_file"):
            config_file = arg

    app = QtGui.QApplication(sys.argv)  # A new instance of QApplication
    form = MapperGUI(configuration_file=config_file)  # We set the form to be our ExampleApp (design)
    form.show()  # Show the form
    app.exec_()  # and execute the app


if __name__ == '__main__':  # if we're running file directly and not importing it
    main(sys.argv[1:])  # run the main function