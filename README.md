## ICA Mapping GUI and Utility

_A neuroimaging package allowing for mapping resultant 3-D ICA probability/t-value maps to known networks._

<img src="docs/imgs/gui_sample.png" alt="Drawing" style="width:100%;"/>


### Background

Processing functional MRI data is a statistically-involved process which aims to filter the input fMRI signal into the
"pure" BOLD signal. One common step to analyze fMRI data is to employ Independent Component Analysis (ICA) on the
time-series signals from each voxel in the scan. This analysis is similar to principle component analysis, but attempts
to isolate sets of voxels that are most statistically independent from one another. Each set, known as an ICA component,
can then be mapped to known resting state networks. Methods and the GUI in this package allow for one to perform this
mapping.

### Mapping Specifics
Mapping is performed by using a spatial correlation of thresholded ICA-maps with the known binary values of the known
(resting-state) network.

### Basic Usage
- Open the GUI using the command line (`python gui/ica_mapping_gui.py`) or using the file explorer
- Data & settings will automatically be loaded using the sample data in the `data` directory
- Click on any ICA and any RSN; Then click the `Plot` button. This displays the networks
- Click the `Run Analysis` button to run the correlations between the chosen ICA network with all of the known networks.
The correlation value will appear in parenthesis next to each known network on the list.
- Choose the most appropriately matched networks, edit the name if desired, and click the `Map Networks` button.
 This adds a new item in the list below. Repeat this step for each desired ICA component.
- When finished click the `Create Report` button to generate the html reports, located in the `output` directory.

### GUI Quick Start

#### Load each attribute using the GUI
1. Open the GUI using the command line (`python gui/ica_mapping_gui.py`) or using the file explorer
2. Set the `Output Directory`, fMRI file, structural MRI file, ICA Component Directory, Resting state
network directory.
3. Click `Run Analysis` to run the correlation analysis on the RSN.
4. `Plot` each pair of statistical maps.

#### Load configuration file (.json)
1. Copy/Past the `config.json` file into a location of your choosing.
2. Edit the fields as you would like:

```javascript
{
  "output_directory": "tmp/",
  "ica":{
    "directory": "../data/ica/",
    "template": "*",
    "search_pattern": "(\\w+)(\\.nii\\.gz|\\.nii)$"
  },
  "rsn":{
    "directory": "../data/rsn/Functional_ROIs/",
    "template": "*/*.nii.gz",
    "search_pattern": "(\\w+)\\\\(\\w+)(\\.nii\\.gz|\\.nii)$",
    "extra_items": ["Noise", "Other"]
  },
  "fmri_file": "../data/func/wrsub-01_task-RestingState_run-1_bold_trim_st_mcf_regfilt_filt.nii",
  "smri_file": "../data/anat/wsub-01_FSPGR.nii",
  "masks": {
    "gm": "../data/anat/rwc1sub-01_FSPGR.nii",
    "wm": "../data/anat/rwc2sub-01_FSPGR.nii",
    "csf": "../data/anat/rwc3sub-01_FSPGR.nii",
    "brain": "../data/anat/wsub-01_FSPGR_brain_mask.nii",
    "cortical_segmentation": ""
  }
}
```

3. Open the GUI either by:
  * executing `python gui/ica_mapping_gui.py -i <config_file>`, OR
  * Clicking `Load>From Settings File` and then choosing the file.
