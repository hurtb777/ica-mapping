import numpy as np
#from sklearn import svm, linear_model as lin, metrics #--kw-- 11/30/2017: unused fns.
from nilearn import image
from nibabel.nifti1 import Nifti1Image, Nifti1Pair #--kw-- 10/23/2017: added class for .img/.hdr pair data format
from nibabel.affines import apply_affine #--kw-- 11/17/2017: fn. to convert coordinates in voxel space to MNI space
#import nipype.interfaces.spm.utils as spm #--kw-- 11/30/2017: spm fns. unused

class Mapper(object):
    """
    Rank each file in `in_files` against the `map_files` templates. Ranking is performed by computing the Matthews
    Correlation Coeficient (Phi-correlation coefficient), which is useful for binary vector correlations. Similar in
    interpretation to the Pearson Coefficient, this value ranges from -1 to 1, were -1 implies a anti-correlation,
    0 implies no correlation at at, and 1 is a perfect positive correlation.
    """
    def __init__(self, map_files, map_filenames=None, in_files=None, in_filenames=None): #--kw-- removed unused threshold var., added input for filenames for use as keys
#        self.in_files, self.map_files, self.threshold = in_files, map_files, threshold  #--kw-- removed unused threshold var., added input for filenames for use as keys
        self.in_files, self.map_files = in_files, map_files
        self.in_filenames, self.map_filenames = in_filenames, map_filenames
        self.corr = {}
        self.matches = {}
        self.matches_top3 = {}
        self.matches_max_coords = {}
        self._load_files()

    def _load_files(self):
        self.corr = {}
        self.in_imgs = [i if isinstance(i, (Nifti1Image, Nifti1Pair)) else image.load_img(i) for i in self.in_files]#--kw-- 11/9/2017: added class for .img/.hdr pair data format
        self.map_imgs = [i if isinstance(i, (Nifti1Image, Nifti1Pair)) else image.load_img(i) for i in self.map_files]#--kw-- 11/9/2017: added class for .img/.hdr pair data format

    def run(self):
        """Generate correlations, find top matches, choose best fit"""
        self.corr = Mapper.spatial_correlations(self.in_imgs, self.map_imgs, self.in_filenames, self.map_filenames) #--kw-- 11/13/2017: see changes in spatial_correlations fn.
        self.assign_matches() #--kw-- 11/10/2017: added, assigns obvious matches
        self.find_top3()      #--kw-- 11/15/2017: added, finds top 3 matches for specialized display in main GUI
        self.matches_max_coords = Mapper.find_max_coords(self.in_imgs, self.map_imgs, self.in_filenames, self.map_filenames, self.matches_top3) #--kw-- 11/17/2017: added, finds (x,y,z) MNI coords for max. overlap 


    def run_one(self, in_file, label=None):
        label = label if label else in_file
        corr = Mapper.spatial_correlations(in_file, self.map_imgs)
        self.corr.update({label: corr[0]})
        return corr

    def get_top_matches(self, in_file, num_matches=None): #--kw-- 11/14/2017: removed unused mininum_corr input, changes in indexing
        corr = self.corr[in_file].iteritems() if in_file in self.corr.keys() else self.spatial_correlations(image.load_img(in_file)) #--kw-- 11/14/2017: corr var. now indexed by ica name
        corr = [(x,y) for x,y in corr] #--kw-- 11/14/2017: convert to list of tuples for sorting
        corr.sort(key=(lambda x: x[1]), reverse=True) #--kw-- 11/14/2017: sort corrs. by value, keeping associated rsn names
        if isinstance(num_matches, int):#--kw-- 11/10/2017: debugging, to match type of var.
            corr = corr[0:num_matches]
        return corr  #--kw-- return ordered tuple, rather than dict

    def assign_matches(self, minimum_correlation=0.3, null_network=None, allow_copies=True): #--kw-- 11/10/2017: changed min. corr. default & default match to improve output
        """Return best match based on correlation."""
        for in_file in self.in_filenames:
            top_corrs = self.get_top_matches(in_file, num_matches=2)#--kw-- 11/10/2017: get top 2 matches for comparison
            if top_corrs[0][1] >= minimum_correlation and top_corrs[0][1] >= 2*top_corrs[1][1]: #--kw-- 11/10/2017: modified criteria for automated matching
                self.matches[in_file] = top_corrs[0][0] #--kw-- 11/13/2017: add ica comp. & rsn template names to matches
            else:
                self.matches[in_file] = null_network #--kw-- 11/13/2017: changed to use ica filenames as keys.  Input to Mapper.in_files typically refs. to Nifti data
        return self.matches
    
    def find_top3(self): #--kw-- 11/15/2017: new fn., output used to set rsn text display in main GUI
        """Finds top 3 template matches for ICA network"""
        top3_corrs = {}
        for in_file in self.in_filenames:
            top3_corrs[in_file] = dict(self.get_top_matches(in_file, num_matches=3))
        self.matches_top3 = top3_corrs
        return(top3_corrs)

    @staticmethod
    def prep_tmap(img, reference=None, center=False, scale=False, threshold=0.2, binary=True): #--kw-- 10/24/2017: changed threshold to remove holes created by resampling w/n clusters, added z-scores for compatibility w/ prev. template matching script
        if isinstance(img, (Nifti1Image, Nifti1Pair)):#--kw-- 11/9/2017: added class for .img/.hdr pair data format
            img = img
        elif img is None and isinstance(reference, (Nifti1Image, Nifti1Pair)):   #--kw-- 11/27/2017: w/u for rsn's not associated with spatial maps, fn. returns reference vol. only.  Max overlap will default to max value in vol.
            img = reference
        else:
            image.load_img(img)
        if isinstance(reference, (str, (Nifti1Image, Nifti1Pair))):#--kw-- 11/9/2017: added class for .img/.hdr pair data format
            dat = image.resample_to_img(source_img=img, target_img=reference).get_data().flatten()
        else:
            dat = img.get_data().flatten()
        if center:     #--kw-- 10/25/2017: NOTE: means not subtracted for consistency w/ previous template matching results
            dat[dat.nonzero()] = dat[dat.nonzero()] - dat[dat.nonzero()].mean()
        if scale: #--kw-- 10/24/2017: scale to standard deviation option, for consistency w/ previous template matching results
            dat[dat.nonzero()] = dat[dat.nonzero()] / dat[dat.nonzero()].std(ddof=1)
        if threshold:   #--kw-- 10/23/2017: thresholding images optional, for consistency w/ previous template matching results
            if binary:
                dat[dat >= threshold] = 1.  #--kw-- 10/24/2017: neg. ICA coefficients treated as noise for ICA template matching
                dat[dat < threshold] = 0.
            else:
                dat[dat < threshold] = 0.      
        return dat

    @staticmethod
    def spatial_correlations(imgs, map_imgs, img_names, map_names):#--kw-- 11/13/2017: need to keep ica & rsn reference name associated with each corr., otherwise display order does not match order of corr in list
        """Rank the `imgs` against each `map_files` templates."""
        corr = {}
        imgs = imgs if hasattr(imgs, '__iter__') else [imgs]  # make iterable
        map_imgs = map_imgs if hasattr(map_imgs, '__iter__') else [map_imgs]  # make iterable
        for i, img in enumerate(imgs):
            corr[img_names[i]] = {}#--kw-- 11/13/2017: need to keep ica & rsn reference name associated with each corr., otherwise display order does not match order of corr in list
            img_arr = Mapper.prep_tmap(img, scale=True, threshold=1, binary=False)   #--kw-- 10/23/2017: thresholding images optional, for consistency w/ previous template matching results
            for ii, mimg in enumerate(map_imgs): #--kw-- 11/13/2017: changed indexing to keep rsn reference name associated with corr. value
                corr[img_names[i]][map_names[ii]] = np.corrcoef(Mapper.prep_tmap(mimg, reference=img), img_arr)[0,1]#--kw-- prob. w/ corr. fn., error from mix of binary & continuous vars.?
        return corr
    
    @staticmethod
    def find_max_coords(imgs, map_imgs, img_names, map_names, top3=None): #--kw-- 11/17/2017: finds max. overlap between ica & rsn maps
        """Find coordinates (in MNI space) of max. overlap between 'imgs' and 'map_imgs' """
        max_coords = {}
        imgs = imgs if hasattr(imgs, '__iter__') else [imgs]  # make iterable
        map_imgs = map_imgs if hasattr(map_imgs, '__iter__') else [map_imgs]  # make iterable
        if top3:
            imgs = [img for i,img in enumerate(imgs) if img_names[i] in top3.keys()]
            img_names = [name for name in img_names if name in top3.keys()]
        for i, img in enumerate(imgs):
            max_coords[img_names[i]] = {}
            img_arr = Mapper.prep_tmap(img, scale=True, threshold=1, binary=False)
            if top3:
                map_imgs_select = [mimg for ii,mimg in enumerate(map_imgs) if map_names[ii] in top3[img_names[i]].keys()]
                map_names_select = [name for name in map_names if name in top3[img_names[i]].keys()]
            else:
                map_imgs_select, map_names_select = map_imgs, map_names
            for ii, mimg in enumerate(map_imgs_select):
                map_arr = Mapper.prep_tmap(mimg, reference=img)
                max_ind = np.array(img_arr * map_arr).argmax()
                
                voxel_coords = np.unravel_index(max_ind, img.shape)
                max_coords[img_names[i]][map_names_select[ii]] = tuple(apply_affine(img.affine, voxel_coords))
        return max_coords
            
