import numpy as np
from sklearn import svm, linear_model as lin, metrics
from nilearn import image
from nibabel.nifti1 import Nifti1Image
import nipype.interfaces.spm.utils as spm


class Mapper(object):
    """
    Rank each file in `in_files` against the `map_files` templates. Ranking is performed by computing the Matthews
    Correlation Coeficient (Phi-correlation coefficient), which is useful for binary vector correlations. Similar in
    interpretation to the Pearson Coefficient, this value ranges from -1 to 1, were -1 implies a anti-correlation,
    0 implies no correlation at at, and 1 is a perfect positive correlation.
    """
    def __init__(self, map_files, in_files=None, threshold=0.5):
        self.in_files, self.map_files, self.threshold = in_files, map_files, threshold
        self.corr = {}
        self.matches = {}
        self._load_files()

    def _load_files(self):
        self.corr = {}
        self.in_imgs = [i if isinstance(i, Nifti1Image) else image.load_img(i) for i in self.in_files]
        self.map_imgs = [i if isinstance(i, Nifti1Image) else image.load_img(i) for i in self.map_files]

    def run(self):
        """Generate correlations, find top matches, choose best fit"""
        self.corr = {self.in_files[i]: Mapper.spatial_correlations(self.in_imgs[i], self) for i in range(len(self.in_files))}

    def run_one(self, in_file, label=None):
        label = label if label else in_file
        corr = Mapper.spatial_correlations(in_file, self.map_imgs)
        self.corr.update({label: corr[0]})
        return corr

    def get_top_matches(self, in_file, num_matches=None, minimum_corr=None):
        corr = self.corr[in_file] if in_file in self.corr.keys() else self.spatial_correlations(image.load_img(in_file))
        ordered = np.argsort(corr)[::-1]  # sort lowest to highest, then reverse ([::-1])
        if not type(num_matches, 'int'):
            ordered = ordered[:num_matches]
        return ordered, corr[ordered]  # sort index, sorted values (high to low)

    def assign_matches(self, minimum_correlation=0.5, null_network='Random', allow_copies=True):
        """Return best match based on correlation."""
        for i, img in enumerate(self.in_imgs):
            top_map_idx = np.argmax(self.corr[self.in_files[i]])
            if self.corr[self.in_files[i]][top_map_idx] >= minimum_correlation:
                self.matches[self.in_files[i]] = self.map_files[i]
            else:
                self.matches[self.in_files[i]] = null_network
        return self.matches

    @staticmethod
    def prep_tmap(img, reference=None, threshold=0.5):
        img = img if isinstance(img, Nifti1Image) else image.load_img(img)
        if isinstance(reference, (str, Nifti1Image)):
            dat = image.resample_to_img(source_img=img, target_img=reference).get_data().flatten()
        else:
            dat = img.get_data().flatten()
        dat[np.abs(dat) >= threshold] = 1.
        dat[np.abs(dat) < threshold] = 0.
        return dat

    @staticmethod
    def spatial_correlations(imgs, map_imgs):
        """Rank the `imgs` against each `map_files` templates."""
        corr = []
        imgs = imgs if hasattr(imgs, '__iter__') else [imgs]  # make iterable
        map_imgs = map_imgs if hasattr(map_imgs, '__iter__') else [map_imgs]  # make iterable
        for img in imgs:
            img_arr = Mapper.prep_tmap(img)
            corr.append([metrics.matthews_corrcoef(Mapper.prep_tmap(mimg, reference=img), img_arr)
                         for i, mimg in enumerate(map_imgs)])
        return corr
