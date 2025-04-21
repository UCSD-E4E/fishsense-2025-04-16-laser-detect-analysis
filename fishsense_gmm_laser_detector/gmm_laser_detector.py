'''Gausian Mean Mixture Laser Detector
'''
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
from pyfishsensedev.laser.laser_detector import LaserDetector
from sklearn.mixture import BayesianGaussianMixture


class PDFMultiDim():
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

    def pdf(self, X: np.ndarray) -> float:
        return self.pdf_multidim(X, self.mean, self.cov_inv, self.cov_det)

    @staticmethod
    def pdf_multidim(X: np.ndarray,
                     mean: np.ndarray,
                     cov_inv: np.ndarray,
                     cov_det: np.ndarray) -> float:
        X = X.astype(np.float32)
        d = X.shape[1]
        diff = X - mean
        exponent = -0.5 * np.einsum('ij,ij->i', diff @ cov_inv, diff)
        denom = np.sqrt((2.0 * np.pi)**d * cov_det)
        return (1.0 / denom) * np.exp(exponent)

class GMMLaserDetector(LaserDetector):
    def __init__(self):
        self.means: np.typing.ArrayLike
        self.weights: np.typing.ArrayLike
        self.n_components: int
        self.cov_invs: List[np.typing.ArrayLike]
        self.cov_dets: List[np.typing.ArrayLike]

        super().__init__()

    def _load_from_gmm(self, gmm: BayesianGaussianMixture):
        self.n_components = len(gmm.means_)
        self.means = gmm.means_
        self.weights = gmm.weights_
        self.cov_dets = [np.linalg.det(cov) for cov in gmm.covariances_]
        self.cov_invs = [np.linalg.inv(cov) for cov in gmm.covariances_]



    def _pdf(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        p = np.zeros(N, dtype=np.float64)
        for idx in range(self.n_components):
            p_comp = PDFMultiDim.pdf_multidim(
                X,
                self.means[idx],
                self.cov_invs[idx],
                self.cov_dets[idx]
            )
            p += self.weights[idx] * p_comp
        return p


    @classmethod
    def initialize_from_pickled_gmm(cls, path_to_model: Path) -> GMMLaserDetector:
        with open(path_to_model, 'rb') as handle:
            gmm = pickle.load(handle)
        retval = GMMLaserDetector()
        retval._load_from_gmm(gmm)
        return retval
    
    @classmethod
    def initialize_from_json(cls, path_to_model: Path) -> GMMLaserDetector:
        with open(path_to_model, 'r', encoding='utf-8') as handle:
            params = json.load(handle)
        retval = GMMLaserDetector()
        retval.n_components = len(params['means'])
        retval.means = np.array(params['means'])
        retval.weights = np.array(params['weights'])
        covariances = np.array(params['covariances'])
        retval.cov_dets = [np.linalg.det(cov) for cov in covariances]
        retval.cov_invs = [np.linalg.inv(cov) for cov in covariances]
        return retval

    def _find_mean_cov(self, img: cv.Mat) -> Tuple[np.ndarray, np.ndarray]:
        img32 = img.astype(np.float32)
        pixels = img32.reshape(-1, 3)
        mean_hsv = np.mean(pixels, axis=0)
        cov_hsv = np.cov(pixels, rowvar=False)
        return mean_hsv, cov_hsv

    def find_laser(self,
                   img_bgr: cv.Mat,
                   *,
                   prior_laser: float = 0.00001,
                   prior_bg: float = 0.99999,
                   threshold: float = 0.5
                   ) -> Optional[Tuple[float, float]]:

        img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

        # Flatten (H*W, 3)
        height, width, _ = img_hsv.shape
        pixels_hsv = img_hsv.reshape(-1, 3).astype(np.float32)

        # Build background model
        bg_mean, bg_cov = self._find_mean_cov(img_hsv)
        pdf_bg = PDFMultiDim(bg_mean, bg_cov)

        # Compute likelihoods
        likelihood_bg = pdf_bg.pdf(pixels_hsv)
        likelihood_laser = self._pdf(pixels_hsv)

        # Combine with priors => posterior
        numerator_laser = likelihood_laser * prior_laser
        numerator_bg = likelihood_bg * prior_bg
        norm_const = numerator_laser + numerator_bg

        # Posterior for laser
        posterior_laser = numerator_laser / norm_const

        # Classify as laser if posterior_laser > posterior_bg => posterior_laser > 0.5
        laser_mask = (posterior_laser > threshold).reshape(height, width)

        # Get centroid
        laser_rows, laser_cols = np.where(laser_mask)
        if len(laser_rows) > 0:
            return np.mean(laser_cols), np.mean(laser_rows)
        return None