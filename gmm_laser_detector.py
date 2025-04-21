import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import glob
import os
from pathlib import Path
import rawpy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pyfishsensedev.plane_detector.slate_detector import SlateDetector
from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.laser.laser_detector import LaserDetector

def pdf_multidim(X, mean, cov_inv, cov_det):
    X = X.astype(np.float32)
    d = X.shape[1]
    diff = X - mean
    exponent = -0.5 * np.einsum('ij,ij->i', diff @ cov_inv, diff)
    denom = np.sqrt((2.0 * np.pi)**d * cov_det)
    return (1.0 / denom) * np.exp(exponent)

class GMM():
    def __init__(self, k, samples):
        self.gmm = BayesianGaussianMixture(n_components=k, max_iter=200, covariance_type='full')
        self.gmm.fit(samples)

        self.means = self.gmm.means_
        self.covs = self.gmm.covariances_
        self.weights = self.gmm.weights_
        
        self.k = k
        self.cov_invs = []
        self.cov_dets = []
        for i in range(self.k):
            cov_inv = np.linalg.inv(self.covs[i])
            cov_det = np.linalg.det(self.covs[i])
            self.cov_invs.append(cov_inv)
            self.cov_dets.append(cov_det)

    def pdf(self, X):
        N = X.shape[0]
        p = np.zeros(N, dtype=np.float64)
        for i in range(self.k):
            p_comp = pdf_multidim(X, self.means[i], self.cov_invs[i], self.cov_dets[i])
            p += self.weights[i] * p_comp
        return p

class PDFMultiDim():
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

    def pdf(self, X):
        return pdf_multidim(X, self.mean, self.cov_inv, self.cov_det)

class GMMLaserDetector(LaserDetector):

    def __init__(self, laser_samples_path):
        super().__init__()

        # Build laser model from GMM
        laser_samples = self.__get_samples(laser_samples_path)
        self.pdf_laser = GMM(k=10, samples=laser_samples)

    def __find_mean_cov(self, img):
        img32 = img.astype(np.float32)
        pixels = img32.reshape(-1, 3)
        mean_hsv = np.mean(pixels, axis=0)
        cov_hsv = np.cov(pixels, rowvar=False)
        return mean_hsv, cov_hsv
    
    def __get_samples(self, path):
        dirs = glob.glob(f"{path}*")
        pixels = np.empty((0, 3), dtype=np.float32)

        for p in dirs:
            img = cv2.imread(p)
            if img is None:
                print(f"Error reading image: {p}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Convert to float32, then flatten
            pixels = np.vstack((pixels, img.reshape(-1, 3).astype(np.float32)))
        return pixels
    
    def find_laser(self, img_bgr):
        """
        Using Bayesian inference, classify pixels with Gaussian models and return laser point
        """
        # Convert to RGB and HSV color spaces
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Flatten (H*W, 3)
        H, W, _ = img_hsv.shape
        pixels_hsv = img_hsv.reshape(-1, 3).astype(np.float32)

        # Priors - right now hard coded, should be related to how many total pixels in image
        prior_laser = 0.00001 # 0.00001
        prior_bg = 0.99999 # 0.99999

        # Build background model
        bg_mean, bg_cov = self.__find_mean_cov(img_hsv)
        pdf_bg = PDFMultiDim(bg_mean, bg_cov)

        # Compute the likelihoods
        likelihood_bg = pdf_bg.pdf(pixels_hsv)
        likelihood_laser = self.pdf_laser.pdf(pixels_hsv)

        # Combine with priors => posterior
        numerator_laser = likelihood_laser * prior_laser
        numerator_bg = likelihood_bg * prior_bg
        norm_const = numerator_laser + numerator_bg

        # Posterior for laser
        posterior_laser = numerator_laser / norm_const
        # Classify as laser if posterior_laser > posterior_bg => posterior_laser > 0.5
        laser_mask = (posterior_laser > 0.5).reshape(H, W)

        # Color the laser pixels in BGR
        img_bgr[laser_mask] = [128, 0, 0]

        # Return the centroid
        laser_rows, laser_cols = np.where(laser_mask)
        centroid_row = None
        centroid_col = None
        if len(laser_rows) > 0:
            centroid_row = np.mean(laser_rows)
            centroid_col = np.mean(laser_cols)

        # Show results
        plt.scatter(
            [centroid_col], [centroid_row],
            color='blue', marker='x', s=50, label='Laser Pixel Mean'
        )
        plt.legend()
        # Shows RGB of original image
        plt.imshow(cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB))
        #plt.imshow(img_rgb)
        plt.title("Laser Pixels (Blue)")
        plt.show()

        return centroid_col, centroid_row

if __name__ == "__main__":
    path = sys.argv[1]
    img_org = cv2.imread(path)

    # Read directly from ORF without compression
    # Not converting colors correctly atm
    #with rawpy.imread(path) as raw:
    #    img_org = raw.postprocess(use_camera_wb=True)

    # Slight denoise
    img_bgr_denoised = cv2.fastNlMeansDenoisingColored(img_org,None,6,6,4,10)
    img_bgr = cv2.bilateralFilter(img_bgr_denoised,10,75,75)
    gmm_laser_detector = GMMLaserDetector(laser_samples_path="laser_data/")
    x, y = gmm_laser_detector.find_laser(img_bgr)