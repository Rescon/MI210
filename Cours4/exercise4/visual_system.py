import os
import numpy as np
import pylab
from sklearn.decomposition import FastICA
import scipy.stats
import h5py
import image_data_analysis


def truncate_non_neg(x):
    """Function that truncates arrays od real numbers into arrays of non negatives.
    Args:
    x(numpy.array): input array
    Returns:
    y(numpy.array): array with positive or zero numbers
    """
    x = np.real(x)
    x = np.where(x>0,x,0)
    return x


def get_power_spectrum_whitening_filter(average_PS,noise_variance):
    """Function that estimates the whitening and denoising power spectrum filter
    Args:
    average_PS(numpy.array): average power spectrum of the observation
    noise_variance(double): variance of the gaussian white noise.
    Returns:
    w(numpy.array): whitening denoising filter
    """
    wPart1 = 1/np.sqrt(average_PS)
    wPart2 = np.divide(average_PS - noise_variance * average_PS.shape[0] * average_PS.shape[1], average_PS) #Or we can use np.floor
    wPart3 = wPart1 * wPart2
    W = truncate_non_neg(wPart3)
    W = np.fft.ifftshift(W)
    W = np.fft.ifft2(W)
    W = truncate_non_neg(W)
    W = np.fft.ifftshift(W)
    return W



def make_whitening_filters_figure(whitening_filters):
    pylab.figure()
    for i,whiteningFilter in enumerate(whitening_filters):
        pylab.subplot(1,len(whitening_filters),i+1)
        vmax = np.max(np.abs(whiteningFilter))
        vmin = -vmax
        pylab.imshow(whiteningFilter,cmap = 'gray',vmax = vmax, vmin = vmin)
        pylab.axis("off")



def get_ICA_input_data(dataset_file_name, sample_size, number_of_samples):
    """ Function that samples the input directory for later to be used by FastICA
    Args:
    inputFileName(str):: Absolute pathway to the image database hdf5 file
    sample_size (tuple(int,int)): size of the samples that are extrated from the images
    nSamples(int): number of samples that should be taken from the database
    Returns:
    X(numpy.array)nSamples, sample_size
    """
    f = h5py.File(dataset_file_name, 'r')
    images = f['images']
    selected_images_index = []
    rng = np.random.default_rng()
    X = np.zeros((number_of_samples, sample_size[0], sample_size[1]))

    for k in range(number_of_samples):
        i = rng.integers(0, images.shape[0])
        while i in selected_images_index:
            i = rng.integers(0, images.shape[0])

        top_left_corner = image_data_analysis.get_sample_top_left_corner(0, images[i].shape[0] - sample_size[0], 0, images[i].shape[1] - sample_size[1])
        X[k] = image_data_analysis.get_sample_image(images[i], sample_size, top_left_corner)
    
    return X



def pre_process(X):
    """Function that preprocess the data to be fed to the ICA algorithm
    Args:
    X(numpy array): input to be preprocessed
    Returns:
    X(numpy.array)
    """
    for i in range(X.shape[0]):
        X[i,:,:] = X[i,:,:] -np.mean(X[i])
    X = np.reshape(X,(X.shape[0], X.shape[1] * X.shape[2]))
    print(X.shape)
    return X

    
def get_IC(X):
    """Function that estimates the independent components of the data
    Args:
    X(numpy.array):preprocessed data
    Returns:
    W(numpy.array) the matrix of the independent sources of the data
    """
    ICA = FastICA(algorithm='parallel', whiten=True, tol=1e-1, max_iter=2000)
    ICA.fit(X)
    W = ICA.components_ #The linear operator to apply to the data to get the independent sources.
    return W

    
   
def make_idependent_components_figure(W, sample_size): 
    W = W.reshape([-1,]+sample_size)
    pylab.figure()
    for i in range(W.shape[0]):
        pylab.subplot(sample_size[0],sample_size[1],i+1)
        pylab.imshow(W[i],cmap = 'gray')
        pylab.axis("off")

def estimate_sources(W,X):
    """Function that estimates the independent sources of the data
    Args:
    W(numpy.array):The matrix of the independent components
    X(numpy.array):preprocessed data
    Returns:
    S(numpy.array) the matrix of the sources of X
    """
    return W.dot(np.transpose(X))

def estimate_sources_kurtosis(S):
    kur = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        kur[i] = scipy.stats.kurtosis(S[i], fisher=True)
    return kur

def make_kurtosis_figure(S):
    kurS = estimate_sources_kurtosis(S)
    X = np.linspace(0, kurS.size, kurS.size)

    pylab.figure()
    pylab.plot(X, kurS)
    pylab.xlabel("Source")
    pylab.ylabel("Kurtosis")
    pylab.title("Kurtosis of the different sources")
    pylab.show()
    print("done")


