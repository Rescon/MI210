import numpy as np
import scipy.fftpack
import os
import pylab
import h5py
import io_image_data
import matplotlib.pyplot as plt

def test(input_file_name):
    f = h5py.File(input_file_name, 'r')
    images = f['images']
    print(images.shape)
    print(images[1].shape)
    plt.imshow(images[2399])

def get_sample_top_left_corner(iMin,iMax,jMin,jMax):
    """ Function that genereates randomly a position between i,j intervals [iMin,iMax], [jMin,jMax]
    Args:
        iMin (int): the i minimum coordinate (i is the column-position of an array)
        iMax (int): the i maximum coordinate (i is the column-position of an array)
        jMin (int): the j minimum coordinate (j is the row-position of an array)
        jMax (int): the j maximum coordinate (j is the row-position of an array)
    Returns:
        [i,j] (tuple(int,int)): random integers such iMin<=i<iMax,jMin<=j<jMax,
    """ 
    rng = np.random.default_rng()
    i = rng.integers(iMin, iMax)
    j = rng.integers(jMin, jMax)
    return [i,j]

def get_sample_image(image, sample_size, top_left_corner):
    """ Function that extracts a sample of an image with a given size and a given position
    Args:
        image (numpy.array) : input image to be sampled
        sample_size (tuple(int,int)): size of the sample
        top_left_corner (tuple(int,int)): positon of the top left corner of the sample within the image
    Returns:
        sample (numpy.array): image sample
    """ 
    #checking that the sample fits in the image
    if top_left_corner[0]+sample_size[0]>image.shape[0] or top_left_corner[1]+sample_size[1]>image.shape[1]:
        raise Exception("Sample too big for the image")
    sample = image[top_left_corner[0]:top_left_corner[0]+sample_size[0],top_left_corner[1]:top_left_corner[1]+sample_size[1]]
    return sample

def get_sample_PS(sample):
    """ Function that calculates the power spectrum of a image sample
    Args:
        sample (numpy.array): image sample
    Returns:
        sample_PS (numpy.array): power spectrum of the sample. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """ 
    ps = np.abs(np.fft.fft2(sample))**2
    ps = np.fft.fftshift(ps)
    return ps

def get_average_PS(input_file_name, sample_size, number_of_samples):
    """ Function that estimates the average power spectrum of a image database
    Args:
        input_file_name (str) : Absolute pathway to the image database stored in the hdf5
        sample_size (tuple(int,int)): size of the samples that are extrated from the images
        number_of_samples (int): number of image samples to consider in calculating the average
    Returns:
        average_PS (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """
    f = h5py.File(input_file_name, 'r')
    images = f['images']
    number_of_images = images.shape[0]
    rng = np.random.default_rng()
    average_PS = np.zeros(sample_size)

    for _ in range(number_of_samples):
        k = rng.integers(0, number_of_images)
        image = images[k]
        top_left_corner = get_sample_top_left_corner(0,image.shape[0]-sample_size[0],0,image.shape[1]-sample_size[1])
        sample = get_sample_image(image, sample_size, top_left_corner)
        ps = get_sample_PS(sample)
        average_PS += ps

    return average_PS/number_of_samples

def get_radial_freq(PS_size):
    """ Function that returns the Discrete Fourier Transform radial frequencies
    Args:
        PS_size (tuple(int,int)): the size of the window to calculate the frequencies
    Returns:
        radial_freq (numpy.array): radial frequencies in crescent order
    """
    fx = np.fft.fftshift(np.fft.fftfreq(PS_size[0], 1./PS_size[0]));
    fy = np.fft.fftshift(np.fft.fftfreq(PS_size[1], 1./PS_size[1]));
    [X,Y] = np.meshgrid(fx,fy);
    R = np.sqrt(X**2+Y**2);
    radial_freq = np.unique(R);
    radial_freq.sort()
    return radial_freq[radial_freq!=0]


def get_radial_PS(average_PS):
    """ Function that estimates the average power radial spectrum of a image database
    Args:
        average_PS (numpy.array) : average power spectrum of the database samples.
    Returns:
        average_PS_radial (numpy.array): average radial power spectrum of the database samples.
    """ 
    radial_freq = get_radial_freq(average_PS.shape)
    radial_PS = np.zeros(radial_freq.shape)
    for k, radius in enumerate(radial_freq):
        for i in range(average_PS.shape[0]):
            for j in range(average_PS.shape[1]):
                if np.sqrt((i - average_PS.shape[0]/2)**2 + (j - average_PS.shape[1]/2)**2) <= radius:
                    radial_PS[k]+=average_PS[i,j]
        radial_PS[k]/=np.pi*radius**2
    return radial_PS


def get_average_PS_local(input_file_name, sample_size, grid_size, number_of_samples):
    """ Function that estimates the local average power spectrum of a image database
    Args:
        input_file_name (str) : Absolute pathway to the image database
        sample_size (tuple(int,int)): size of the samples that are extrated from the images
        grid_size (tuple(int,int)): size of the grid that define the borders of each local region
        number_of_samples (int): number of image samples to consider in calculating the average
    Returns:
        average_PS_local (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """ 
    averagePS = np.zeros((32*grid_size[0],32*grid_size[1]))
    average_PS_local = np.reshape(averagePS,(grid_size[0],grid_size[1],32,32))

    with h5py.File(input_file_name, 'r')as f:
        dataset = f['images']

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                iMin = np.floor(dataset.shape[1]/grid_size[0])*i
                iMax = np.floor(dataset.shape[1]/grid_size[0])*(i+1)-sample_size[0]
                jMin = np.floor(dataset.shape[2]/grid_size[1])*j
                jMax = np.floor(dataset.shape[2]/grid_size[1])*(j+1)-sample_size[1]

                for nums in range(number_of_samples):
                    img = f.get('images')[nums % dataset.shape[0]]
                    topLeftCorner = get_sample_top_left_corner(iMin, iMax, jMin, jMax)
                    sample = get_sample_image(img, sample_size, topLeftCorner)
                    imgPS = get_sample_PS(sample)
                    average_PS_local[i,j] = np.array(average_PS_local[i,j]) + np.array(imgPS)

                average_PS_local[i, j] = average_PS_local[i, j] / number_of_samples

    return average_PS_local

def make_average_PS_figure(average_PS):
    """ Function that makes and save the figure with the power spectrum
    Args:
        average_PS (numpy.array): the average power spectrum in an array of shape [sampleShape[0],sampleShape[1]]
    """ 
    pylab.figure()
    pylab.imshow(np.log(average_PS),cmap = "gray")
    pylab.contour(np.log(average_PS))
    pylab.axis("off")
  #  pylab.savefig(figure_file_name)

def make_average_PS_radial_figure(radial_freq,average_PS_radial):
    """ Function that makes and save the figure with the power spectrum
    Args:
        average_PS (numpy.array) : the average power spectrum
        average_PS_radial (numpy.array): the average radial power spectrum
    """ 
    pylab.figure()
    pylab.loglog(radial_freq,average_PS_radial,'.')
    pylab.xlabel("Frequency")
    pylab.ylabel("Radial Power Spectrum")
    
    


def make_average_PS_local_figure(average_PS_local,grid_size):
    """ Function that makes and save the figure with the local power spectrum
    Args:
        average_PS_local (numpy.array): the average power spectrum in an array of shape [grid_size[0],grid_size[1],sampleShape[0],sampleShape[1]
        grid_size (tuple): size of the grid
    """ 
    pylab.figure()
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            pylab.subplot(grid_size[0],grid_size[1],i*grid_size[1]+j+1)
            pylab.imshow(np.log(average_PS_local[i,j]),cmap = "gray")
            pylab.contour(np.log(average_PS_local[i,j]))
            pylab.axis("off")








