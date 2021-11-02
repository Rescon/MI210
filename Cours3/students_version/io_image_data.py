import h5py

def saveH5(file_name,dataset_name,np_array):
    """ Function that saves numpy arrays in a binary file h5
    Args:
        file_name (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        dataset_name (str): the dataset name 
        np_array (numpy.array): the data to be saved
    """

    f = h5py.File(file_name, "w")
    f.create_dataset(dataset_name,data =np_array);
    f.close()
    
    
def readH5(file_name, dataset_name):
    """ Function that reads numpy arrays in a binary file hdf5
    Args:
        file_name (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        dataset_name (str): the dataset name 
    Return:
        np_array (numpy.array): the read data
    """

    f = h5py.File(file_name, "r")
    np_array = f[dataset_name][:]
    return np_array
