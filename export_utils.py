import numpy as np
import os
from vtkmodules.vtkCommonCore import VTK_FLOAT
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter
from vtkmodules.vtkCommonDataModel import vtkImageData
import vtkmodules.util.numpy_support as numpy_support
from utils import convert_to_3D


def numpyToVTK(data):
    data_type = VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)

    img = vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])
    return img


def export_results(filename, tensor: np.array):
    img = numpyToVTK(tensor)
    writer = vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(img)
    writer.Write()


def export_time_dependent(U: list, name, N, directory='.'):
    for i in range(len(U)):
        u_tensor = convert_to_3D(U[i], N, N, N)
        n = str(i).zfill(4)
        filename = os.path.join(directory, name + n + '.vti')
        export_results(filename, u_tensor)
