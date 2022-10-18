import torch
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
import os
import torch
import numpy
import vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa
from torch_geometric.data import Data
from embed.SpectralEmbedding.embedding import Embedding
from torch_geometric.transforms import FaceToEdge

# with zipfile.ZipFile('mindboggle/lh.zip', 'r') as existing_zip:
#     existing_zip.extractall('mindboggle/lh')
# with zipfile.ZipFile('mindboggle/lh_eig.zip', 'r') as existing_zip:
#     existing_zip.extractall('mindboggle/lh_eig')
path = 'mindboggle/mesh'

filelist = [f for f in os.listdir(path) if not f.startswith('.')]
filelist.sort()
filelist.remove('ico')
filelist.remove('sphere.sh')

data_list1 = []
data_list2 = []
data_list3 = []

for i in tqdm(range(101)):
    # f1 = open(path + '/' + filelist[i] + '/lh.x.txt', 'r')
    # f2 = open(path + '/' + filelist[i] + '/lh.y.txt', 'r')
    # f3 = open(path + '/' + filelist[i] + '/lh.z.txt', 'r')
    # f4 = open(path + '/' + filelist[i] + '/lh.curv.txt', 'r')
    # f5 = open(path + '/' + filelist[i] + '/lh.iH.txt', 'r')
    # f6 = open(path + '/' + filelist[i] + '/lh.sulc.txt', 'r')
    # f7 = open(path + '/' + filelist[i] + '/lh.thickness.txt', 'r')
    # f8 = open(path + '/' + filelist[i] + '/lh.label.txt', 'r')

    # line1 = f1.read().splitlines()
    # line2 = f2.read().splitlines()
    # line3 = f3.read().splitlines()
    # line4 = f4.read().splitlines()
    # line5 = f5.read().splitlines()
    # line6 = f6.read().splitlines()
    # line7 = f7.read().splitlines()
    # line8 = f8.read().splitlines()

    # f1.close(),f2.close(),f3.close(),f4.close(),f5.close(),f6.close(),f7.close(),f8.close()

    # line1 = [float(i) for i in line1]
    # line2 = [float(i) for i in line2]
    # line3 = [float(i) for i in line3]
    # line4 = [float(i) for i in line4]
    # line5 = [float(i) for i in line5]
    # line6 = [float(i) for i in line6]
    # line7 = [float(i) for i in line7]
    # line8 = [float(i) for i in line8]

    # x1,x2,x3,x4,x5,x6,x7,y = torch.tensor(line1),torch.tensor(line2),torch.tensor(line3),torch.tensor(line4)\
    #     ,torch.tensor(line5),torch.tensor(line6),torch.tensor(line7),torch.tensor(line8)
    # x1,x2,x3,x4,x5,x6,x7 = x1.view(-1,1),x2.view(-1,1),x3.view(-1,1),x4.view(-1,1),x5.view(-1,1),x6.view(-1,1),x7.view(-1,1)

    # x = torch.cat((x1,x2,x3,x4,x5,x6,x7),-1)
    # y -= 1
    
    reader = vtk.vtkPolyDataReader()
    mesh_path = os.path.join(path,filelist[i],'rh.white.vtk')
    reader.SetFileName(mesh_path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    polydata = reader.GetOutput()
    numpy_array_of_face = dsa.WrapDataObject(polydata).Polygons
    numpy_array_of_vertex = dsa.WrapDataObject(polydata).Points
    face = []
    for j in range(int(numpy_array_of_face.shape[0]/4)):
        a = [numpy_array_of_face[4*j+1],numpy_array_of_face[4*j+2],numpy_array_of_face[4*j+3]]
        face.append(a)
    face = np.array(face)
    vertex = np.array(numpy_array_of_vertex)
    
    ref_data_embedded = Embedding(face, vertex, 100, norm = False, is_weight = False)
    ref_data_embedded.embedding()
    u = torch.tensor(ref_data_embedded.vectors).float()
    e = torch.tensor(ref_data_embedded.Lambda).float().unsqueeze(-1)
    
    ref_data_embedded = Embedding(face, vertex, 100, norm = True, is_weight = False)
    ref_data_embedded.embedding()
    nu = torch.tensor(ref_data_embedded.vectors).float()
    ne = torch.tensor(ref_data_embedded.Lambda).float().unsqueeze(-1)
    
    ref_data_embedded = Embedding(face, vertex, 100, norm = True, is_weight = True)
    ref_data_embedded.embedding()
    wu = torch.tensor(ref_data_embedded.vectors).float()
    we = torch.tensor(ref_data_embedded.Lambda).float().unsqueeze(-1)


    f4 = open(path + '/' + filelist[i] + '/rh.curv.txt', 'r')
    f5 = open(path + '/' + filelist[i] + '/rh.iH.txt', 'r')
    f6 = open(path + '/' + filelist[i] + '/rh.sulc.txt', 'r')
    f7 = open(path + '/' + filelist[i] + '/rh.thickness.txt', 'r')
    f8 = open('mindboggle/label' + '/' + filelist[i] + '/rh.label.txt', 'r')


    line4 = f4.read().splitlines()
    line5 = f5.read().splitlines()
    line6 = f6.read().splitlines()
    line7 = f7.read().splitlines()
    line8 = f8.read().splitlines()

    f4.close(),f5.close(),f6.close(),f7.close(),f8.close()


    line4 = [float(i) for i in line4]
    line5 = [float(i) for i in line5]
    line6 = [float(i) for i in line6]
    line7 = [float(i) for i in line7]
    line8 = [float(i) for i in line8]

    x4,x5,x6,x7,y = torch.tensor(line4),torch.tensor(line5),torch.tensor(line6),torch.tensor(line7),torch.tensor(line8)
    x4,x5,x6,x7 = x4.view(-1,1),x5.view(-1,1),x6.view(-1,1),x7.view(-1,1)

    x = torch.cat((torch.tensor(vertex),x4,x5,x6,x7),-1)
    y -= 1

    mu = torch.mean(x,0)
    sigma = torch.sqrt(torch.mean((x-mu)**2,0))
    x_n = (x-mu)/sigma
    
    face = torch.tensor(face)
    face = face.transpose(1,0)
    make_edge = FaceToEdge()
    edge_index = make_edge(Data(x = x_n.float(), face = face)).edge_index

    data = Data(x=x_n.float(), edge_index=edge_index,y=y.long(), u = u, e = e)
    data_list1.append(data)
    data = Data(x=x_n.float(), edge_index=edge_index,y=y.long(), u = nu, e = ne)
    data_list2.append(data)
    data = Data(x=x_n.float(), edge_index=edge_index,y=y.long(), u = wu, e = we)
    data_list3.append(data)


torch.save(data_list1,'eig_unnorm_rh')
torch.save(data_list2,'eig_norm_rh')
torch.save(data_list3,'eig_weight_rh')
