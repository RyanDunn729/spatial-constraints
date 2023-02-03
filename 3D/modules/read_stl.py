import numpy as np
from stl.mesh import Mesh

def extract_stl_info(filename):
    mesh_import = Mesh.from_file(filename)
    all_points = mesh_import.points

    uniq_pts = all_points.reshape(3*len(mesh_import.points),3)
    vertices,_ = np.unique(uniq_pts,axis=0,return_index=True)
    vertices = np.float64(vertices)

    print('Num faces: ',len(all_points))
    print('Num "vertices": ',len(vertices))

    faces = all_points.reshape(len(all_points),3,3)
    face_centroids = np.mean(faces,axis=1)
    normals = mesh_import.get_unit_normals()

    return face_centroids, normals

def extract_stl_info_old(filename):
    mesh_import = Mesh.from_file(filename)
    faces = mesh_import.points
    all_points = faces.reshape(3*len(mesh_import.points),3)
    vertices,_ = np.unique(all_points,axis=0,return_index=True)
    vertices = np.float64(vertices)

    print('Num faces: ',len(faces))
    # print('Num vertices: ',len(vertices))

    f_norms = np.tile(mesh_import.get_unit_normals(),(3,1))
    set = np.vstack((mesh_import.v0,mesh_import.v1,mesh_import.v2))
    v_norms = np.zeros((len(vertices),3))
    for i,vert in enumerate(vertices):
        ind = np.argwhere((vert == set).all(1)).ravel()
        v_norms[i] = np.sum(f_norms[ind],axis=0)
    # ind = np.argwhere(np.linalg.norm(v_norms,axis=1)==0).ravel()
    # print(vertices[ind])
    v_norms = np.einsum('ij,i->ij',v_norms,1/np.linalg.norm(v_norms,axis=1))

    return vertices, v_norms

if __name__ == '__main__':
    import pickle
    import time
    # filename_exact = 'stl-files/Bunny.stl'
    # filename = 'stl-files/Bunny_77.stl'
    # filename = 'stl-files/Bunny_108.stl'
    # filename = 'stl-files/Bunny_252.stl'
    # filename = 'stl-files/Bunny_297.stl'
    # filename = 'stl-files/Bunny_327.stl'
    # filename = 'stl-files/Bunny_377.stl'
    # filename = 'stl-files/Bunny_412.stl'
    # filename = 'stl-files/Bunny_502.stl'
    # filename = 'stl-files/Bunny_677.stl'
    # filename = 'stl-files/Bunny_1002.stl'
    # filename = 'stl-files/Bunny_5002.stl'
    # filename = 'stl-files/Bunny_10002.stl'
    # filename = 'stl-files/Bunny_25002.stl'
    # filename = 'stl-files/Bunny_40802.stl'
    # filename = 'stl-files/Bunny_63802.stl'
    filename = 'stl-files/Bunny_100002.stl'

    # filename = 'stl-files/Heart_5002.stl'

    # flag = "Fuselage"
    # filename = 'stl-files/Fuselage_25k.stl'

    # flag = "Human"
    # filename = 'stl-files/Human_25k.stl'

    # flag = "Battery"
    # filename = 'stl-files/Battery_25k.stl'

    # flag = "Luggage"
    # filename = 'stl-files/Luggage_25k.stl'

    # flag = "Wing"
    # filename = 'stl-files/Wing_25k.stl'

    # filename = 'stl-files/armadillo_exact.stl'
    # filename = 'stl-files/armadillo_100k.stl'
    
    # filename = 'stl-files/buddha_exact.stl'
    # filename = 'stl-files/buddha_100k.stl'

    # filename = 'stl-files/dragon_exact.stl'
    # filename = 'stl-files/dragon_100k.stl'

    t1 = time.time()
    verts, norms = extract_stl_info(filename)
    t2 = time.time()
    print('Runtime: ',t2-t1)

    data = np.stack((verts, norms))