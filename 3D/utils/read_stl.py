import numpy as np
from stl.mesh import Mesh

def extract_stl_info(filename):
    mesh_import = Mesh.from_file(filename)
    all_points = mesh_import.points

    uniq_pts = all_points.reshape(3*len(mesh_import.points),3)
    vertices,_ = np.unique(uniq_pts,axis=0,return_index=True)
    vertices = np.float64(vertices)

    print('Number of triangles: ',len(all_points))

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
    import time
    # filename_exact = 'geom_shape/Bunny.stl'
    # filename = 'geom_shapes/Bunny_500.stl'
    # filename = 'geom_shapes/Bunny_808.stl'
    # filename = 'geom_shapes/Bunny_1310.stl'
    # filename = 'geom_shapes/Bunny_2120.stl'
    # filename = 'geom_shapes/Bunny_3432.stl'
    # filename = 'geom_shapes/Bunny_5555.stl'
    # filename = 'geom_shapes/Bunny_9000.stl'
    # filename = 'geom_shapes/Bunny_14560.stl'
    # filename = 'geom_shapes/Bunny_25000.stl'
    # filename = 'geom_shapes/Bunny_38160.stl'
    # filename = 'geom_shapes/Bunny_64000.stl'
    filename = 'geom_shapes/Bunny_100000.stl'

    # flag = "Fuselage"
    # filename = 'geom_shape/Fuselage_25k.stl'

    # flag = "Human"
    # filename = 'geom_shape/Human_25k.stl'

    # flag = "Battery"
    # filename = 'geom_shape/Battery_25k.stl'

    # flag = "Luggage"
    # filename = 'geom_shape/Luggage_25k.stl'

    # flag = "Wing"
    # filename = 'geom_shape/Wing_25k.stl'

    # filename = 'geom_shape/armadillo_exact.stl'
    # filename = 'geom_shape/armadillo_100k.stl'
    
    # filename = 'geom_shape/buddha_exact.stl'
    # filename = 'geom_shape/buddha_100k.stl'

    # filename = 'geom_shape/dragon_exact.stl'
    # filename = 'geom_shape/dragon_100k.stl'

    t1 = time.time()
    verts, norms = extract_stl_info(filename)
    t2 = time.time()
    print('Runtime: ',t2-t1)

    data = np.stack((verts, norms))