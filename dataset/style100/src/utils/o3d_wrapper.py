import numpy as np
import open3d as o3d
USE_PYMESH = False
# try:
#     import pymesh
# except ImportError as e:
#     print(e)
#     print("Failed to load pymesh. Open3d will be used")
#     USE_PYMESH = False


def read_obj(path):
    """
    read verts and faces from obj file. This func will convert quad mesh to triangle mesh
    """
    with open(path) as f:
        lines = f.read().splitlines()
    verts = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            verts.append(np.array([float(k) for k in line.split(' ')[1:]]))
        elif line.startswith('f '):
            try:
                onef = np.array([int(k) for k in line.split(' ')[1:]])
            except ValueError:
                continue
            if len(onef) == 4:
                faces.append(onef[[0, 1, 2]])
                faces.append(onef[[0, 2, 3]])
            elif len(onef) > 4:
                pass
            else:
                faces.append(onef)
    if len(faces) == 0:
        return np.stack(verts), None
    else:
        return np.stack(verts), np.stack(faces)-1


class MeshO3d(object):
    def __init__(self, v=None, f=None,vc=None, filename=None):
        self.m = o3d.geometry.TriangleMesh()
        if v is not None:
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        elif filename is not None:
            v, f = read_obj(filename)
            self.m = o3d.geometry.TriangleMesh()
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        self.vc = vc # stub, not work.
    @property
    def v(self):
        return np.asarray(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = o3d.utility.Vector3dVector(value.astype(np.float32))

    @property
    def f(self):
        return np.asarray(self.m.triangles)

    @f.setter
    def f(self, value):
        self.m.triangles = o3d.utility.Vector3iVector(value.astype(np.int32))

    @property
    def arr(self):
        return [self.v,self.f]

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=True)

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=False, compressed=True)

class MeshO3d_compatile(object):
    def __init__(self, v=None, f=None,vc=None, filename=None):
        self.m = o3d.geometry.TriangleMesh()
        if v is not None:
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int))
        elif filename is not None:
            v, f = read_obj(filename)
            self.m = o3d.geometry.TriangleMesh()
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int))
        self.vc = vc # stub, not work.
    @property
    def v(self):
        return np.asarray(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = o3d.utility.Vector3dVector(value.astype(np.float))

    @property
    def f(self):
        return np.asarray(self.m.triangles)

    @f.setter
    def f(self, value):
        self.m.triangles = o3d.utility.Vector3iVector(value.astype(np.int))

    @property
    def arr(self):
        return [self.v,self.f]

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=True)

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=False, compressed=True)


class MeshPyMesh(object):
    def __init__(self, v=None, f=None, vc=None, filename=None):
        if v is not None:
            self.m = pymesh.form_mesh(v, f)
        elif filename is not None:
            self.m = pymesh.load_mesh(filename)
        self.m.add_attribute('vertex_red')
        self.m.add_attribute('vertex_green')
        self.m.add_attribute('vertex_blue')
        if vc is not None:
            self.vc = vc

    @property
    def v(self):
        return np.copy(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = value

    @property
    def f(self):
        return np.copy(self.m.faces)

    @f.setter
    def f(self, value):
        self.m.faces = value

    @property
    def vc(self):
        return np.stack((self.m.get_attribute('vertex_red'), self.m.get_attribute('vertex_green'),
                         self.m.get_attribute('vertex_blue')), 1)/255

    @vc.setter
    def vc(self, value):
        value = np.copy(value) * 255
        self.m.set_attribute('vertex_red', value[:, 0])
        self.m.set_attribute('vertex_green', value[:, 1])
        self.m.set_attribute('vertex_blue', value[:, 2])

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        pymesh.save_mesh(fpath, self.m)

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        if self.vc.size == 0:
            pymesh.save_mesh(fpath, self.m)
        else:
            # import IPython; IPython.embed()
            pymesh.save_mesh(fpath, self.m, 'vertex_red', 'vertex_green', 'vertex_blue')

class MeshPyMeshLike(object):
    def __init__(self, v=None, f=None, vc=None, filename=None):
        self.m = o3d.geometry.TriangleMesh()
        if v is not None:
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        elif filename is not None:
            v, f = read_obj(filename)
            self.m = o3d.geometry.TriangleMesh()
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        if vc is not None:
            self._vc = vc
        else:
            self._vc = None

    @property
    def v(self):
        return np.asarray(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = o3d.utility.Vector3dVector(value.astype(np.float32))

    @property
    def f(self):
        return np.asarray(self.m.triangles)

    @f.setter
    def f(self, value):
        self.m.triangles = o3d.utility.Vector3iVector(value.astype(np.int32))

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        # o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=True)
        with open(fpath,'w') as f:
            if self.vc is not None:
                assert len(self.vc) == len(self.v)
                for vv,vc in zip(self.v,self.vc):
                    f.write(f'v {" ".join([str(ele) for ele in vv])} {" ".join([str(ele) for ele in vc[:3]])}\n')
            else:
                for vv in self.v:
                    f.write(f'v {" ".join([str(ele) for ele in vv])}\n')
            for ff in self.f:
                f.write(f'f {" ".join([str(ele + 1) for ele in ff])}\n')

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=False, compressed=True)

    @property
    def vc(self):
        return self._vc
        # return np.stack((self.m.get_attribute('vertex_red'), self.m.get_attribute('vertex_green'),
        #                  self.m.get_attribute('vertex_blue')), 1)/255

    @vc.setter
    def vc(self, value):
        self._vc = value.copy()
        # value = np.copy(value) * 255
        # self.m.set_attribute('vertex_red', value[:, 0])
        # self.m.set_attribute('vertex_green', value[:, 1])
        # self.m.set_attribute('vertex_blue', value[:, 2])


def write_obj_distri(path, body_joint, joint_color, edge):
   with open(path, 'w') as f:
       for i in range(body_joint.shape[0]):
           f.write("v " + str(body_joint[i, 0])+ " " + str(body_joint[i, 1]) + " " + str(body_joint[i, 2]) + " " + str(joint_color[i, 0])+ " " + str(joint_color[i, 1]) + " " + str(joint_color[i, 2]))
           f.write('\n')
       for j in range(edge.shape[0]):
           f.write("f " + str(edge[j, 0]+1) + " " + str(edge[j, 1]+1) + " " + str(edge[j, 2]+1) + "\n")

if USE_PYMESH:
    Mesh = MeshPyMesh
else:
    # Mesh = MeshO3d
    Mesh = MeshO3d_compatile