import operator

import numpy as np
import numpy.core.umath_tests as ut

# import data_utils.AnimationStructure as AnimationStructure
# import data_utils.AnimationPositions as AnimationStructure
from src.data_utils.Quaternions import Quaternions

class Animation:
    """
    Animation is a numpy-like wrapper for animation data
    
    Animation data consists of several arrays consisting
    of F frames and J joints.
    
    The animation is specified by
    
        rotations : (F, J) Quaternions | Joint Rotations
        positions : (F, J, 3) ndarray  | Joint Positions
    
    The base pose is specified by
    
        orients   : (J) Quaternions    | Joint Orientations
        offsets   : (J, 3) ndarray     | Joint Offsets
        
    And the skeletal structure is specified by
        
        parents   : (J) ndarray        | Joint Parents
    """
    
    def __init__(self, rotations, positions, orients, offsets, parents):
        
        self.rotations = rotations
        self.positions = positions
        self.orients   = orients
        self.offsets   = offsets
        self.parents   = parents
    
    def __op__(self, op, other):
        return Animation(
            op(self.rotations, other.rotations),
            op(self.positions, other.positions),
            op(self.orients, other.orients),
            op(self.offsets, other.offsets),
            op(self.parents, other.parents))

    def __iop__(self, op, other):
        self.rotations = op(self.roations, other.rotations)
        self.positions = op(self.roations, other.positions)
        self.orients   = op(self.orients, other.orients)
        self.offsets   = op(self.offsets, other.offsets)
        self.parents   = op(self.parents, other.parents)
        return self
    
    def __sop__(self, op):
        return Animation(
            op(self.rotations),
            op(self.positions),
            op(self.orients),
            op(self.offsets),
            op(self.parents))
    
    def __add__(self, other): return self.__op__(operator.add, other)
    def __sub__(self, other): return self.__op__(operator.sub, other)
    def __mul__(self, other): return self.__op__(operator.mul, other)
    def __div__(self, other): return self.__op__(operator.div, other)
    
    def __abs__(self): return self.__sop__(operator.abs)
    def __neg__(self): return self.__sop__(operator.neg)
    
    def __iadd__(self, other): return self.__iop__(operator.iadd, other)
    def __isub__(self, other): return self.__iop__(operator.isub, other)
    def __imul__(self, other): return self.__iop__(operator.imul, other)
    def __idiv__(self, other): return self.__iop__(operator.idiv, other)
    
    def __len__(self): return len(self.rotations)
    
    def __getitem__(self, k):
        if isinstance(k, tuple):
            return Animation(
                self.rotations[k],
                self.positions[k],
                self.orients[k[1:]],
                self.offsets[k[1:]],
                self.parents[k[1:]]) 
        else:
            return Animation(
                self.rotations[k],
                self.positions[k],
                self.orients,
                self.offsets,
                self.parents) 
        
    def __setitem__(self, k, v): 
        if isinstance(k, tuple):
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k[1:], v.orients)
            self.offsets.__setitem__(k[1:], v.offsets)
            self.parents.__setitem__(k[1:], v.parents)
        else:
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k, v.orients)
            self.offsets.__setitem__(k, v.offsets)
            self.parents.__setitem__(k, v.parents)
        
    @property
    def shape(self): return (self.rotations.shape[0], self.rotations.shape[1])
            
    def copy(self): return Animation(
        self.rotations.copy(), self.positions.copy(), 
        self.orients.copy(), self.offsets.copy(), 
        self.parents.copy())
    
    def repeat(self, *args, **kw):
        return Animation(
            self.rotations.repeat(*args, **kw),
            self.positions.repeat(*args, **kw),
            self.orients, self.offsets, self.parents)
        
    def ravel(self):
        return np.hstack([
            self.rotations.log().ravel(),
            self.positions.ravel(),
            self.orients.log().ravel(),
            self.offsets.ravel()])
        
    @classmethod
    def unravel(clas, anim, shape, parents):
        nf, nj = shape
        rotations = anim[nf*nj*0:nf*nj*3]
        positions = anim[nf*nj*3:nf*nj*6]
        orients   = anim[nf*nj*6+nj*0:nf*nj*6+nj*3]
        offsets   = anim[nf*nj*6+nj*3:nf*nj*6+nj*6]
        return cls(
            Quaternions.exp(rotations), positions,
            Quaternions.exp(orients), offsets,
            parents.copy())
    
    
""" Maya Interaction """

def load_to_maya(anim, names=None, radius=0.5):
    """
    Load Animation Object into Maya as Joint Skeleton
    loads each frame as a new keyfame in maya.
    
    If the animation is too slow or too fast perhaps
    the framerate needs adjusting before being loaded
    such that it matches the maya scene framerate.
    
    
    Parameters
    ----------
    
    anim : Animation
        Animation to load into Scene
        
    names : [str]
        Optional list of Joint names for Skeleton
    
    Returns
    -------
    
    List of Maya Joint Nodes loaded into scene
    """
    
    import pymel.core as pm
    
    joints = []
    frames = range(1, len(anim)+1)
    
    if names is None: names = ["joint_" + str(i) for i in range(len(anim.parents))]
    
    for i, offset, orient, parent, name in zip(range(len(anim.offsets)), anim.offsets, anim.orients, anim.parents, names):
    
        if parent < 0:
            pm.select(d=True)
        else:
            pm.select(joints[parent])
        
        joint = pm.joint(n=name, p=offset, relative=True, radius=radius)
        joint.setOrientation([orient[1], orient[2], orient[3], orient[0]])
        
        curvex = pm.nodetypes.AnimCurveTA(n=name + "_rotateX")
        curvey = pm.nodetypes.AnimCurveTA(n=name + "_rotateY")
        curvez = pm.nodetypes.AnimCurveTA(n=name + "_rotateZ")  
        
        jrotations = (-Quaternions(orient[np.newaxis]) * anim.rotations[:,i]).euler()
        curvex.addKeys(frames, jrotations[:,0])
        curvey.addKeys(frames, jrotations[:,1])
        curvez.addKeys(frames, jrotations[:,2])
        
        pm.connectAttr(curvex.output, joint.rotateX)
        pm.connectAttr(curvey.output, joint.rotateY)
        pm.connectAttr(curvez.output, joint.rotateZ)
        
        offsetx = pm.nodetypes.AnimCurveTU(n=name + "_translateX")
        offsety = pm.nodetypes.AnimCurveTU(n=name + "_translateY")
        offsetz = pm.nodetypes.AnimCurveTU(n=name + "_translateZ")
        
        offsetx.addKeys(frames, anim.positions[:,i,0])
        offsety.addKeys(frames, anim.positions[:,i,1])
        offsetz.addKeys(frames, anim.positions[:,i,2])
        
        pm.connectAttr(offsetx.output, joint.translateX)
        pm.connectAttr(offsety.output, joint.translateY)
        pm.connectAttr(offsetz.output, joint.translateZ)
        
        joints.append(joint)
    
    return joints

def load_from_maya(root, start, end):
    """
    Load Animation Object from Maya Joint Skeleton    
    
    Parameters
    ----------
    
    root : PyNode
        Root Joint of Maya Skeleton
        
    start, end : int, int
        Start and End frame index of Maya Animation
    
    Returns
    -------
    
    animation : Animation
        Loaded animation from maya
        
    names : [str]
        Joint names from maya   
    """

    import pymel.core as pm
    
    original_time = pm.currentTime(q=True)
    pm.currentTime(start)
    
    """ Build Structure """
    
    names, parents = AnimationStructure.load_from_maya(root)
    descendants = AnimationStructure.descendants_list(parents)
    orients = Quaternions.id(len(names))
    offsets = np.array([pm.xform(j, q=True, translation=True) for j in names])
    
    for j, name in enumerate(names):
        scale = pm.xform(pm.PyNode(name), q=True, scale=True, relative=True)
        if len(descendants[j]) == 0: continue
        offsets[descendants[j]] *= scale
    
    """ Load Animation """

    eulers    = np.zeros((end-start, len(names), 3))
    positions = np.zeros((end-start, len(names), 3))
    rotations = Quaternions.id((end-start, len(names)))
    
    for i in range(end-start):
        
        pm.currentTime(start+i+1, u=True)
        
        scales = {}
        
        for j, name, parent in zip(range(len(names)), names, parents):
            
            node = pm.PyNode(name)
            
            if i == 0 and pm.hasAttr(node, 'jointOrient'):
                ort = node.getOrientation()
                orients[j] = Quaternions(np.array([ort[3], ort[0], ort[1], ort[2]]))
            
            if pm.hasAttr(node, 'rotate'):    eulers[i,j]    = np.radians(pm.xform(node, q=True, rotation=True))
            if pm.hasAttr(node, 'translate'): positions[i,j] = pm.xform(node, q=True, translation=True)
            if pm.hasAttr(node, 'scale'):     scales[j]      = pm.xform(node, q=True, scale=True, relative=True)

        for j in scales:
            if len(descendants[j]) == 0: continue
            positions[i,descendants[j]] *= scales[j] 
        
        positions[i,0] = pm.xform(root, q=True, translation=True, worldSpace=True)
    
    rotations = orients[np.newaxis] * Quaternions.from_euler(eulers, order='xyz', world=True)
    
    """ Done """
    
    pm.currentTime(original_time)
    
    return Animation(rotations, positions, orients, offsets, parents), names
    
    
def transforms_local(anim):
    """
    Computes Animation Local Transforms
    
    As well as a number of other uses this can
    be used to compute global joint transforms,
    which in turn can be used to compete global
    joint positions
    
    Parameters
    ----------
    
    anim : Animation
        Input animation
        
    Returns
    -------
    
    transforms : (F, J, 4, 4) ndarray
    
        For each frame F, joint local
        transforms for each joint J
    """
    
    transforms = anim.rotations.transforms()
    transforms = np.concatenate([transforms, np.zeros(transforms.shape[:2] + (3, 1))], axis=-1)
    transforms = np.concatenate([transforms, np.zeros(transforms.shape[:2] + (1, 4))], axis=-2)
    transforms[:,:,0:3,3] = anim.positions
    transforms[:,:,3:4,3] = 1.0
    return transforms

    
def transforms_multiply(t0s, t1s):
    """
    Transforms Multiply
    
    Multiplies two arrays of animation transforms
    
    Parameters
    ----------
    
    t0s, t1s : (F, J, 4, 4) ndarray
        Two arrays of transforms
        for each frame F and each
        joint J
        
    Returns
    -------
    
    transforms : (F, J, 4, 4) ndarray
        Array of transforms for each
        frame F and joint J multiplied
        together
    """
    
    return ut.matrix_multiply(t0s, t1s)
    
def transforms_inv(ts):
    fts = ts.reshape(-1, 4, 4)
    fts = np.array(list(map(lambda x: np.linalg.inv(x), fts)))
    return fts.reshape(ts.shape)
    
def transforms_blank(anim):
    """
    Blank Transforms
    
    Parameters
    ----------
    
    anim : Animation
        Input animation
    
    Returns
    -------
    
    transforms : (F, J, 4, 4) ndarray
        Array of identity transforms for 
        each frame F and joint J
    """

    ts = np.zeros(anim.shape + (4, 4)) 
    ts[:,:,0,0] = 1.0; ts[:,:,1,1] = 1.0;
    ts[:,:,2,2] = 1.0; ts[:,:,3,3] = 1.0;
    return ts
    
def transforms_global(anim):
    """
    Global Animation Transforms
    
    This relies on joint ordering
    being incremental. That means a joint
    J1 must not be a ancestor of J0 if
    J0 appears before J1 in the joint
    ordering.
    
    Parameters
    ----------
    
    anim : Animation
        Input animation
    
    Returns
    ------
    
    transforms : (F, J, 4, 4) ndarray
        Array of global transforms for 
        each frame F and joint J
    """
    
    joints  = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals  = transforms_local(anim)
    globals = transforms_blank(anim)

    globals[:,0] = locals[:,0]

    for i in range(1, anim.shape[1]):
        globals[:,i] = transforms_multiply(globals[:,anim.parents[i]], locals[:,i])

    return globals
    
    
def positions_global(anim):
    """
    Global Joint Positions
    
    Given an animation compute the global joint
    positions at at every frame
    
    Parameters
    ----------
    
    anim : Animation
        Input animation
        
    Returns
    -------
    
    positions : (F, J, 3) ndarray
        Positions for every frame F 
        and joint position J
    """
    
    positions = transforms_global(anim)[:,:,:,3]
    return positions[:,:,:3] / positions[:,:,3,np.newaxis]
    
""" Rotations """
    
def rotations_global(anim):
    """
    Global Animation Rotations
    
    This relies on joint ordering
    being incremental. That means a joint
    J1 must not be a ancestor of J0 if
    J0 appears before J1 in the joint
    ordering.
    
    Parameters
    ----------
    
    anim : Animation
        Input animation
        
    Returns
    -------
    
    points : (F, J) Quaternions
        global rotations for every frame F 
        and joint J
    """

    joints  = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals  = anim.rotations
    globals = Quaternions.id(anim.shape)
    
    globals[:,0] = locals[:,0]
    
    for i in range(1, anim.shape[1]):
        globals[:,i] = globals[:,anim.parents[i]] * locals[:,i]
        
    return globals
    
def rotations_parents_global(anim):
    rotations = rotations_global(anim)
    rotations = rotations[:,anim.parents]
    rotations[:,0] = Quaternions.id(len(anim))
    return rotations
    
def rotations_load_to_maya(rotations, positions, names=None):
    """
    Load Rotations into Maya
    
    Loads a Quaternions array into the scene
    via the representation of axis
    
    Parameters
    ----------
    
    rotations : (F, J) Quaternions 
        array of rotations to load
        into the scene where
            F = number of frames
            J = number of joints
    
    positions : (F, J, 3) ndarray 
        array of positions to load
        rotation axis at where:
            F = number of frames
            J = number of joints
            
    names : [str]
        List of joint names
    
    Returns
    -------
    
    maxies : Group
        Grouped Maya Node of all Axis nodes
    """
    
    import pymel.core as pm

    if names is None: names = ["joint_" + str(i) for i in range(rotations.shape[1])]
    
    maxis = []
    frames = range(1, len(positions)+1)
    for i, name in enumerate(names):
    
        name = name + "_axis"
        axis = pm.group(
             pm.curve(p=[(0,0,0), (1,0,0)], d=1, n=name+'_axis_x'),
             pm.curve(p=[(0,0,0), (0,1,0)], d=1, n=name+'_axis_y'),
             pm.curve(p=[(0,0,0), (0,0,1)], d=1, n=name+'_axis_z'),
             n=name)
        
        axis.rotatePivot.set((0,0,0))
        axis.scalePivot.set((0,0,0))
        axis.childAtIndex(0).overrideEnabled.set(1); axis.childAtIndex(0).overrideColor.set(13)
        axis.childAtIndex(1).overrideEnabled.set(1); axis.childAtIndex(1).overrideColor.set(14)
        axis.childAtIndex(2).overrideEnabled.set(1); axis.childAtIndex(2).overrideColor.set(15)
    
        curvex = pm.nodetypes.AnimCurveTA(n=name + "_rotateX")
        curvey = pm.nodetypes.AnimCurveTA(n=name + "_rotateY")
        curvez = pm.nodetypes.AnimCurveTA(n=name + "_rotateZ")  
        
        arotations = rotations[:,i].euler()
        curvex.addKeys(frames, arotations[:,0])
        curvey.addKeys(frames, arotations[:,1])
        curvez.addKeys(frames, arotations[:,2])
        
        pm.connectAttr(curvex.output, axis.rotateX)
        pm.connectAttr(curvey.output, axis.rotateY)
        pm.connectAttr(curvez.output, axis.rotateZ)
        
        offsetx = pm.nodetypes.AnimCurveTU(n=name + "_translateX")
        offsety = pm.nodetypes.AnimCurveTU(n=name + "_translateY")
        offsetz = pm.nodetypes.AnimCurveTU(n=name + "_translateZ")
        
        offsetx.addKeys(frames, positions[:,i,0])
        offsety.addKeys(frames, positions[:,i,1])
        offsetz.addKeys(frames, positions[:,i,2])
        
        pm.connectAttr(offsetx.output, axis.translateX)
        pm.connectAttr(offsety.output, axis.translateY)
        pm.connectAttr(offsetz.output, axis.translateZ)
    
        maxis.append(axis)
        
    return pm.group(*maxis, n='RotationAnimation')   
    
""" Offsets & Orients """

def orients_global(anim):

    joints  = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals  = anim.orients
    globals = Quaternions.id(anim.shape[1])
    
    globals[:,0] = locals[:,0]
    
    for i in range(1, anim.shape[1]):
        globals[:,i] = globals[:,anim.parents[i]] * locals[:,i]
        
    return globals

    
def offsets_transforms_local(anim):
    
    transforms = anim.orients[np.newaxis].transforms()
    transforms = np.concatenate([transforms, np.zeros(transforms.shape[:2] + (3, 1))], axis=-1)
    transforms = np.concatenate([transforms, np.zeros(transforms.shape[:2] + (1, 4))], axis=-2)
    transforms[:,:,0:3,3] = anim.offsets[np.newaxis]
    transforms[:,:,3:4,3] = 1.0
    return transforms
    
    
def offsets_transforms_global(anim):
    
    joints  = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals  = offsets_transforms_local(anim)
    globals = transforms_blank(anim)

    globals[:,0] = locals[:,0]
    
    for i in range(1, anim.shape[1]):
        globals[:,i] = transforms_multiply(globals[:,anim.parents[i]], locals[:,i])
        
    return globals
    
def offsets_global(anim):
    offsets = offsets_transforms_global(anim)[:,:,:,3]
    return offsets[0,:,:3] / offsets[0,:,3,np.newaxis]
    
""" Lengths """

def offset_lengths(anim):
    return np.sum(anim.offsets[1:]**2.0, axis=1)**0.5
    
    
def position_lengths(anim):
    return np.sum(anim.positions[:,1:]**2.0, axis=2)**0.5
    
    
""" Skinning """

def skin(anim, rest, weights, mesh, maxjoints=4):
    
    full_transforms = transforms_multiply(
        transforms_global(anim), 
        transforms_inv(transforms_global(rest[0:1])))
    
    weightids = np.argsort(-weights, axis=1)[:,:maxjoints]
    weightvls = np.array(list(map(lambda w, i: w[i], weights, weightids)))
    weightvls = weightvls / weightvls.sum(axis=1)[...,np.newaxis]
    
    verts = np.hstack([mesh, np.ones((len(mesh), 1))])
    verts = verts[np.newaxis,:,np.newaxis,:,np.newaxis]
    verts = transforms_multiply(full_transforms[:,weightids], verts)    
    verts = (verts[:,:,:,:3] / verts[:,:,:,3:4])[:,:,:,:,0]

    return np.sum(weightvls[np.newaxis,:,:,np.newaxis] * verts, axis=2)
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from scipy.spatial import distance

# from data_utils.Quaternions import Quaternions
# import data_utils.Animation as Animation


# import AnimationStructure

def constrain(positions, constraints):
    """
    Constrain animation positions given
    a number of VerletParticles constrains

    Parameters
    ----------

    positions : (F, J, 3) ndarray
        array of joint positions for
        F frames and J joints

    constraints : [(int, int, float, float, float)]
        A list of constraints in the format:
        (Joint1, Joint2, Masses1, Masses2, Lengths)

    Returns
    -------

    positions : (F, J, 3) ndarray
        joint positions for F
        frames and J joints constrained
        using the supplied constraints
    """

    from VerletParticles import VerletParticles

    particles = VerletParticles(positions, gravity=0.0, timestep=0.0)
    for i, j, w0, w1, l in constraints:
        particles.add_length_constraint(i, j, w0, w1, l)

    return particles.constrain()


def extremities(positions, count, **kwargs):
    """
    List of most extreme frame indices

    Parameters
    ----------

    positions : (F, J, 3) ndarray
        array of joint positions for
        F frames and J joints

    count : int
        Number of indices to return,
        does not include first and last
        frame which are always included

    static : bool
        Find extremities where root
        translation has been removed

    Returns
    -------

    indices : (C) ndarray
        Returns C frame indices of the
        most extreme frames including
        the first and last frames.

        Therefore if `count` it specified
        as `4` will return and array of
        `6` indices.
    """

    if kwargs.pop('static', False):
        positions = positions - positions[:, 0][:, np.newaxis, :]

    positions = positions.reshape((len(positions), -1))

    distance_matrix = distance.squareform(distance.pdist(positions))

    keys = [0]
    for _ in range(count - 1):
        keys.append(int(np.argmax(np.min(distance_matrix[keys], axis=0))))
    return np.array(keys)


def load_to_maya(positions, names=None, parents=None, color=None, radius=0.1, thickness=5.0):
    import pymel.core as pm
    import maya.mel as mel

    if names is None:
        names = ['joint_%i' % i for i in xrange(positions.shape[1])]

    if color is None:
        color = (0.5, 0.5, 0.5)

    mpoints = []
    frames = range(1, len(positions) + 1)
    for i, name in enumerate(names):

        # try:
        #    point = pm.PyNode(name)
        # except pm.MayaNodeError:
        #    point = pm.sphere(p=(0,0,0), n=name, radius=radius)[0]
        point = pm.sphere(p=(0, 0, 0), n=name, radius=radius)[0]

        jpositions = positions[:, i]

        for j, attr, attr_name in zip(xrange(3),
                                      [point.tx, point.ty, point.tz],
                                      ["_translateX", "_translateY", "_translateZ"]):
            conn = attr.listConnections()
            if len(conn) == 0:
                curve = pm.nodetypes.AnimCurveTU(n=name + attr_name)
                pm.connectAttr(curve.output, attr)
            else:
                curve = conn[0]
            curve.addKeys(frames, jpositions[:, j])

        mpoints.append(point)

    if parents != None:

        for i, p in enumerate(parents):
            if p == -1: continue
            pointname = names[i]
            parntname = names[p]
            conn = pm.PyNode(pointname).t.listConnections()
            if len(conn) != 0: continue

            curve = pm.curve(p=[[0, 0, 0], [0, 1, 0]], d=1, n=names[i] + '_curve')
            pm.connectAttr(pointname + '.t', names[i] + '_curve.cv[0]')
            pm.connectAttr(parntname + '.t', names[i] + '_curve.cv[1]')
            pm.select(curve)
            pm.runtime.AttachBrushToCurves()
            stroke = pm.selected()[0]
            brush = pm.listConnections(stroke.getChildren()[0] + '.brush')[0]
            pm.setAttr(brush + '.color1', color)
            pm.setAttr(brush + '.globalScale', thickness)
            pm.setAttr(brush + '.endCaps', 1)
            pm.setAttr(brush + '.tubeSections', 20)
            mel.eval('doPaintEffectsToPoly(1,0,0,1,100000);')
            mpoints += [stroke, curve]

    return pm.group(mpoints, n='AnimationPositions'), mpoints


def load_from_maya(root, start, end):
    import pymel.core as pm

    def rig_joints_list(s, js):
        for c in s.getChildren():
            if 'Geo' in c.name(): continue
            if isinstance(c, pm.nodetypes.Joint):     js = rig_joints_list(c, js); continue
            if isinstance(c, pm.nodetypes.Transform): js = rig_joints_list(c, js); continue
        return [s] + js

    joints = rig_joints_list(root, [])

    names = map(lambda j: j.name(), joints)
    positions = np.empty((end - start, len(names), 3))

    original_time = pm.currentTime(q=True)
    pm.currentTime(start)

    for i in range(start, end):

        pm.currentTime(i)
        for j in joints: positions[i - start, names.index(j.name())] = j.getTranslation(space='world')

    pm.currentTime(original_time)

    return positions, names


def loop(positions, forward='z'):
    fid = 'xyz'.index(forward)

    data = positions.copy()
    trajectory = data[:, 0:1, fid].copy()

    data[:, :, fid] -= trajectory
    diff = data[0] - data[-1]
    data += np.linspace(
        0, 1, len(data))[:, np.newaxis, np.newaxis] * diff[np.newaxis]
    data[:, :, fid] += trajectory

    return data


def extend(positions, length, forward='z'):
    fid = 'xyz'.index(forward)

    data = positions.copy()

    while len(data) < length:
        next = positions[1:].copy()
        next[:, :, fid] += data[-1, 0, fid]
        data = np.concatenate([data, next], axis=0)

    return data[:length]


def redirect(positions, joint0, joint1, forward='z'):
    forwarddir = {
        'x': np.array([[[1, 0, 0]]]),
        'y': np.array([[[0, 1, 0]]]),
        'z': np.array([[[0, 0, 1]]]),
    }[forward]

    direction = (positions[:, joint0] - positions[:, joint1]).mean(axis=0)[np.newaxis, np.newaxis]
    direction = direction / np.sqrt(np.sum(direction ** 2))

    rotation = Quaternions.between(direction, forwarddir).constrained_y()

    return rotation * positions


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" Maya Functions """


def load_from_maya(root):
    """
    Load joint parents and names from maya

    Parameters
    ----------

    root : PyNode
        Root Maya Node

    Returns
    -------

    (names, parents) : ([str], (J) ndarray)
        List of joint names and array
        of indices representing the parent
        joint for each joint J.

        Joint index -1 is used to represent
        that there is no parent joint
    """

    import pymel.core as pm

    names = []
    parents = []

    def unload_joint(j, parents, par):
        id = len(names)
        names.append(j)
        parents.append(par)

        children = [c for c in j.getChildren() if
                    isinstance(c, pm.nt.Transform) and
                    not isinstance(c, pm.nt.Constraint) and
                    not any(pm.listRelatives(c, s=True)) and
                    (any(pm.listRelatives(c, ad=True, ap=False, type='joint')) or isinstance(c, pm.nt.Joint))]

        map(lambda c: unload_joint(c, parents, id), children)

    unload_joint(root, parents, -1)

    return (names, parents)


""" Family Functions """


def joints(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    joints : (J) ndarray
        Array of joint indices
    """
    return np.arange(len(parents), dtype=int)


def joints_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    joints : [ndarray]
        List of arrays of joint idices for
        each joint
    """
    return list(joints(parents)[:, np.newaxis])


def parents_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    parents : [ndarray]
        List of arrays of joint idices for
        the parents of each joint
    """
    return list(parents[:, np.newaxis])


def children_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    children : [ndarray]
        List of arrays of joint indices for
        the children of each joint
    """

    def joint_children(i):
        return [j for j, p in enumerate(parents) if p == i]

    return list(map(lambda j: np.array(joint_children(j)), joints(parents)))


def descendants_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    descendants : [ndarray]
        List of arrays of joint idices for
        the descendants of each joint
    """

    children = children_list(parents)

    def joint_descendants(i):
        return sum([joint_descendants(j) for j in children[i]], list(children[i]))

    return list(map(lambda j: np.array(joint_descendants(j)), joints(parents)))


def ancestors_list(parents):
    """
    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    ancestors : [ndarray]
        List of arrays of joint idices for
        the ancestors of each joint
    """

    decendants = descendants_list(parents)

    def joint_ancestors(i):
        return [j for j in joints(parents) if i in decendants[j]]

    return list(map(lambda j: np.array(joint_ancestors(j)), joints(parents)))


""" Mask Functions """


def mask(parents, filter):
    """
    Constructs a Mask for a give filter

    A mask is a (J, J) ndarray truth table for a given
    condition over J joints. For example there
    may be a mask specifying if a joint N is a
    child of another joint M.

    This could be constructed into a mask using
    `m = mask(parents, children_list)` and the condition
    of childhood tested using `m[N, M]`.

    Parameters
    ----------

    parents : (J) ndarray
        parents array

    filter : (J) ndarray -> [ndarray]
        function that outputs a list of arrays
        of joint indices for some condition

    Returns
    -------

    mask : (N, N) ndarray
        boolean truth table of given condition
    """
    m = np.zeros((len(parents), len(parents))).astype(bool)
    jnts = joints(parents)
    fltr = filter(parents)
    for i, f in enumerate(fltr): m[i, :] = np.any(jnts[:, np.newaxis] == f[np.newaxis, :], axis=1)
    return m


def joints_mask(parents): return np.eye(len(parents)).astype(bool)


def children_mask(parents): return mask(parents, children_list)


def parents_mask(parents): return mask(parents, parents_list)


def descendants_mask(parents): return mask(parents, descendants_list)


def ancestors_mask(parents): return mask(parents, ancestors_list)


""" Search Functions """


def joint_chain_ascend(parents, start, end):
    chain = []
    while start != end:
        chain.append(start)
        start = parents[start]
    chain.append(end)
    return np.array(chain, dtype=int)


""" Constraints """


def constraints(anim, **kwargs):
    """
    Constraint list for Animation

    This constraint list can be used in the
    VerletParticle solver to constrain
    a animation global joint positions.

    Parameters
    ----------

    anim : Animation
        Input animation

    masses : (F, J) ndarray
        Optional list of masses
        for joints J across frames F
        defaults to weighting by
        vertical height

    Returns
    -------

    constraints : [(int, int, (F, J) ndarray, (F, J) ndarray, (F, J) ndarray)]
        A list of constraints in the format:
        (Joint1, Joint2, Masses1, Masses2, Lengths)

    """

    masses = kwargs.pop('masses', None)

    children = children_list(anim.parents)
    constraints = []

    points_offsets = Animation.offsets_global(anim)
    points = Animation.positions_global(anim)

    if masses is None:
        masses = 1.0 / (0.1 + np.absolute(points_offsets[:, 1]))
        masses = masses[np.newaxis].repeat(len(anim), axis=0)

    for j in xrange(anim.shape[1]):

        """ Add constraints between all joints and their children """
        for c0 in children[j]:

            dists = np.sum((points[:, c0] - points[:, j]) ** 2.0, axis=1) ** 0.5
            constraints.append((c0, j, masses[:, c0], masses[:, j], dists))

            """ Add constraints between all children of joint """
            for c1 in children[j]:
                if c0 == c1: continue

                dists = np.sum((points[:, c0] - points[:, c1]) ** 2.0, axis=1) ** 0.5
                constraints.append((c0, c1, masses[:, c0], masses[:, c1], dists))

    return constraints


""" Graph Functions """


def graph(anim):
    """
    Generates a weighted adjacency matrix
    using local joint distances along
    the skeletal structure.

    Joints which are not connected
    are assigned the weight `0`.

    Joints which actually have zero distance
    between them, but are still connected, are
    perturbed by some minimal amount.

    The output of this routine can be used
    with the `scipy.sparse.csgraph`
    routines for graph analysis.

    Parameters
    ----------

    anim : Animation
        input animation

    Returns
    -------

    graph : (N, N) ndarray
        weight adjacency matrix using
        local distances along the
        skeletal structure from joint
        N to joint M. If joints are not
        directly connected are assigned
        the weight `0`.
    """

    graph = np.zeros(anim.shape[1], anim.shape[1])
    lengths = np.sum(anim.offsets ** 2.0, axis=1) ** 0.5 + 0.001

    for i, p in enumerate(anim.parents):
        if p == -1: continue
        graph[i, p] = lengths[p]
        graph[p, i] = lengths[p]

    return graph


def distances(anim):
    """
    Generates a distance matrix for
    pairwise joint distances along
    the skeletal structure

    Parameters
    ----------

    anim : Animation
        input animation

    Returns
    -------

    distances : (N, N) ndarray
        array of pairwise distances
        along skeletal structure
        from some joint N to some
        joint M
    """

    distances = np.zeros((anim.shape[1], anim.shape[1]))
    generated = distances.copy().astype(bool)

    joint_lengths = np.sum(anim.offsets ** 2.0, axis=1) ** 0.5
    joint_children = children_list(anim)
    joint_parents = parents_list(anim)

    def find_distance(distances, generated, prev, i, j):

        """ If root, identity, or already generated, return """
        if j == -1: return (0.0, True)
        if j == i: return (0.0, True)
        if generated[i, j]: return (distances[i, j], True)

        """ Find best distances along parents and children """
        par_dists = [(joint_lengths[j], find_distance(distances, generated, j, i, p)) for p in joint_parents[j] if
                     p != prev]
        out_dists = [(joint_lengths[c], find_distance(distances, generated, j, i, c)) for c in joint_children[j] if
                     c != prev]

        """ Check valid distance and not dead end """
        par_dists = [a + d for (a, (d, f)) in par_dists if f]
        out_dists = [a + d for (a, (d, f)) in out_dists if f]

        """ All dead ends """
        if (out_dists + par_dists) == []: return (0.0, False)

        """ Get minimum path """
        dist = min(out_dists + par_dists)
        distances[i, j] = dist;
        distances[j, i] = dist
        generated[i, j] = True;
        generated[j, i] = True

    for i in xrange(anim.shape[1]):
        for j in xrange(anim.shape[1]):
            find_distance(distances, generated, -1, i, j)

    return distances


def edges(parents):
    """
    Animation structure edges

    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    edges : (M, 2) ndarray
        array of pairs where each
        pair contains two indices of a joints
        which corrisponds to an edge in the
        joint structure going from parent to child.
    """

    return np.array(list(zip(parents, joints(parents)))[1:])


def incidence(parents):
    """
    Incidence Matrix

    Parameters
    ----------

    parents : (J) ndarray
        parents array

    Returns
    -------

    incidence : (N, M) ndarray

        Matrix of N joint positions by
        M edges which each entry is either
        1 or -1 and multiplication by the
        joint positions returns the an
        array of vectors along each edge
        of the structure
    """

    es = edges(parents)

    inc = np.zeros((len(parents) - 1, len(parents))).astype(np.int)
    for i, e in enumerate(es):
        inc[i, e[0]] = 1
        inc[i, e[1]] = -1

    return inc.T
