import os.path as osp
import argparse
import numpy as np
import torch
import trimesh
from SMPLX import smplx
from SMPLX.smplx.joint_names import Body
from SMPLX.read_from_npy import npy2info, info2dict
from tqdm.auto import tqdm, trange
from pathlib import Path
import os
os.environ['PYOPENGL_PLATFORM'] = "egl"

def write_obj(
    model_folder,
    motion_file,
    output_folder,
    model_type="smplh",
    gender="neutral",
    num_betas=10,
    num_expression_coeffs=10,
    use_face_contour=False,
    device="cpu"
):
    output_folder = Path(output_folder)
    assert output_folder.exists()

    # open motion file
    motion = np.load(motion_file, allow_pickle=True)
    try:
        poses = motion["poses"]
        gender = str(motion.get("gender", "neutral"))
        trans = motion.get("trans", None)
        betas = motion.get("betas", np.zeros([poses.shape[0], 10]))
    except:
        poses, trans, gender, betas = npy2info(motion, 10)

    # don't know where this is documented but it's from this part of amass
    # https://github.com/nghorbani/amass/blob/master/src/amass/data/prepare_data.py#L39-L40
    # gdr2num = {'male':-1, 'neutral':0, 'female':1}
    # gdr2num_rev = {v:k for k,v in gdr2num.items()}

    model = smplx.create(model_folder, model_type=model_type,
                        gender=gender, use_face_contour=use_face_contour,
                        num_betas=num_betas,
                        num_expression_coeffs=num_expression_coeffs,
                        ext="npz", use_pca=False, batch_size=poses.shape[0])

    model = model.eval().to(device)
    inputs = info2dict(poses, trans, betas, model_type, device=device)
    output = model(**inputs)
    vertices = output.vertices.detach().cpu().numpy()

    for pose_idx in range(vertices.shape[0]):
        curr_vert = vertices[pose_idx]

        vertex_colors = np.ones([curr_vert.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        # process=False to avoid creating a new mesh
        tri_mesh = trimesh.Trimesh(
            curr_vert, model.faces, vertex_colors=vertex_colors, process=False
        )

        '''
        humanact12 smpl 转 smplx
        仅和 amass 格式对齐时使用
        '''
        if "humanact" in motion_file:
            transf = trimesh.transformations.rotation_matrix(np.radians(90), (1, 0, 0))
            tri_mesh.apply_transform(transf)
        ###################

        output_path = output_folder / "{0:04d}.obj".format(pose_idx)
        tri_mesh.export(str(output_path))

    del model 
    del motion




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-X Demo")

    parser.add_argument(
        "--model-folder", required=True, type=str, help="The path to the model folder"
    )
    parser.add_argument(
        "--motion-file",
        required=True,
        type=str,
        help="The path to the motion file to process",
    )
    parser.add_argument(
        "--output-folder", required=True, type=str, help="The path to the output folder"
    )
    parser.add_argument(
        "--model-type",
        default="smplh",
        type=str,
        choices=["smpl", "smplh", "smplx", "mano", "flame"],
        help="The type of model to load",
    )
    parser.add_argument(
        "--num-expression-coeffs",
        default=10,
        type=int,
        dest="num_expression_coeffs",
        help="Number of expression coefficients.",
    )
    parser.add_argument(
        "--ext", type=str, default="npz", help="Which extension to use for loading"
    )
    parser.add_argument(
        "--sample-expression",
        default=True,
        dest="sample_expression",
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Sample a random expression",
    )
    parser.add_argument(
        "--use-face-contour",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Compute the contour of the face",
    )

    args = parser.parse_args()

    def resolve(path):
        return osp.expanduser(osp.expandvars(path))

    model_folder = resolve(args.model_folder)
    motion_file = resolve(args.motion_file)
    output_folder = resolve(args.output_folder)
    model_type = args.model_type
    ext = args.ext
    num_expression_coeffs = args.num_expression_coeffs
    sample_expression = args.sample_expression

    write_obj(
        model_folder,
        motion_file,
        output_folder,
        model_type,
        ext=ext,
        sample_expression=sample_expression,
        use_face_contour=args.use_face_contour,
    )
