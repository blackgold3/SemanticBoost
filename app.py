import os, sys
import gradio as gr
from huggingface_hub import snapshot_download
css = """
.dfile {height: 85px}
.ov {height: 185px}
"""


from huggingface_hub import snapshot_download
from motion.visual_api import Visualize  
import moviepy.editor as mpy
import torch
import json

with open("motion/path.json", "r") as f:
    json_dict = json.load(f)

def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

def prepare():
    if not os.path.exists("body_models") or not os.path.exists("weights"):
        REPO_ID = 'Kleinhe/CAMD'
        snapshot_download(repo_id=REPO_ID, local_dir='./', local_dir_use_symlinks=False)

    if not os.path.exists("tada-extend"):
        import subprocess
        import platform
        command = "bash scripts/tada_goole.sh"
        subprocess.call(command, shell=platform.system() != 'Windows')

def demo(prompt, mode, condition, render_mode="joints", skip_steps=0, out_size=1024, tada_role=None):
    prompt = prompt
    if prompt is None:
        prompt = ""
    
    path = None
    out_paths = [None, None, None]
    joints_paths = [None, None, None]
    smpl_paths = [None, None, None]

    if tada_role == "None":
        tada_role = None

    for i in range(len(mode)):
        kargs = {
            "mode":mode[i],
            "device":"cuda" if torch.cuda.is_available() else "cpu",
            "condition":condition,
            "smpl_path":json_dict["smpl_path"],
            "skip_steps":skip_steps,
            "path":json_dict,
            "tada_base":json_dict["tada_base"],
            "tada_role":tada_role
        }
        visual = Visualize(**kargs)
        render_mode = render_mode

        joint_path = "results/joints/{}_joint.npy".format(mode[i])
        smpl_path = "results/joints/{}_smpl.npy".format(mode[i])

        output = visual.predict(prompt, path, render_mode, joint_path, smpl_path)

        if render_mode == "joints":
            pics = visual.joints_process(output, prompt)
        elif render_mode.startswith("pyrender"):
            meshes, _ = visual.get_mesh(output)
            pics = visual.pyrender_process(meshes, out_size, out_size)
        
        out_path = "results/motion/temp{}.mp4".format(i)
        vid = mpy.ImageSequenceClip([x[:, :, :] for x in pics], fps=20)
        vid.write_videofile(out_path, remove_temp=True)

        if mode[i] == "cadm":
            out_paths[0] = out_path
            joints_paths[0] = joint_path
            smpl_paths[0] = smpl_path
        elif mode[i] == "cadm-augment":
            out_paths[1] = out_path
            joints_paths[1] = joint_path
            smpl_paths[1] = smpl_path
        elif mode[i] == "mdm":
            out_paths[2] = out_path
            joints_paths[2] = joint_path
            smpl_paths[2] = smpl_path
    
    return out_paths + joints_paths + smpl_paths
    

def t2m_demo():
    prepare()
    os.makedirs("results/motion", exist_ok=True)
    os.makedirs("results/joints", exist_ok=True)
    os.makedirs("results/smpls", exist_ok=True)

    tada_base = json_dict["tada_base"]
    files = os.listdir(os.path.join(tada_base, "MESH"))
    files = sorted(files)
    if files[0].startswith("."):
        files.pop(0)
    files = ["None"] +  files

    with gr.Blocks(analytics_enabled=False, css=css) as t2m_interface:
        gr.Markdown("<div align='center'> <h2> 🤷‍♂️ SemanticBoost: Elevating Motion Generation with Augmented Textual Cues </span> </h2> \
                    <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #000000' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #000000' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Row().style(equal_height=True):
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            with gr.Row():
                                demo_mode = gr.CheckboxGroup(choices=['cadm', 'cadm-augment','mdm'], default=["cadm"], label='Mode', info="Choose models to run demos, more models cost more time.")
                                skip_steps = gr.Number(value=0, label="Skip-Steps", info="The number of skip-steps during diffusion process (0 -> 999)", minimum=0, maximum=999, precision=0)

                            with gr.Row():
                                condition = gr.Radio(['text', 'uncond'], value='text', label='Condition', info="If sythesize motion with prompt?")
                                out_size = gr.Number(value=1024, label="Resolution", info="The resolution of output videos", minimum=224, maximum=2048, precision=0)

                            with gr.Row():
                                render_mode = gr.Radio(['joints','pyrender_fast', 'pyrender_slow'], value='joints', label='Render', info="If render results to 3D meshes? Pyrender need more time.")
                                tada_role = gr.Dropdown(files, value="None", multiselect=False, label="TADA Role", info="Choose 3D role to render")

                            with gr.Row():
                                prompt = gr.Textbox(value=None, placeholder="120,A person walks forward and does a handstand.", label="Prompt for Model -> (Length,Text)")

                submit = gr.Button('Visualize', variant='primary')

            with gr.Column(variant='panel'):      
                with gr.Tabs():
                    with gr.TabItem('Results'):
                        with gr.Row():
                            with gr.Column():
                                gen_video = gr.Video(label="CADM", format="mp4", autoplay=True, elem_classes="ov")
                            with gr.Column():
                                joint_file = gr.File(label="CADM-Joints", value=None, elem_classes="dfile")
                                smpl_file = gr.File(label="CADM-SMPL", value=None, elem_classes="dfile")

                        with gr.Row():
                            with gr.Column():
                                gen_video1 = gr.Video(label="CADM-Augment", format="mp4", autoplay=True, elem_classes="ov")
                            with gr.Column():
                                joint_file1 = gr.File(label="CADM-Augment-Joints", value=None, elem_classes="dfile")
                                smpl_file1 = gr.File(label="CADM-Augment-SMPL", value=None, elem_classes="dfile")
                                
                        with gr.Row():
                            with gr.Column():
                                gen_video2 = gr.Video(label="MDM", format="mp4", autoplay=True, elem_classes="ov")
                            with gr.Column():
                                joint_file2 = gr.File(label="MDM-Joints", value=None, elem_classes="dfile")
                                smpl_file2 = gr.File(label="MDM-SMPL", value=None, elem_classes="dfile")

                        
        submit.click(
                fn=demo,
                inputs=[prompt,
                        demo_mode,
                        condition,
                        render_mode,    
                        skip_steps,    
                        out_size,
                        tada_role               
                        ], 
                outputs=[gen_video, gen_video1, gen_video2, joint_file, joint_file1, joint_file2, smpl_file, smpl_file1, smpl_file2]
                )

    return t2m_interface
 

if __name__ == "__main__":
    demo = t2m_demo()
    demo.queue(max_size=10)
    demo.launch(debug=True)



