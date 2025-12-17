import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from torch import optim
from utils.schedule import CustomScheduler
import numpy as np
import os
from diffusion.resample import create_named_schedule_sampler
import functools
from model.cfg_sampler import ClassifierFreeSampleModel
from eval.evaluate_process import EvaluateProcess
from eval.eval_methods import calculate_frechet_distance, calculate_activation_statistics, calculate_diversity, calculate_multimodality
from data_loaders import split_read_loader
from model.ema import ModelEmaV2
from src_t2m.model_util import load_model_wo_clip, create_model_and_diffusion
import torch.distributed as dist
from dataset.smr.smr2eval_rep import smr2t2m
from dataset.JP.jp2eval_rep import jp2t2m
from utils.log import console_out
from datetime import datetime

class ModelTrainer(pl.LightningModule):
    def __init__(self, args, model, diffusion, target_dir) -> None:
        super(ModelTrainer, self).__init__()
        '''
        输出日志初始化
        '''
        self.save_dir = target_dir

        if self.global_rank == 0:
            currentDateAndTime = datetime.now()
            currentTime = currentDateAndTime.strftime("%H_%M_%D")
            currentTime = currentTime.replace("/", "_")
            logger = console_out(os.path.join(self.save_dir, "run_{}.log".format(currentTime)))
            logger.info(args)  
            self.cout = logger.info
        
        self.save_hyperparameters("args")
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.num_frames = args.nframes
        self.eval_during_training = args.eval_during_training
        self.log_step = args.log_interval
        self.num_steps = args.num_steps
        self.ema = args.ema
        self.rep = args.rep
        self.preprocess = None
        self.eval_mode = args.eval_mode
        if self.ema:
            self.ema_model = ModelEmaV2(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad = False

        self.joints_num = 22
        self.lr_anneal_steps = args.lr_anneal_steps
        self.optimizer_name = "Adamw"
        self.lr = args.lr
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        
        '''
        测试及验证用初始化
        '''
        if self.ema:
            eval_model = self.ema_model.module
        else:
            eval_model = self.model 
        if args.guidance_param != 1:
            self.eval_model = ClassifierFreeSampleModel(eval_model)   # wrapping model with the classifier-free sampler
        else:
            self.eval_model = eval_model

        self.count = 0
        self.recall = 0
        self.match = 0
        self.pred = []
        self.motion = []
        self.recall_real = 0
        self.match_real = 0
        self.multimodel = []
        

    def forward(self, prompt, num_repetitions=1, inpainting_mask=None):
        self.eval_model.eval()
        inpainting_mask = inpainting_mask.permute(0, 2, 1).unsqueeze(2)
        model_kwargs = {'y':{'text': prompt, "inpainting_mask":inpainting_mask, 'lengths':self.num_frames}}
        if self.args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(num_repetitions, device=self.device) * self.args.guidance_param

        sample_fn = self.diffusion.p_sample_loop
        sample = sample_fn(
            self.eval_model,
            (num_repetitions, self.eval_model.njoints, self.eval_model.nfeats, self.num_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
            eval_mask=None
        )
        sample = sample["output"]   #### [bs, njoints, nfeats, num_frames]
        sample = sample.permute(0, 3, 1, 2).float()
        sample = sample.squeeze(3) #### [bs, 196, 263] or [bs, 196, 23, 6]
        return sample
    
    def configure_optimizers(self):
        optimizer = None
        if self.optimizer_name == "Adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        elif self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9)
        else:
            raise ValueError("Not a support optimizer")

        if self.lr_anneal_steps != 0:
            scheduler = CustomScheduler(optimizer, self.log_step, self.lr_anneal_steps, eta_min=self.args.lr * 0.1)
        else:
            scheduler = None

        if optimizer and scheduler:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif optimizer:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        self.model.train()
        _, _, prompt, _, motion, m_length, mask, inpainting_mask = batch  

        inpainting_mask = inpainting_mask.unsqueeze(3).permute(0, 2, 3, 1)
        model_kwargs = {'y':{'text': prompt, 'lengths':m_length, 'mask':mask, "inpainting_mask":inpainting_mask}}
    
        t, weights = self.schedule_sampler.sample(motion.shape[0])
        t = t.to(self.device)
        weights = weights.to(self.device)
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            motion,  # [bs, ch, image_size, image_size]
            t,  # [bs](int) sampled timesteps
            model_kwargs=model_kwargs,
        )
        losses = compute_losses()     
        loss = (losses["loss"] * weights).mean()     
        self.log("train_loss", loss, on_step=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        res = super().on_train_batch_end(outputs, batch, batch_idx)

        if self.ema:
            self.ema_model.update(self.model)

        if self.global_step % self.log_step == 0 and self.global_rank == 0:
            if isinstance(outputs, list):
                curr_loss = sum([outputs[i]["loss"].detach().cpu().item() for i in range(len(outputs))]) / len(outputs)
            else:
                curr_loss = outputs["loss"].detach().cpu().item()
            self.cout("[step: %06d]: [loss: %.4f]"%(self.global_step, curr_loss))
        return res

    def on_train_epoch_start(self):
        res = super().on_train_epoch_start()
        if self.global_rank == 0:
            self.cout("begin to train a new epoch, steps process -> [%d / %d]"%(self.global_step, self.num_steps))
        return res

    def validation_step(self, batch, batch_idx, mode="wo_mm"):
        if self.eval_during_training:
            evaluator = EvaluateProcess(self.args.evaluate_base_dir, self.args.evaluator_model_path, self.device)
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, mask, inpainting_mask = batch
            motion = torch.from_numpy(self.preprocess(motion.cpu().numpy())).to(self.device)
            bs = motion.shape[0]
            motion = motion.to(self.device)
            if self.rep == "smr":       ##### ground truth 计算，对齐表征
                motion = smr2t2m(motion, self.joints_num)
                motion = torch.from_numpy(motion).to(self.device)
            elif self.rep == "JP":
                motion = jp2t2m(motion, self.joints_num)
                motion = torch.from_numpy(motion).to(self.device)     

            #################### eval process ####################
            if mode in ["mm_short", "full"]:    ############ 推理阶段，计算推理结果 
                motion_multimodality_batch = []
                for k in range(30):
                    temp_motion = self.forward(caption, bs, inpainting_mask=inpainting_mask)  #### [bs, 196, 263]
                    temp_motion = torch.from_numpy(self.preprocess(temp_motion.cpu().numpy())).to(self.device)
                    if self.rep == "smr":
                        temp_motion = smr2t2m(temp_motion, self.joints_num)
                        temp_motion = torch.from_numpy(temp_motion).to(self.device)   
                    elif self.rep == "JP":
                        temp_motion = jp2t2m(temp_motion, self.joints_num)
                        temp_motion = torch.from_numpy(temp_motion).to(self.device)                         
                    if k == 0:
                        predict_motion = temp_motion

                    '''
                    multimodality 是计算稳定性的指标，需要一条 Prompt 跑个 30次
                    其他指标不需要，所以一般不要开启 mm_short 测试模式
                    '''
                    result = evaluator.infer(temp_motion, motion, word_embeddings, pos_one_hots, sent_len, m_length)
                    temp_mm = result["em_pred_norm"]
                    temp_mm = torch.cat(temp_mm, dim=0).reshape(word_embeddings.shape[0], 1, -1)
                    motion_multimodality_batch.append(temp_mm)

                self.multimodel.append(torch.cat(motion_multimodality_batch, dim=1))
            else:
                predict_motion = self.forward(caption, bs, inpainting_mask=inpainting_mask)  #### [bs, 196, 263]
                predict_motion = torch.from_numpy(self.preprocess(predict_motion.cpu().numpy())).to(self.device)

                if self.rep == "smr":
                    predict_motion = smr2t2m(predict_motion, self.joints_num)
                    predict_motion = torch.from_numpy(predict_motion).to(self.device)
                elif self.rep == "JP":
                    predict_motion = jp2t2m(predict_motion, self.joints_num)
                    predict_motion = torch.from_numpy(predict_motion).to(self.device)

            result = evaluator.infer(predict_motion, motion, word_embeddings, pos_one_hots, sent_len, m_length)
            for i in range(len(result["em_norm"])):
                em_norm = result["em_norm"][i]
                em_pred_norm = result["em_pred_norm"][i]
                real_R = result["real_R"][i]
                real_match = result["real_match"][i]
                temp_R = result["temp_R"][i]
                temp_match = result["temp_match"][i]

                self.motion.append(em_norm)
                self.pred.append(em_pred_norm)
                self.count += em_norm.shape[0]
                self.match += temp_match
                self.recall += temp_R
                self.match_real += real_match
                self.recall_real += real_R
        else:
            super().validation_step()

    def on_validation_epoch_end(self, save_ckpt=True, mode="wo_mm", evaluate=False):
        if self.eval_during_training:
            os.makedirs("valid_temp", exist_ok=True)
            save_dict = {
                "count":self.count,
                "recall":np.array(self.recall),
                "match":self.match,
                "recall_real":np.array(self.recall_real),
                "match_real":self.match_real,
            }
            motion_annotation_np = torch.cat(self.motion, dim=0).float().cpu().numpy()
            motion_pred_np = torch.cat(self.pred, dim=0).float().cpu().numpy()
            save_dict["motion"] = motion_annotation_np
            save_dict["pred"] = motion_pred_np
            if mode in ["mm_short"]:  
                motion_multimodality = torch.cat(self.multimodel, dim=0).float().cpu().numpy()   
                save_dict["multimodel"] = motion_multimodality

            np.savez("valid_temp/%03d.npz"%(self.global_rank), **save_dict)  ###### 为了多卡数据同步，先把所有文件都存在一个临时文件夹，主要是为了多卡测试
            
            if self.args.ngpu * self.args.nhost > 1:  
                dist.barrier()
            else:
                pass
            
            if self.global_rank == 0:   #### 只在第一张卡合并数据
                motion_annotation_np = []
                motion_pred_np = []
                if mode in ["mm_short"]:
                    motion_multimodality = []
                count = 0
                recall = 0
                match = 0
                recall_real = 0
                match_real = 0

                rank = 0
                while True:             ############ 所有卡的数据读取完成结束
                    path = "valid_temp/%03d.npz"%(rank)
                    try:
                        curr_dict = np.load(path)
                        motion_annotation_np.append(curr_dict["motion"]) 
                        motion_pred_np.append(curr_dict["pred"])
                        if mode in ["mm_short"]:  
                            motion_multimodality.append(curr_dict["multimodel"])
                        count += curr_dict["count"]
                        recall += curr_dict["recall"]
                        match += curr_dict["match"]
                        recall_real += curr_dict["recall_real"]
                        match_real += curr_dict["match_real"]         
                    except Exception as e:      
                        files = os.listdir("valid_temp")
                        for f in files:
                            curr_path = os.path.join("valid_temp", f)
                            os.remove(curr_path)
                        break

                    rank += 1 
                
                output_dict = {}
                print("total evaluate number per time:%d"%(count))
                motion_annotation_np = np.concatenate(motion_annotation_np, axis=0)
                motion_pred_np = np.concatenate(motion_pred_np, axis=0)
                gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
                mu, cov= calculate_activation_statistics(motion_pred_np)

                try:
                    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
                except Exception as e:
                    print("error information: ", e)
                    fid = 100

                if count >= 100:
                    try:
                        diversity_real = calculate_diversity(motion_annotation_np, 300 if count > 300 else 100)
                    except Exception as e:
                        print("error information: ", e)
                        diversity_real = 100
                    try:
                        diversity = calculate_diversity(motion_pred_np, 300 if count > 300 else 100)
                    except Exception as e:
                        print("error information: ", e)
                        diversity = 100
                else:
                    diversity_real = 0
                    diversity = 0

                if mode in ["mm_short"]:
                    motion_multimodality = np.concatenate(motion_multimodality, axis=0)
                    multimodality = calculate_multimodality(motion_multimodality, 10)
                else:
                    multimodality = 0

                output_dict["Fid"] = fid
                output_dict["Diversity"] = diversity
                output_dict["MultiModel"] = multimodality
                output_dict["Real_Diversity"] = diversity_real

                self.cout("[fid] : [%.4f]"%fid,)
                self.cout("[Diversity | Real] : [%.4f | %.4f]"%(diversity, diversity_real),)
                self.cout("[MultiModel] : [%.4f]"%multimodality,)

                matching_score_pred = match / count
                matching_score_real = match_real / count
                R_precision = recall / count
                R_precision_real = recall_real / count
                top1, top2, top3 = R_precision.tolist()
                rtop1, rtop2, rtop3 = R_precision_real.tolist()

                output_dict["Top1"] = top1
                output_dict["Top2"] = top2
                output_dict["Top3"] = top3
                output_dict["MMDist"] = matching_score_pred
                output_dict["Real_Top1"] = rtop1
                output_dict["Real_Top2"] = rtop2
                output_dict["Real_Top3"] = rtop3
                output_dict["Real_MMDist"] = matching_score_real

                self.cout("[top1 | Real] : [%.4f | %.4f]"%(top1, rtop1))
                self.cout("[top2 | Real] : [%.4f | %.4f]"%(top2, rtop2))
                self.cout("[top3 | Real] : [%.4f | %.4f]"%(top3, rtop3))
                self.cout("[MMDist | Real] : [%.4f | %.4f]"%(matching_score_pred, matching_score_real))

                name = "S%06d_F%.4f_T%.4f.pth"%(self.global_step, fid, top1)
        
                if not evaluate: ####### evalute = True 是测试， evaluate = False 是 valid
                    self.log_dict(output_dict, on_epoch=True)

                ############################################## save ##############################
                if save_ckpt:
                    self.save_ckpt(name)
            else:
                output_dict = {}

            self.count = 0
            self.recall = 0
            self.match = 0
            self.recall_real = 0
            self.match_real = 0
            self.motion = []
            self.pred = []
            self.multimodel = []
            return output_dict
        else:
            name = "model_%06d.pth"%(self.global_step)
            if save_ckpt:
                self.save_ckpt(name)
            return {}
        
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, mode=self.eval_mode)

    def on_test_epoch_end(self):
        init = super().on_test_epoch_end()
        res_dict = self.on_validation_epoch_end(False, self.eval_mode, evaluate=True)
        if self.args.ngpu * self.args.nhost > 1:
            dist.barrier()
        else:
            pass
        if self.global_rank == 0:
            torch.save(res_dict, "valid_temp/test_stage.pth")
        return init 

    def save_ckpt(self, name="last.pth"):
        state_dict = self.model.state_dict()
        if self.ema:
            ema_dict = self.ema_model.module.state_dict()
        else:
            ema_dict = None
        # Do not save CLIP weights
        clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del state_dict[e]
            if ema_dict is not None:
                del ema_dict[e]
        
        optimizer = self.optimizers()

        save_dict = {
            "step": self.global_step,
            "model": state_dict,
            "opt": optimizer.state_dict()
        }

        if ema_dict is not None:
            save_dict["ema"] = ema_dict

        torch.save(save_dict, os.path.join(self.save_dir, name))


def train_model(args):
    save_dir = os.path.join(args.base_dir, args.task_name)
    os.makedirs(save_dir, exist_ok=True)
    model, diffusion = create_model_and_diffusion(args, args.model_mode)
    pl_model = ModelTrainer(args, model, diffusion, save_dir)
    if os.path.exists(args.ft_path):
        print("loading pre-trained model %s"%(args.ft_path))
        state_dict = torch.load(args.ft_path, map_location='cpu')
        load_model_wo_clip(pl_model.model, state_dict["model"])
        if args.ema:
            load_model_wo_clip(pl_model.ema_model.module, state_dict["ema"])

    train_loader = split_read_loader.DATALoader(args, args.batch_size, train=True)
    valid_loader = split_read_loader.DATALoader(args, args.eval_batch_size, train=False)
    pl_model.preprocess = valid_loader.dataset.inv_transform
    lr_monitor = LearningRateMonitor(logging_interval="step")
    eval_batchs = 1.0
    logger_pl = TensorBoardLogger(save_dir, "motionMonitor")
    # logger_pl = CSVLogger(save_dir)

    if args.mix_precision:
        if args.gpu == "A100" or args.gpu == "H20":
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    else:
        precision = "32-true"
    
    checkpoint_callback = ModelCheckpoint(      ##### 定期检查点
        monitor="train_loss",
        filename ="save_during_train",
        dirpath=save_dir,
        save_last=True,
        every_n_train_steps = args.log_interval
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    if os.path.exists(os.path.join(save_dir, "last.pth")):     ##### 是不是需要续上断开的训练, 默认用最后一次的, save_last = True 会额外保存一个last.pth
        last_path = os.path.join(save_dir, "last.pth")
    else:
        last_path = None

    enable_progress_bar = False
    num_sanity_val_steps = 0
    strategy = "auto"

    trainer = pl.Trainer(log_every_n_steps=args.log_interval, val_check_interval=args.save_interval, check_val_every_n_epoch=None,
                        max_steps=args.num_steps, limit_val_batches=eval_batchs, enable_checkpointing=True,
                        callbacks=[lr_monitor, checkpoint_callback], logger=logger_pl, precision=precision,  enable_progress_bar=enable_progress_bar, num_sanity_val_steps=num_sanity_val_steps, 
                        devices=args.ngpu, num_nodes=args.nhost, strategy=strategy)

    trainer.fit(pl_model, train_loader, valid_loader, ckpt_path=last_path)

def eval_model(args):
    save_dir = os.path.join(args.base_dir, args.task_name)
    os.makedirs(save_dir, exist_ok=True)

    model, diffusion = create_model_and_diffusion(args, args.model_mode)
    pl_model = ModelTrainer(args, model, diffusion, save_dir)
    pl_model.eval_during_training = True

    valid_loader = split_read_loader.DATALoader(args, args.eval_batch_size, train=False)
    pl_model.preprocess = valid_loader.dataset.inv_transform
    if not os.path.exists(args.ft_path):
        assert "Model Path does not exist !!!"
    state_dict = torch.load(args.ft_path, map_location='cpu')
    try:
        if args.ema:
            print("EMA Checkpoints Loading.")
            state_dict =  state_dict["ema"]
        else:
            print("Normal Checkpoints Loading.")
            state_dict =  state_dict["model"]
    except:
        print("Origin MDM Checkpoints Loading.")
        state_dict =  state_dict
    
    try:    # guidance = 1
        load_model_wo_clip(pl_model.eval_model, state_dict)
    except:     # guidance != 1
        load_model_wo_clip(pl_model.eval_model.model, state_dict)
        
    if args.eval_mode in ["wo_mm", "full"]:
        repeat_times = 10
        total = -1
    elif args.eval_mode == "mm_short":
        repeat_times = 5
        total = 500

    if total != -1:
        test_batches = total / args.ngpu / args.nhost / args.eval_batch_size
        if np.floor(test_batches) == test_batches:
            test_batches = int(test_batches)
        else:
            test_batches = int(test_batches) + 1
    else:
        test_batches = 1.0


    pl_logger = CSVLogger(save_dir)
    enable_progress_bar = False
    trainer = pl.Trainer(devices=args.ngpu, num_nodes=args.nhost, limit_test_batches=test_batches, logger=pl_logger, enable_progress_bar=enable_progress_bar)   
    Fid = []
    Diversity = []
    Top1 = []
    Top2 = []
    Top3 = []
    MMDist = []
    mm = []
    Real_Diversity = []
    Real_Top1 = []
    Real_Top2 = []
    Real_Top3 = []
    Real_MMDist = []

    for i in range(repeat_times):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>> test repeat [%02d | %02d] <<<<<<<<<<<<<<<<<<<<<<"%(i, repeat_times))
        trainer.test(pl_model, valid_loader, verbose=False)
        out_dict = torch.load("valid_temp/test_stage.pth")

        if "Fid" in out_dict and out_dict["Fid"] == 100:
            continue
        elif "Diversity" in out_dict and out_dict["Diversity"] == 100:
            continue
        elif "MultiModel" in out_dict and out_dict["MultiModel"] == 100:
            continue
        elif "Real_Diversity" in out_dict and out_dict["Real_Diversity"] == 100:
            continue

        if "Fid" in out_dict:
            Fid.append(out_dict["Fid"])
        if "Diversity" in out_dict:
            Diversity.append(out_dict["Diversity"])
        if "Top1" in out_dict:
            Top1.append(out_dict["Top1"])
        if "Top2" in out_dict:
            Top2.append(out_dict["Top2"])
        if "Top3" in out_dict:
            Top3.append(out_dict["Top3"])
        if "MMDist" in out_dict:
            MMDist.append(out_dict["MMDist"])
        if "MultiModel" in out_dict:
            mm.append(out_dict["MultiModel"])
        if "Real_Top1" in out_dict:
            Real_Top1.append(out_dict["Real_Top1"])
        if "Real_Top2" in out_dict:
            Real_Top2.append(out_dict["Real_Top2"])
        if "Real_Top3" in out_dict:
            Real_Top3.append(out_dict["Real_Top3"])
        if "Real_MMDist" in out_dict:
            Real_MMDist.append(out_dict["Real_MMDist"])
        if "Real_Diversity" in out_dict:
            Real_Diversity.append(out_dict["Real_Diversity"])

    print('final result:')
    if len(Fid) != 0:
        fid = np.array(Fid)
        print(f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Diversity) != 0:
        div = np.array(Diversity)
        print(f"Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Top1) != 0:
        top1 = np.array(Top1)
        print(f"Top1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Top2) != 0:
        top2 = np.array(Top2)
        print(f"Top2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Top3) != 0:
        top3 = np.array(Top3)
        print(f"Top3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_times):.3f}")
    if len(MMDist) != 0:
        matching = np.array(MMDist)
        print(f"matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_times):.3f}")
    if len(mm) != 0:
        mm = np.array(mm)
        print(f"MultiModel. {np.mean(mm):.3f}, conf. {np.std(mm)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Real_Diversity) != 0:
        Real_Diversity = np.array(Real_Diversity)
        print(f"Real_Diversity. {np.mean(Real_Diversity):.3f}, conf. {np.std(Real_Diversity)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Real_MMDist) != 0:
        Real_MMDist = np.array(Real_MMDist)
        print(f"Real_MMDist. {np.mean(Real_MMDist):.3f}, conf. {np.std(Real_MMDist)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Real_Top1) != 0:
        Real_Top1 = np.array(Real_Top1)
        print(f"Real_Top1. {np.mean(Real_Top1):.3f}, conf. {np.std(Real_Top1)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Real_Top2) != 0:
        Real_Top2 = np.array(Real_Top2)
        print(f"Real_Top2. {np.mean(Real_Top2):.3f}, conf. {np.std(Real_Top2)*1.96/np.sqrt(repeat_times):.3f}")
    if len(Real_Top3) != 0:
        Real_Top3 = np.array(Real_Top3)
        print(f"Real_Top3. {np.mean(Real_Top3):.3f}, conf. {np.std(Real_Top3)*1.96/np.sqrt(repeat_times):.3f}")

if __name__ == "__main__":
    from src_t2m.config import opt
    from utils.fixseed import fixseed

    fixseed(10)
    save_dir = os.path.join(opt.base_dir, opt.task_name)
    os.makedirs(save_dir, exist_ok=True)

    if opt.evaluate:
        eval_model(opt)
    else:
        train_model(opt)