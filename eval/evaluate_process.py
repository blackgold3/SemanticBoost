from eval.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval.eval_methods import calculate_R_precision
import numpy as np
import torch
import os

class EvaluateProcess(object):
    def __init__(self, evaluate_base_dir, evaluator_model_path, device="cuda") -> None:
        self.device = device
        self.eval_wrapper = EvaluatorMDMWrapper("humanml", device, evaluator_model_path)
        self.mean_eval = torch.from_numpy(np.load(os.path.join(evaluate_base_dir, "mean.npy")))
        self.std_eval = torch.from_numpy(np.load(os.path.join(evaluate_base_dir, "std.npy")))
        self.std_eval = self.std_eval.to(self.device)
        self.mean_eval = self.mean_eval.to(self.device)

    def infer(self, predict_motion, motion, word_embeddings, pos_one_hots, sent_len, m_length):
        #################### eval process ####################
        predict_motion = (predict_motion - self.mean_eval) / self.std_eval
        motion =  (motion - self.mean_eval) / self.std_eval
        total_batch = int(word_embeddings.shape[0] / 32)
        result = {"em_norm":[], "em_pred_norm":[], "real_R":[], "real_match":[], "temp_R":[], "temp_match":[]}
        for i in range(total_batch):
            begin = 32 * i
            end = begin + 32
            curr_sent_len = sent_len[begin:end]
            curr_sent_len, indices = torch.sort(curr_sent_len, descending=True)
            et, em = self.eval_wrapper.get_co_embeddings(word_embeddings[begin:end][indices], pos_one_hots[begin:end][indices], curr_sent_len, motion[begin:end][indices], m_length[begin:end][indices])
            et_pred, em_pred = self.eval_wrapper.get_co_embeddings(word_embeddings[begin:end][indices], pos_one_hots[begin:end][indices], curr_sent_len, predict_motion[begin:end][indices], m_length[begin:end][indices])
            et_norm, em_norm = et, em
            et_pred_norm, em_pred_norm = et_pred, em_pred
        
            mask_norm = torch.isnan(em_norm).to(self.device)    #### smr 转 t2m 可能导致奇怪的问题，出现 nan 的现象，几个维度，影响不大
            em_norm[mask_norm] = 0
            mask_pred = torch.isnan(em_pred_norm).to(self.device)
            em_pred_norm[mask_pred] = 0
            real_R, real_match = calculate_R_precision(et_norm.cpu().numpy(), em_norm.cpu().numpy(), top_k=3, sum_all=True)
            temp_R, temp_match = calculate_R_precision(et_pred_norm.cpu().numpy(), em_pred_norm.cpu().numpy(), top_k=3, sum_all=True)

            result["em_norm"].append(em_norm)
            result["em_pred_norm"].append(em_pred_norm)
            result["real_R"].append(real_R)
            result["real_match"].append(real_match)
            result["temp_R"].append(temp_R)
            result["temp_match"].append(temp_match)

        return result
