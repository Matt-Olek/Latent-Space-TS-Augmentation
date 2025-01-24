import torch

def eval_gmms(gmms, num_eval_samples, logs):
    
    # For each class (repreented by a GMM), generate num_eval_samples samples
    
    class_log_probs = []
    class_trusts = []
    
    for class_idx, gmm in enumerate(gmms):
        trust = []
        z_samples, _ = gmm.sample(num_eval_samples)
        
        # For every gmm, get the log likelihood of the samples
        log_probs = []
        
        for class_id, class_gmm in enumerate(gmms):
            log_probs.append(class_gmm.score_samples(z_samples))
            if class_id != class_idx:
                trust.append(class_gmm.score_samples(z_samples))

        log_probs = torch.stack(log_probs, dim=1)
        class_log_probs.append(log_probs)
        trust = 1 - torch.exp(torch.stack(trust, dim=1)).sum(dim=1).mean()
        class_trusts.append(trust)
        
    all_probs = torch.stack(class_log_probs, dim=0)
    class_trusts = torch.stack(class_trusts, dim=0)
    mean_trust = class_trusts.mean()
    
    logs["mean_trust"] = mean_trust.item()
    logs["class_trusts"] = class_trusts.cpu().numpy()
        
    return mean_trust, class_trusts, all_probs, logs