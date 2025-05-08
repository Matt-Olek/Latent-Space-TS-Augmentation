import torch
from time import sleep
from scipy.stats.distributions import chi2

def eval_gmms(gmms, num_eval_samples, logs, step, which="naive"):
    
    if which == "naive":
        return _naive_eval_gmms(gmms, num_eval_samples, logs, step)


def _naive_eval_gmms(gmms, num_eval_samples, logs, step):
    
    # For each class (repreented by a GMM), generate num_eval_samples samples
    
    class_log_probs = []
    class_trusts = []
    
    for class_idx, gmm in enumerate(gmms):
        trust = []
        z_samples, _ = gmm.sample(num_eval_samples)
        
        # For every gmm, get the log likelihood of the samples
        log_probs = []
        
        for class_id, class_gmm in enumerate(gmms):
            log_probs.append(torch.from_numpy(class_gmm.score_samples(z_samples)))

        log_probs = torch.stack(log_probs, dim=1)
        class_log_probs.append(log_probs)
        
        # For each class compute the proportion of samples that are more likely to belong to that class than another, for every other class
        proportions = (log_probs.max(dim=1).indices == class_idx).float()
        
        # Compute the trust as the proportion of samples that are more likely to belong to the class than any other
        trust = proportions.mean()
        class_trusts.append(trust)
        
    all_probs = torch.stack(class_log_probs, dim=0)
    class_trusts = torch.stack(class_trusts, dim=0)
    mean_trust = class_trusts.mean()
    
    logs[f"mean_trust_step_{step}"] = mean_trust.cpu().numpy()
    if not logs.get("class_trusts"):
        logs["class_trusts"] = [class_trusts.cpu().numpy()]
    else:
        logs["class_trusts"].append(class_trusts.cpu().numpy())
        
    return mean_trust, class_trusts, all_probs, logs