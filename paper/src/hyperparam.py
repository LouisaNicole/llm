from constants import get_spaced_tbe, get_probed_res_prefix
import copy
from argparse import Namespace
from args import *

class Hyperparam:
    def __init__(self, 
                founda_model,  lr,batch_size,epoch,rank,alpha):
        self.founda_model = founda_model
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.rank = rank
        self.alpha = alpha

    def __str__(self):
        return f'{self.founda_model}-lr{self.lr}b{self.batch_size}e{self.epoch}r{self.rank}a{self.alpha}'


class Cidtfpr_Hyperparam(Hyperparam):
    def __init__(self, args):
        super().__init__(args.founda_model, args.lr, args.batch_size, args.epoch, args.rank, args.alpha)
        self.prob_noise = args.prob
        self.scale = args.scale
        self.has_finetune = args.has_finetune
        self.has_llm = args.has_llm
        self.has_resp = args.has_resp
        self.has_sca = args.has_sca

    def __str__(self):
        ret = super().__str__() + f'p{self.prob_noise}f{self.scale}' \
                f'hr{int(self.has_resp)}hf{int(self.has_finetune)}hl{int(self.has_llm)}'
        if not self.has_sca:
            ret += f'hc0'
        return ret

class Cidtfrr_Hyperparam(Hyperparam):
    def __init__(self, args):
        super().__init__(args.founda_model, args.lr, args.batch_size, args.epoch, args.rank, args.alpha)
        self.sigma = args.sigma
        self.prob_noise = args.prob
        self.scale = args.scale
        self.abs_time = args.abs_time
        self.has_timing = args.has_timing
        self.has_finetune = args.has_finetune
        self.has_llm = args.has_llm
        self.spaced_tbe = get_spaced_tbe(self.founda_model) 

    def __str__(self):
        return super().__str__() + f's{self.sigma}p{self.prob_noise}f{self.scale}at{int(self.abs_time)}' \
                f'ht{int(self.has_timing)}hf{int(self.has_finetune)}hl{int(self.has_llm)}'

def get_new_model_name(attackname, train_ds_collection_id, hyperparam, rr_model=None):
    if attackname == 'rr':
        return attackname + '-' + train_ds_collection_id + '-' + str(hyperparam)
    elif attackname == 'pr':
        return attackname + '-' + train_ds_collection_id + '-' + str(hyperparam)+ '-' + rr_model
    else:
        assert False

def get_map_new_model_name(founda_model, train_ds_collection_id,
                            batch_size,
                            rank,
                            alpha,
                            rr_epoch,
                            pr_epoch,
                            rr_lr,
                            pr_lr,
                            at,
                            hr,
                            hf,
                            rr_hl,
                            pr_hl,
                            ht,
                            sigma,
                            scale,
                            rr_prob,
                            pr_prob,
                            hc=1):
    map_new_model_name = {}
    rr_args = dict()
    rr_args['founda_model'] = founda_model
    rr_args['batch_size'] = batch_size
    rr_args['epoch'] = rr_epoch
    rr_args['rank'] = rank
    rr_args['alpha'] = float(alpha)
    rr_args['lr'] = float(rr_lr)

    rr_args['sigma'] = float(sigma)
    rr_args['prob'] = float(rr_prob)
    rr_args['scale'] = float(scale)
    rr_args['abs_time'] = at
    rr_args['has_resp'] = hr
    rr_args['has_finetune'] = hf
    rr_args['has_timing'] = ht
    rr_args['has_llm'] = rr_hl
    map_new_model_name['rr'] = get_new_model_name('rr', train_ds_collection_id, Cidtfrr_Hyperparam(Namespace(**rr_args)))

    pr_args = copy.deepcopy(rr_args)
    pr_args['epoch'] = pr_epoch
    pr_args['lr'] = float(pr_lr)
    pr_args['prob'] = float(pr_prob)
    pr_args['has_sca'] = hc
    pr_args['has_llm'] = pr_hl
    map_new_model_name['pr'] = get_new_model_name('pr', train_ds_collection_id, Cidtfpr_Hyperparam(Namespace(**pr_args)), map_new_model_name['rr'])
    return map_new_model_name

def get_new_model_name_args(args):
    parser = get_hyperparam_argparser()
    args = parser.parse_args(args)

    # echo new model name
    hyperparam = Cidtfrr_Hyperparam(args) if args.attack == 'rr' else Cidtfpr_Hyperparam(args)
    return get_new_model_name(args.attack, args.train_ds_collection_id, hyperparam, args.rr_model)

def get_probed_res_prefix_args(args):
    parser = get_hyperparam_argparser()
    args = parser.parse_args(args)

    return get_probed_res_prefix(args.device, args.operationsys, args.framework, args.hardware)

if __name__ == '__main__':
    import os
    print(get_new_model_name_args(os.argv))