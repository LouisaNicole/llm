import argparse
def get_attack_argparser():
    parser = argparse.ArgumentParser(description="Attack")
    parser.add_argument('--local_rank', type=int, help='local rank passed from distributed launcher') # for deepspeed 
    parser.add_argument('--device', type=str, help="", required=True)
    parser.add_argument('--operationsys', type=str, help="", required=True)
    parser.add_argument('--framework', type=str, help="", required=True)
    parser.add_argument('--hardware', type=str, help="", required=True)
    parser.add_argument('--founda-model', type=str, help="", required=True)
    parser.add_argument('--train-ds-collection-id', type=str, required=True)
    parser.add_argument('--eval-ds-collection-id', type=str)
    parser.add_argument('--sigma', type=float, default=None) # for --attack=rr
    parser.add_argument('--prob', type=float, required=True)
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--abs-time',type=int, default=None) # for --attack=rr
    parser.add_argument('--has-resp',type=int, default=None) # for --attack=pr
    parser.add_argument('--has-timing',type=int, default=None) # for --attack=rr
    parser.add_argument('--has-finetune',type=int, required=True) # must be set for --mode=train
    parser.add_argument('--has-llm',type=int, required=True) # must be set for --mode=train
    parser.add_argument('--has-sca',type=int, default=1) 
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--validate', type=int, required=True)
    parser.add_argument('--prefill-bs', type=int, default=None)
    parser.add_argument('--embd-quant', type=str, default='F16')
    parser.add_argument('--victim-llm', type=int, default=None) # for --mode=eval
    parser.add_argument('--eval-steps', type=int, default=200)
    parser.add_argument('--will-test', type=int, default=1, help='Whether the GEN_BASE will be tested')
    parser.add_argument('--rr-model', type=str, default=None) # for --attack=pr
    parser.add_argument('--attack', type=str, choices=['pr', 'rr'], required=True)
    parser.add_argument('--mock-gpt', type=int, default=0)
    return parser

def get_hyperparam_argparser():
    parser = get_attack_argparser()
    for action in parser._actions:
        if action.dest in ['validate', 'mode']: 
            action.required = False
    return parser

def get_analyze_results_argparser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--local_rank', type=int, help='local rank passed from distributed launcher') # for deepspeed 
    
    parser.add_argument('--device', type=str, help="", required=True)
    parser.add_argument('--operationsys', type=str, help="", required=True)
    parser.add_argument('--framework', type=str, help="", required=True)
    parser.add_argument('--hardware', type=str, help="", required=True)
   
    parser.add_argument('--eval-ds-collection-id', type=str, required=True) 
    parser.add_argument('--new-model-name', type=str, required=True)
    parser.add_argument('--victim-llm', type=int, default=None)
    parser.add_argument('--validate', type=int, required=True)
    parser.add_argument('--prefill-bs', type=int, default=None)
    parser.add_argument('--embd-quant', type=str, default='F16')
    return parser
