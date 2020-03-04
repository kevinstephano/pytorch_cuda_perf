import torch
import torch.nn.functional as F
import argparse

import reduction_ext

parser = argparse.ArgumentParser(description='Element Wise')
parser.add_argument('--batches', default='640', type=int, help='Batches of Input')
parser.add_argument('--elements', default='1024', type=int, help='Sequence Length of Input')
parser.add_argument('--trials', default=20, type=int, help='Number of Trials to Execute')
parser.add_argument('--warmup-trials', default=5, type=int, help='Warmup Trials to discard')
parser.add_argument('--type', default='shfl', type=str, help='Type of Reduction')

args = parser.parse_args()

if not torch.cuda.is_available():
    raise NotImplementedError('Running on CPU is not supported')
torch.cuda.set_device(0)

start_evt = []
stop_evt = []
for recorded_trial in range(0, args.trials) :
    start_evt.append(torch.cuda.Event(enable_timing=True))
    stop_evt.append(torch.cuda.Event(enable_timing=True))

inputs        = torch.ones(args.batches, args.elements, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(False)
#inputs        = torch.empty(args.batches, args.elements, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(False)
outputs       = torch.empty(args.batches, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(False)

for trial in range(0, args.trials + args.warmup_trials) :
    evt_idx       = trial - args.warmup_trials

    inputs2       = torch.ones(int(3145728 / args.elements), args.elements, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(False)
    if evt_idx >= 0 :
        start_evt[evt_idx].record()

    if args.type == 'shfl' :
        outputs = reduction_ext.exec(inputs, outputs)
    elif args.type == 'shared' :
        outputs = reduction_ext.exec_shared(inputs, outputs)
    else :
        assert False, "Bad Type: {}".format(args.type)

    if evt_idx >= 0 :
        stop_evt[evt_idx].record()
    #print(outputs)

torch.cuda.synchronize()
elapsed_time = 0.0
for evt_idx in range(0, args.trials) :
    my_time = start_evt[evt_idx].elapsed_time(stop_evt[evt_idx])
    elapsed_time += my_time
    #print("TIME: ", my_time)

print(">>>Size: {:.03f} MB Batches: {:3d} Elements: {:3d} Time: {:.3f} ms".format(
    args.batches*args.elements* 2 / 1000000,               \
    args.batches,                                          \
    args.elements,                                         \
    elapsed_time / args.trials))
