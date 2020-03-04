import subprocess
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Element Wise')
parser.add_argument('--batch_start', default='640', type=int, help='Sequence Length of Input')
parser.add_argument('--batch_stop', default='2560', type=int, help='Sequence Length of Input')
parser.add_argument('--batch_inc', default='640', type=int, help='Sequence Length of Input')
parser.add_argument('--elements', default='1024', type=str, help='Sequence Length of Input')
parser.add_argument('--trials', default='20', type=str, help='Number of Trials to Execute')
parser.add_argument('--warmup-trials', default='5', type=str, help='Warmup Trials to discard')
parser.add_argument('--type', default='shfl', type=str, help='Reduction Type')

args = parser.parse_args()

default_list = ['nvprof', '--print-gpu-trace', '--csv', '--log-file', 'prof_file.csv', 'python', 'reduction.py', '--trials', args.trials, '--warmup-trials', args.warmup_trials, '--elements', args.elements, '--type', args.type]

assert args.batch_stop % args.batch_inc == 0, "ERROR! Your batch is not divisible by your increment."
for batch in range(int(args.batch_start),int(args.batch_stop) + int(args.batch_inc), int(args.batch_inc)) :
    cmd_list = default_list.copy()
    cmd_list += ['--batches' , str(batch)]
    output = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    assert os.path.exists('prof_file.csv'), "ERROR: Run failed. No profiler output!"
    df = pd.read_csv('prof_file.csv', header=3)
    df = df[df['Name'].str.contains("my_reduction_kernel|my_reduction_shared_kernel", na=False)]
    mean_val = df[5:]['Duration'].astype(float).mean()
    total_elems = int(args.elements) * batch + batch
    total_bytes = total_elems * 2
    expected_val = total_bytes / (1024.0 * 796000000.0) * 1000000.0
    efficiency = expected_val / mean_val * 100.0
    print(">>>Size: {:.03f} MB Batches: {:3d} Elements: {:3d} Time: {:03f} us {:.01f} %EFF".format(total_bytes/1000000.0, batch, int(args.elements), mean_val, efficiency))
    os.remove("prof_file.csv")
