#!/usr/bin/env python3
import sys
import os
import statistics
import re
import numpy as np
import pandas as pd


# BASE_PATH     = 'results_v6_40GB'
BASE_PATH     = 'results'
BASE_PATH_REF = 'results_reference_ae'
CG_CRASH_LOGS_PATH = BASE_PATH_REF + '/table-2'
NVPROF_LOGS_PATH   = BASE_PATH_REF + '/table-4'
DATA_DIR      = "./data"

SCHED_LOG_SUF  = 'sched-log'
SCHED_STAT_SUF = 'sched-stats'
WRKLDR_LOG_SUF = 'workloader-log'
GPU_UTIL_SUF   = 'sched_gpu.csv'
FP32_UTIL_SUF   = 'sched_fp32_util.csv'
MEMCOPY_UTIL_SUF   = 'sched_mem_copy.csv'
SM_UTIL_SUF   = 'sched_sm_occupancy.csv'
UTIL_WORKLOAD_LIST = ['16_16jobs', '16_32jobs']
COMPUTE_ONLY_DATA_NAME = 'time_compute_only.csv'
MEMORY_REDUCTION_DATA_NAME = 'memory.csv'

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


def usage_and_exit():
    print()
    print('  Usage:')
    print('    {} <figure-or-table> [--ref]'.format(sys.argv[0]))
    print()  
    print('    If providing a figure, it must be one of:')
    print('      {figure-4, figure-5, figure-6, figure-7, figure-8}')
    print('    If providing a table, it must be one of:')
    print('      {table-2, table-3, table-4}')
    print()
    print('    The --ref switch is optional. It will use the reference logs,')
    print('    which are the same logs used in the paper. Note that table-2')
    print('    and table-4 always use the reference logs (i.e. the switch')
    print('    has no effect).')
    print()
    sys.exit(1)


def format_multistream_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'single-assignment.4', WRKLDR_LOG_SUF),
        # # 1 GPU
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'single-assignment.1', WRKLDR_LOG_SUF),
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'cg.2', WRKLDR_LOG_SUF),
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.3', WRKLDR_LOG_SUF),
        # # 2 GPU or 20GB
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'single-assignment.2', WRKLDR_LOG_SUF),
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'cg.4', WRKLDR_LOG_SUF),
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.6', WRKLDR_LOG_SUF),
        # # 3 GPU or 30GB
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'single-assignment.3', WRKLDR_LOG_SUF),
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'cg.6', WRKLDR_LOG_SUF),
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.9', WRKLDR_LOG_SUF),
        # 40GB
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'cg.8', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.12', WRKLDR_LOG_SUF),
    )

def format_multistream_ms_filenames(workload):
    # # 1 GPU
    # return ('{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.1', WRKLDR_LOG_SUF),
    #         '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.3', WRKLDR_LOG_SUF))
    # # 2 GPU
    # return ('{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.2', WRKLDR_LOG_SUF),
    #         '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.6', WRKLDR_LOG_SUF))
    # # 3 GPU
    # return ('{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.3', WRKLDR_LOG_SUF),
    #         '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.9', WRKLDR_LOG_SUF))
    # # 20GB
    # return ('{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.4', WRKLDR_LOG_SUF),
    #         '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.6', WRKLDR_LOG_SUF))
    # # 30GB
    # return ('{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.4', WRKLDR_LOG_SUF),
    #         '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.9', WRKLDR_LOG_SUF))
    # 40GB
    return ('{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.4', WRKLDR_LOG_SUF),
            '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.12', WRKLDR_LOG_SUF))

def format_multistream_ms_mem_filenames(workload):
    # # 1 GPU
    # return '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.5', WRKLDR_LOG_SUF)
    # 2 GPU or 20GB
    # return '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.10', WRKLDR_LOG_SUF)
    # 3 GPU or 30GB
    # return '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.15', WRKLDR_LOG_SUF)
    # 40GB
    return '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.20', WRKLDR_LOG_SUF)

def format_motivation_1_filenames(workload):
    return '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.4', WRKLDR_LOG_SUF)

def format_motivation_2_filenames(workload):
    return '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.2', WRKLDR_LOG_SUF)

def format_motivation_2_mem_filenames(workload):
    return '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.4', WRKLDR_LOG_SUF)

def format_rodinia_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'single-assignment.4', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'cg.8', WRKLDR_LOG_SUF),
        # '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.12', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.12', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, '_ms_MultiGPU.12', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, '_ms_mem_MultiGPU.20', WRKLDR_LOG_SUF),
    )


def format_rodinia_util_filenames(workload):
    return (
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'single-assignment.2', GPU_UTIL_SUF),
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'cg.10', GPU_UTIL_SUF),
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'mgb_basic.12', GPU_UTIL_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'MultiGPU.12', GPU_UTIL_SUF),
    )


def format_rodinia_alg_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb.12', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.12', WRKLDR_LOG_SUF),
    )


def format_rodinia_nvprof_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(NVPROF_LOGS_PATH, workload, 'single-assignment.2', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(NVPROF_LOGS_PATH, workload, 'mgb.12', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(NVPROF_LOGS_PATH, workload, 'mgb_basic.12', WRKLDR_LOG_SUF),
    )


def format_darknet_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'zero.8', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.8', WRKLDR_LOG_SUF),
    )


def format_darknet_util_filenames(workload):
    return (
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'zero.8', GPU_UTIL_SUF),
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'mgb_basic.8', GPU_UTIL_SUF),
    )


def format_crash_filename(workload):
    return '{}/{}.{}'.format(CG_CRASH_LOGS_PATH, workload, WRKLDR_LOG_SUF)


def convert_time_to_ms(t):
    m = re.match(r"([0-9]*\.*[0-9]*)([a-z]*)", t)
    if m:
        v = float(m.group(1))
        if   'ns' == m.group(2):
            v /= 1000000
        elif 'us' == m.group(2):
            v /= 1000
        elif 'ms' == m.group(2):
            pass
        elif 's' == m.group(2):
            v *= 1000
        else:
            print('Unexpected time: {}'.format(t))
            sys.exit(1)
    else:
        print('Unexpected time: {}'.format(t))
        sys.exit(1)
    return v


def calc_average_kernel_times(cmd_to_invocations, avgs, which):
    for cmd, invocations in cmd_to_invocations.items():
        for invocation in invocations:
            for kernel_name, t in invocation.items():
                k = cmd + kernel_name
                if k not in avgs[which]:
                    avgs[which][k] = 0
                avgs[which][k] += t
        avgs[which][k] = 1.0 * avgs[which][k] / len(invocations)

def post_process_motivation_1():
    print()
    print('Post-process Motivation 1')
    print()
    multiGPU_throughputs = []
    multiGPU_ms_throughputs = []
    for workload in workloads_motivation_1:
        multiGPU_filename = format_motivation_1_filenames(workload)

        multiGPU_throughput, _ = parse_workloader_log(multiGPU_filename)

        multiGPU_throughputs.append(multiGPU_throughput)
    
    for workload in workloads_motivation_1_ms:
        multiGPU_ms_filename = format_motivation_1_filenames(workload)

        multiGPU_ms_throughput, _ = parse_workloader_log(multiGPU_ms_filename)

        multiGPU_ms_throughputs.append(multiGPU_ms_throughput)
    
    multiGPU_throughput_improvements = [ t[1] / t[0] for t in zip(multiGPU_throughputs, multiGPU_ms_throughputs) ]
    avg_multiGPU_throughput_improvement = statistics.mean(multiGPU_throughput_improvements)
    
    print('single stream throughput: ', multiGPU_throughputs)
    print('multi stream throughput: ', multiGPU_ms_throughputs)
    print('THROUGHPUT IMPROVEMENT: ', avg_multiGPU_throughput_improvement)

def post_process_motivation_2():
    print()
    print('Post-process Motivation 2')
    print()
    multiGPU_throughputs = []
    multiGPU_ms_throughputs = []
    for workload in workloads_motivation_2_ms:
        multiGPU_filename = format_motivation_2_filenames(workload)

        multiGPU_throughput, _ = parse_workloader_log(multiGPU_filename)

        multiGPU_throughputs.append(multiGPU_throughput)
    
    for workload in workloads_motivation_2_ms_mem:
        multiGPU_ms_filename = format_motivation_2_mem_filenames(workload)

        multiGPU_ms_throughput, _ = parse_workloader_log(multiGPU_ms_filename)

        multiGPU_ms_throughputs.append(multiGPU_ms_throughput)
    
    multiGPU_throughput_improvements = [ t[1] / t[0] for t in zip(multiGPU_throughputs, multiGPU_ms_throughputs) ]
    avg_multiGPU_throughput_improvement = statistics.mean(multiGPU_throughput_improvements)
    
    print('multi stream throughput: ', multiGPU_throughputs)
    print('multi stream with memory schedule throughput: ', multiGPU_ms_throughputs)
    print('THROUGHPUT IMPROVEMENT: ', avg_multiGPU_throughput_improvement)

def post_process_figure_1():
    print()
    print('Post-process Figure 1')
    print()
    sa_throughputs  = []
    sa_ms_throughputs  = []
    mgb_throughputs = []
    ms_throughputs = []
    ms_mem_throughputs = []
    for workload in workloads_multistream:
        sa_filename, _, mgb_filename = format_multistream_filenames(workload)

        sa_throughput, _   = parse_workloader_log(sa_filename)
        mgb_throughput, _  = parse_workloader_log(mgb_filename)

        sa_throughputs.append(sa_throughput)
        mgb_throughputs.append(mgb_throughput)

    for workload in workloads_multistream_ms:
        sa_ms_filename, ms_filename = format_multistream_ms_filenames(workload)
        sa_ms_throughput, _   = parse_workloader_log(sa_ms_filename)
        ms_throughput, _ = parse_workloader_log(ms_filename)
        sa_ms_throughputs.append(sa_ms_throughput)
        ms_throughputs.append(ms_throughput)
    
    for workload in workloads_multistream_ms_mem:
        ms_mem_filename = format_multistream_ms_mem_filenames(workload)
        ms_mem_throughput, _ = parse_workloader_log(ms_mem_filename)
        ms_mem_throughputs.append(ms_mem_throughput)

    sa_throughput_improvements  = [ 1 for _ in sa_throughputs ]
    sa_ms_throughput_improvements  = [ t[1] / t[0] for t in zip(sa_throughputs, sa_ms_throughputs) ]
    mgb_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, mgb_throughputs) ]
    ms_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, ms_throughputs) ]
    ms_mem_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, ms_mem_throughputs) ]
    avg_sa_ms_throughput_improvement  = statistics.mean(sa_ms_throughput_improvements)
    avg_mgb_throughput_improvement = statistics.mean(mgb_throughput_improvements)
    avg_ms_throughput_improvement = statistics.mean(ms_throughput_improvements)
    avg_ms_mem_throughput_improvement = statistics.mean(ms_mem_throughput_improvements)
    
    print('sa throughput: ', sa_throughputs)
    print('sa_ms throughput: ', sa_ms_throughputs)
    print('mgb throughput: ', mgb_throughputs)
    print('multiGPU_ms throughput: ', ms_throughputs)
    print('multiGPU_ms_mem throughput: ', ms_mem_throughputs)
    
    print('THROUGHPUT')
    print('. \t sa \t sa_ms \t case \tMultiGPU_ms\tMultiGPU_ms_mem')
    for idx, workload in enumerate(workloads_rodinia):
        name = rodinia_workloads_to_abbr_name[workload]
        print('{} \t {} \t {} \t {} \t {} \t\t {}'.format(name, round(sa_throughputs[idx],2), round(sa_ms_throughputs[idx],2),
                                  round(mgb_throughputs[idx],2),
                                  round(ms_throughputs[idx],2),
                                  round(ms_mem_throughputs[idx],2)))
    print('{} \t {} \t {} \t {} \t {} \t\t {}'.format('Avg.', round(statistics.mean(sa_throughputs),2), 
                                  round(statistics.mean(sa_ms_throughputs),2),
                                  round(statistics.mean(mgb_throughputs),2),
                                  round(statistics.mean(ms_throughputs),2),
                                  round(statistics.mean(ms_mem_throughputs),2)))

    print('THROUGHPUT IMPROVEMENT')
    print('. \t sa \t sa_ms \t case \tMultiGPU_ms\tMultiGPU_ms_mem')
    for idx, workload in enumerate(workloads_rodinia):
        name = rodinia_workloads_to_abbr_name[workload]
        print('{} \t {} \t {} \t {} \t {} \t\t {}'.format(name, 1, round(sa_ms_throughput_improvements[idx],2),
                                  round(mgb_throughput_improvements[idx],2),
                                  round(ms_throughput_improvements[idx],2),
                                  round(ms_mem_throughput_improvements[idx],2)))
    print('{} \t {} \t {} \t {} \t {} \t\t {}'.format('Avg.', 1, round(avg_sa_ms_throughput_improvement,2),
                                  round(avg_mgb_throughput_improvement,2),
                                  round(avg_ms_throughput_improvement,2),
                                  round(avg_ms_mem_throughput_improvement,2)))
    print()

    print('SUMMARY')
    print('Avg SA_MS Throughput Improvement: {}'.format(round(avg_sa_ms_throughput_improvement,2)))
    print('Avg CASE  Throughput Improvement: {}'.format(round(avg_mgb_throughput_improvement,2)))
    print('Avg MultiGPU_ms Throughput Improvement: {}'.format(round(avg_ms_throughput_improvement,2)))
    print('Avg MultiGPU_ms_mem Throughput Improvement: {}'.format(round(avg_ms_mem_throughput_improvement,2)))
    print()
    
    sa_throughput_improvements.append(1)
    sa_ms_throughput_improvements.append(avg_sa_ms_throughput_improvement)
    mgb_throughput_improvements.append(avg_mgb_throughput_improvement)
    ms_throughput_improvements.append(avg_ms_throughput_improvement)
    ms_mem_throughput_improvements.append(avg_ms_mem_throughput_improvement)
    index = [rodinia_workloads_to_abbr_name[workload] for workload in workloads_rodinia]
    index.append('Avg.')
    df = pd.DataFrame([sa_throughput_improvements, mgb_throughput_improvements, sa_ms_throughput_improvements, 
                       ms_throughput_improvements, ms_mem_throughput_improvements]).T
    df.columns = ['SA', 'CASE', 'SA_MS', 'MultiGPU_ms', 'MultiGPU_ms_mem']
    df.index = index
    csv_name = f"{DATA_DIR}/throughput.csv"
    df.to_csv(csv_name)
    print('Results saved to ', csv_name)

def get_util_average(filepath):
    df = pd.read_csv(filepath)
    avg_utilization_per_device = 0
    non_zero_count = 0
    min_begin = df['device_0'].size
    max_end = 0
    for column in ['device_0', 'device_1', 'device_2', 'device_3']:
        begin = 0
        end = df[column].size - 1
        for i in range(df[column].size):
            if df[column][i] != 0:
                begin = i
                break
        for i in range(df[column].size - 1, -1, -1):
            if df[column][i] != 0:
                end = i
                break
        if begin < min_begin:
            min_begin = begin
        if end > max_end:
            max_end = end
        if begin <= end:
            avg_utilization_per_device += df[column][begin:end].sum()
            non_zero_count += end - begin + 1
    return df[['device_0', 'device_1', 'device_2', 'device_3']][min_begin:max_end].mean(axis=None)

def post_process_figure_2():
    print()
    print('Post-process Figure 2')
    print()
    
    for workload in UTIL_WORKLOAD_LIST:
        filename = f"A100_{workload}"
        sa_filename, cg_filename, mgb_filename = format_multistream_filenames(filename)
        sa_ms_filename, ms_filename = format_multistream_ms_filenames(filename + '_ms')
        ms_mem_filename = format_multistream_ms_mem_filenames(filename + '_ms_mem')
        
        sa_utils = []
        sa_ms_utils = []
        cg_utils = []
        mgb_utils = []
        scheduler_utils = []
        ms_utils = []
        ms_mem_utils = []
        for suffix in [FP32_UTIL_SUF, MEMCOPY_UTIL_SUF, SM_UTIL_SUF]:
            sa_util_filename = sa_filename.replace('.' + WRKLDR_LOG_SUF, '-' + suffix)
            sa_ms_util_filename = sa_ms_filename.replace('.' + WRKLDR_LOG_SUF, '-' + suffix)
            cg_util_filename = cg_filename.replace('.' + WRKLDR_LOG_SUF, '-' + suffix)
            mgb_util_filename = mgb_filename.replace('.' + WRKLDR_LOG_SUF, '-' + suffix)
            ms_util_filename = ms_filename.replace('.' + WRKLDR_LOG_SUF, '-' + suffix)
            ms_mem_util_filename = ms_mem_filename.replace('.' + WRKLDR_LOG_SUF, '-' + suffix)

            sa_util = get_util_average(sa_util_filename)
            sa_ms_util = get_util_average(sa_ms_util_filename)
            # cg_util = get_util_average(cg_util_filename)
            mgb_util = get_util_average(mgb_util_filename)
            ms_util = get_util_average(ms_util_filename)
            ms_mem_util = get_util_average(ms_mem_util_filename)

            sa_utils.append(1)
            sa_ms_utils.append(sa_ms_util / sa_util)
            # cg_utils.append(cg_util / sa_util)
            mgb_utils.append(mgb_util / sa_util)
            ms_utils.append(ms_util / sa_util)
            ms_mem_utils.append(ms_mem_util / sa_util)

        df_utils = pd.DataFrame({
            'SA': sa_utils,
            'CASE': mgb_utils,
            'SA_MS': sa_ms_utils,
            'MultiGPU_ms': ms_utils,
            'MultiGPU_ms_mem': ms_mem_utils
        }, index=['FP32 Util.', 'Mem. Bw. Util.', 'SM Occupancy'])
        # df_utils = pd.DataFrame({
        #     'SA': sa_utils,
        #     'CG': cg_utils,
        #     'CASE': mgb_utils,
        #     'MultiGPU': ms_utils,
        #     'MultiGPU_mem': ms_mem_utils
        # }, index=['FP32 Util.', 'Mem. Bw. Util.', 'SM Occupancy'])

        csv_name = f"{DATA_DIR}/multiGPU_utilization_40GB_{workload}.csv"
        df_utils.to_csv(csv_name)
        print(df_utils)
        print('Utilization results saved to ', csv_name)

def post_process_compute_only():
    filename = os.path.join(DATA_DIR, COMPUTE_ONLY_DATA_NAME)
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    df = pd.read_csv(filename)
    df = df.set_index('Benchmark')

    methods = ['Taskflow Time(ms)', 'GrCUDA Time (ms)', 'MS Time (ms)']
    speedup_columns = []
    for method in methods:
        speedup_name = method.replace('Time', 'Speedup')
        speedup_columns.append(speedup_name)
        df[speedup_name] = df['Original Time (ms)'] / df[method]

    df = df[speedup_columns]
    avg_row = df.mean().to_frame().T
    avg_row.index = ['Average']
    df = pd.concat([df, avg_row]).round(2)
    print(df)
    return

def post_process_memory_reduction():
    filename = os.path.join(DATA_DIR, MEMORY_REDUCTION_DATA_NAME)
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    df = pd.read_csv(filename)
    df = df.set_index('Benchmark')

    methods = ['Max Memory (MB)']
    speedup_columns = []
    for method in methods:
        speedup_name = 'Reduction Ratio'
        speedup_columns.append(speedup_name)
        df[speedup_name] = (df['Original Max Memory (MB)'] - df[method]) / df['Original Max Memory (MB)'] * 100

    # df = df[speedup_columns]
    avg_row = df.mean().to_frame().T
    avg_row.index = ['Average']
    df = pd.concat([df, avg_row]).round(1)
    print(df)
    return

def parse_workloader_log(filename):
    #
    # Examples of what we're looking for:
    #
    # time since start:
    #   Worker 4: TIME_SINCE_START     4 9.485850095748901
    # bmark times:
    #   Worker 0: TOTAL_BENCHMARK_TIME 0 2.99345064163208
    #   Worker 1: TOTAL_BENCHMARK_TIME 2 1.2312407493591309
    # total time:
    #   Worker 0: TOTAL_EXPERIMENT_TIME 4.400961637496948
    #
    time_since_start = {}
    bmark_times = []
    total_time  = 0
    throughput = 0
    with open(filename) as f:
        for line in f:
            if 'TOTAL_BENCHMARK_TIME' in line:
                line = line.strip().split()
                bmark_times.append( (int(line[3]), float(line[4])) )
            elif 'TOTAL_EXPERIMENT_TIME' in line:
                total_time = float(line.strip().split()[3])
            elif 'TIME_SINCE_START' in line:
                time_since_start[float(line.strip().split()[3])] = float(line.strip().split()[4])

    throughput = len(bmark_times) / float(total_time)

    # grab just the sorted-by-worker time_since_start values
    # (k is the worker idx. v is its time since start)
    time_since_start = [v for k, v in sorted(time_since_start.items())]

    return throughput, time_since_start


def parse_utilization_log(filename):
    utilizations = []
    peak_utilization = 0
    with open(filename) as f:
        next(f) # skip header
        for line in f:
            line_vec = line.strip().split(',')
            timestamp = line_vec[0]
            device  = int(line_vec[1])
            device_1  = int(line_vec[2])
            device_2  = int(line_vec[3])
            device_3  = int(line_vec[4])
            utilization = device + device_1 + device_2 + device_3
            if utilization > peak_utilization:
                peak_utilization = utilization
            utilization = utilization / 4 / 100
            utilizations.append(utilization)
    peak_utilization = peak_utilization / 4 /100
    avg_utilization = statistics.mean(utilizations)
    return peak_utilization, avg_utilization


def parse_nvprof_log(workload_log, cmd_to_invocations):
    count = 0
    with open(workload_log) as f:
        inside_1 = False
        inside_2 = False
        cmd_str = ''
        for line in f:
            line = line.strip()

            # case: line begins a section of nvprof output
            if "NVPROF" in line:
                COMMAND_NEEDLE = ', command: '
                cmd_str = line[line.find(COMMAND_NEEDLE) + len(COMMAND_NEEDLE):]
                if cmd_str not in cmd_to_invocations:
                    cmd_to_invocations[cmd_str] = []
                invocation = {}
                assert inside_1 == False
                assert inside_2 == False
                inside_1 = True

            # case: we're inside a section of nvprof output.
            elif inside_1:
                assert inside_2 == False
                # case: we are starting to see the lines that we're interested
                # in. They start once we see 'GPU activities'.
                if 'GPU activities' in line:
                    line = line.split()
                    t = convert_time_to_ms(line[3])
                    kernel_name = ' '.join(line[8:])
                    invocation[kernel_name] = t
                    inside_1 = False
                    inside_2 = True
            elif inside_2:
                assert inside_1 == False
                # case: we were seeing lines we cared about, they end once we
                # see 'API calls'.
                if 'API calls' in line:
                    inside_2 = False
                    cmd_to_invocations[cmd_str].append(invocation)
                    cmd_str = ''
                    count += 1
                # case: we are still seeing lines we care about
                else:
                    line = line.split()
                    t = convert_time_to_ms(line[1])
                    kernel_name = ' '.join(line[6:])
                    invocation[kernel_name] = t
        assert inside_1 == False
        assert inside_2 == False

def post_process_single_stream():
    print()
    print('Post-process single stream')
    print()
    sa_throughputs  = []
    cg_throughputs  = []
    mgb_throughputs = []
    multiGPU_throughputs = []
    mem_throughputs = []
    for workload in workloads_multistream:
        sa_filename, cg_filename, mgb_filename = format_multistream_filenames(workload)

        sa_throughput, _   = parse_workloader_log(sa_filename)
        cg_throughput, _     = parse_workloader_log(cg_filename)
        mgb_throughput, _  = parse_workloader_log(mgb_filename)

        sa_throughputs.append(sa_throughput)
        cg_throughputs.append(cg_throughput)
        mgb_throughputs.append(mgb_throughput)

    for workload in workloads_multistream_ms:
        _, ms_filename = format_multistream_ms_filenames(workload)
        ms_throughput, _ = parse_workloader_log(ms_filename)
        multiGPU_throughputs.append(ms_throughput)
    
    for workload in workloads_multistream_ms_mem:
        ms_mem_filename = format_multistream_ms_mem_filenames(workload)
        ms_mem_throughput, _ = parse_workloader_log(ms_mem_filename)
        mem_throughputs.append(ms_mem_throughput)

    sa_throughput_improvements  = [ 1 for _ in sa_throughputs ]
    cg_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, cg_throughputs) ]
    mgb_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, mgb_throughputs) ]
    multiGPU_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, multiGPU_throughputs) ]
    mem_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, mem_throughputs) ]
    avg_cg_throughput_improvement  = statistics.mean(cg_throughput_improvements)
    avg_mgb_throughput_improvement = statistics.mean(mgb_throughput_improvements)
    avg_multiGPU_throughput_improvement = statistics.mean(multiGPU_throughput_improvements)
    avg_mem_throughput_improvement = statistics.mean(mem_throughput_improvements)
    
    print('sa throughput: ', sa_throughputs)
    print('cg throughput: ', cg_throughputs)
    print('mgb throughput: ', mgb_throughputs)
    print('multiGPU throughput: ', multiGPU_throughputs)
    print('mem throughput: ', mem_throughputs)
    
    print('THROUGHPUT')
    print('. \t sa \t cg \t case \tMultiGPU\tMultiGPU_mem')
    for idx, workload in enumerate(workloads_rodinia):
        name = rodinia_workloads_to_abbr_name[workload]
        print('{} \t {} \t {} \t {} \t {} \t\t {}'.format(name, round(sa_throughputs[idx],2), round(cg_throughputs[idx],2),
                                  round(mgb_throughputs[idx],2),
                                  round(multiGPU_throughputs[idx],2),
                                  round(mem_throughputs[idx],2)))
    print('{} \t {} \t {} \t {} \t {} \t\t {}'.format('Avg.', round(statistics.mean(sa_throughputs),2), 
                                  round(statistics.mean(cg_throughputs),2),
                                  round(statistics.mean(mgb_throughputs),2),
                                  round(statistics.mean(multiGPU_throughputs),2),
                                  round(statistics.mean(mem_throughputs),2)))

    print('THROUGHPUT IMPROVEMENT')
    print('. \t sa \t cg \t case \tMultiGPU\tMultiGPU_mem')
    for idx, workload in enumerate(workloads_rodinia):
        name = rodinia_workloads_to_abbr_name[workload]
        print('{} \t {} \t {} \t {} \t {} \t\t {}'.format(name, 1, round(cg_throughput_improvements[idx],2),
                                  round(mgb_throughput_improvements[idx],2),
                                  round(multiGPU_throughput_improvements[idx],2),
                                  round(mem_throughput_improvements[idx],2)))
    print('{} \t {} \t {} \t {} \t {} \t\t {}'.format('Avg.', 1, round(avg_cg_throughput_improvement,2),
                                  round(avg_mgb_throughput_improvement,2),
                                  round(avg_multiGPU_throughput_improvement,2),
                                  round(avg_mem_throughput_improvement,2)))
    print()

    print('SUMMARY')
    print('Avg CG Throughput Improvement: {}'.format(round(avg_cg_throughput_improvement,2)))
    print('Avg CASE Throughput Improvement: {}'.format(round(avg_mgb_throughput_improvement,2)))
    print('Avg MultiGPU Throughput Improvement: {}'.format(round(avg_multiGPU_throughput_improvement,2)))
    print('Avg MultiGPU_mem Throughput Improvement: {}'.format(round(avg_mem_throughput_improvement,2)))
    print()
    
    sa_throughput_improvements.append(1)
    cg_throughput_improvements.append(avg_cg_throughput_improvement)
    mgb_throughput_improvements.append(avg_mgb_throughput_improvement)
    multiGPU_throughput_improvements.append(avg_multiGPU_throughput_improvement)
    mem_throughput_improvements.append(avg_mem_throughput_improvement)
    index = [rodinia_workloads_to_abbr_name[workload] for workload in workloads_rodinia]
    index.append('Avg.')
    df = pd.DataFrame([sa_throughput_improvements, cg_throughput_improvements, mgb_throughput_improvements, 
                       multiGPU_throughput_improvements, mem_throughput_improvements]).T
    df.columns = ['SA', 'CG', 'CASE', 'MultiGPU', 'MultiGPU_mem']
    df.index = index
    csv_name = f"{DATA_DIR}/throughput_single_stream.csv"
    df.to_csv(csv_name)
    print('Results saved to ', csv_name)


post_process_funcs = {
    'motivation-1': post_process_motivation_1,
    'motivation-2': post_process_motivation_2,
    'figure-1': post_process_figure_1,
    'figure-2': post_process_figure_2,
    'compute-only': post_process_compute_only,
    'memory-reduction': post_process_memory_reduction,
    'single-stream': post_process_single_stream
}

workloads_motivation_1 = [
    'motivation-1'
]

workloads_motivation_1_ms = [
    'motivation-1-ms'
]

workloads_motivation_2_ms = [
    'motivation-2-ms'
]

workloads_motivation_2_ms_mem = [
    'motivation-2-ms-mem'
]

workloads_multistream = [
    'A100_50_16jobs',
    'A100_33_16jobs',
    'A100_25_16jobs',
    'A100_16_16jobs',
    'A100_50_32jobs',
    'A100_33_32jobs',
    'A100_25_32jobs',
    'A100_16_32jobs',
]

workloads_multistream_ms = [
    'A100_50_16jobs_ms',
    'A100_33_16jobs_ms',
    'A100_25_16jobs_ms',
    'A100_16_16jobs_ms',
    'A100_50_32jobs_ms',
    'A100_33_32jobs_ms',
    'A100_25_32jobs_ms',
    'A100_16_32jobs_ms',
]

workloads_multistream_ms_mem = [
    'A100_50_16jobs_ms_mem',
    'A100_33_16jobs_ms_mem',
    'A100_25_16jobs_ms_mem',
    'A100_16_16jobs_ms_mem',
    'A100_50_32jobs_ms_mem',
    'A100_33_32jobs_ms_mem',
    'A100_25_32jobs_ms_mem',
    'A100_16_32jobs_ms_mem',
]

workloads_rodinia = [
    'A100_50_16jobs',
    'A100_33_16jobs',
    'A100_25_16jobs',
    'A100_16_16jobs',
    'A100_50_32jobs',
    'A100_33_32jobs',
    'A100_25_32jobs',
    'A100_16_32jobs',
]

rodinia_workloads_to_name = {
    'A100_50_16jobs': '16-job-1:1-mix',
    'A100_33_16jobs': '16-job-2:1-mix',
    'A100_25_16jobs': '16-job-3:1-mix',
    'A100_16_16jobs': '16-job-5:1-mix',
    'A100_50_32jobs': '32-job-1:1-mix',
    'A100_33_32jobs': '32-job-2:1-mix',
    'A100_25_32jobs': '32-job-3:1-mix',
    'A100_16_32jobs': '32-job-5:1-mix',
}

rodinia_workloads_to_abbr_name = {
    'A100_50_16jobs': 'W1',
    'A100_33_16jobs': 'W2',
    'A100_25_16jobs': 'W3',
    'A100_16_16jobs': 'W4',
    'A100_50_32jobs': 'W5',
    'A100_33_32jobs': 'W6',
    'A100_25_32jobs': 'W7',
    'A100_16_32jobs': 'W8',
}

workloads_rodinia_nvprof = [
    'A100_50_16jobs_1',
    'A100_33_16jobs_1',
    'A100_25_16jobs_1',
    'A100_16_16jobs_1',
    'A100_50_32jobs_1',
    'A100_33_32jobs_1',
    'A100_25_32jobs_1',
    'A100_16_32jobs_1',
]

workloads_darknet = [
    'A100_2_8jobs',
    'A100_8jobs_1',
    'A100_8jobs_2',
    'A100_8jobs_3',
]

darknet_workload_to_nn_task = {
    'A100_2_8jobs': 'predict',
    'A100_8jobs_1': 'detect',
    'A100_8jobs_2': 'generate',
    'A100_8jobs_3': 'train',
}

workloads_crash = [
    'A100_16_16jobs.cg.10',
    'A100_16_16jobs.cg.12',
    'A100_16_16jobs.cg.6',
    'A100_16_16jobs.cg.8',
    'A100_16_32jobs.cg.10',
    'A100_16_32jobs.cg.12',
    'A100_16_32jobs.cg.6',
    'A100_16_32jobs.cg.8',
    'A100_25_16jobs.cg.10',
    'A100_25_16jobs.cg.12',
    'A100_25_16jobs.cg.6',
    'A100_25_16jobs.cg.8',
    'A100_25_32jobs.cg.10',
    'A100_25_32jobs.cg.12',
    'A100_25_32jobs.cg.6',
    'A100_25_32jobs.cg.8',
    'A100_33_16jobs.cg.10',
    'A100_33_16jobs.cg.12',
    'A100_33_16jobs.cg.6',
    'A100_33_16jobs.cg.8',
    'A100_33_32jobs.cg.10',
    'A100_33_32jobs.cg.12',
    'A100_33_32jobs.cg.6',
    'A100_33_32jobs.cg.8',
    'A100_50_16jobs.cg.10',
    'A100_50_16jobs.cg.12',
    'A100_50_16jobs.cg.6',
    'A100_50_16jobs.cg.8',
    'A100_50_32jobs.cg.10',
    'A100_50_32jobs.cg.12',
    'A100_50_32jobs.cg.6',
    'A100_50_32jobs.cg.8',
]


if len(sys.argv) != 2 and len(sys.argv) != 3:
    usage_and_exit()
if sys.argv[1] not in post_process_funcs:
    usage_and_exit()

if len(sys.argv) == 3:
    if sys.argv[2] != '--ref':
        usage_and_exit()
    BASE_PATH = BASE_PATH_REF + '/' + sys.argv[1]

post_process_funcs[sys.argv[1]]()
