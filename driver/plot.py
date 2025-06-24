import sys
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math

DATA_DIR   = './data' 
FIG_DIR    = './figs'
RESULT_DIR = './results_motivation'
SCHED_LOG_SUF  = 'sched-log'
SCHED_STAT_SUF = 'sched-stats'
WRKLDR_LOG_SUF = 'workloader-log'
GPU_UTIL_SUF   = 'sched_gpu.csv'
SM_ACTIVE_SUF    = 'sched_sm_active.csv'
SM_OCCUPANCY_SUF = 'sched_sm_occupancy.csv'

SINGLE_COLUMN_FIG_WIDTH = 14.28
DOUBLE_COLUMN_FIG_WIDTH = 8.04

PROFILE_PREPARE_TIME = 3
PROFILE_END_TIME = 1.8

THROUGHPUT_DATA_LIST = ['throughput_v6_20GB.csv', 'throughput_v6_30GB.csv', 'throughput_v6_40GB.csv']
SINGLE_STREAM_DATA_LIST = ['throughput_4GPU_single_stream.csv']
DIFFERENT_NUM_GPU_DATA_LIST = ['throughput_1GPU.csv', 'throughput_2GPU.csv', 'throughput_3GPU.csv', 'throughput_4GPU.csv']
DIFFERENT_NUM_GPU_4090_DATA_LIST = ['throughput_1GPU_4090.csv', 'throughput_2GPU_4090.csv', 'throughput_3GPU_4090.csv', 'throughput_4GPU_4090.csv']
SINGLE_TASK_DATA_PATH = os.path.join(DATA_DIR, 'time_single_task.csv')
THROUGHPUT_FIG_PATH  = os.path.join(DATA_DIR, 'multiGPU_throughput.pdf')
MULTIGPU_UTIL_DATA_LIST = ['multiGPU_utilization_40GB_16_16jobs.csv', 'multiGPU_utilization_40GB_16_32jobs.csv']
DIFFERENT_STREAM_TIME_DATA = os.path.join(DATA_DIR, 'time_different_stream.csv')
DIFFERENT_STREAM_MEMORY_DATA = os.path.join(DATA_DIR, 'memory_different_stream.csv')

_liz_palette_kv = {
    "lancet": {
        "pink": "#ffc39f",
        "green": "#69c46f",
        "red": "#c91731",
        "blue": "#0b6c80",
        "gray": "#adb6b6",
        "black": "#000000",
        "purple": "#79459f"
    },
    "lancet_light": {
        "pink": "#f5e1c1",
        "green": "#8bdb92",
        "red": "#e05151",
        "blue": "#297787",
        "gray": "#adb6b6",
        "black": "#000000",
        "purple": "#9d6ebf"
    }
}


_liz_palette = {
   "lancet": np.array(["#fdaf91", "#74d17a", "#de1a39", "#0b6c80", "#adb6b6", "#000000", "#79459f"]),
   "lancet_light": np.array(['#f5e1c1', '#8bdb92', '#e05151', '#297787', '#adb6b6', '#000000', '#9d6ebf'])
}


lancet = _liz_palette['lancet']
lancet_light = _liz_palette['lancet_light']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['legend.framealpha'] = 0.7
scheduler_map = {'SA': 'SA', 
                 'CG': 'CG', 
                 'SA_MS': r'{\sc HuntK}',
                 'CASE': 'CASE', 
                 'Serial': 'Serial',
                 'Taskflow': 'Taskflow',
                 'GrSched': 'GrSched',
                 'MultiGPU_ms': r'{\sc HuntKT}', 
                 'MultiGPU_ms_mem': r'{\sc HuntKTm}',
                 'MultiGPU': r'{\sc HuntT}',
                 'MultiGPU_mem': r'{\sc HuntTm}'
                }

def register_cmap(name=None):
    if not hasattr(register_cmap, "registered"):
        setattr(register_cmap, "registered", False)
    color2rgb = mpl.colors.ColorConverter().to_rgb
    palettes_list = _liz_palette_kv.items()
    if name in _liz_palette_kv:
        palettes_list = [(name, _liz_palette_kv[name])]
    for name, colors in palettes_list:
        cmap = mpl.colors.ListedColormap(list(map(color2rgb, colors.values())), name=name)
        mpl.colormaps.register(cmap=cmap)
        print("register " + name)


def set_style():
    register_cmap()
    label_size = 22
    font_size = 16
    paper_rc = {'lines.linewidth': 5, 'lines.markersize': 14, 'axes.labelweight': 'bold',
                'axes.labelsize': label_size, 'axes.titlesize': label_size, 
                'font.size': font_size, 'legend.fontsize': 16, 
                'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
                'font.family': 'Arial'}
    sns.set_style('ticks')
    sns.set_context('poster', rc=paper_rc)
    sns.set_palette('lancet_light')


def lizify(ax: plt.Axes):
    ax.xaxis.get_label().set_fontweight('bold')
    ax.yaxis.get_label().set_fontweight('bold')
    ax.yaxis.grid(visible=True, which="minor", color="#eee", linewidth=1.5)
    ax.yaxis.grid(visible=True, which="major", color='#aaa')


def lineplot(data=None, *, x=None, y=None, 
             hue=None, hue_order=None, ax=None, markers=None, palette=None,
             linewidth=5, markersize=15, alpha=0.8, **kwargs):
    for mm, name in enumerate(hue_order):
        qiepian = data[data[hue] == name]
        if markers is None:
            mkr = 'X'
        else:
            mkr = markers[mm]
        if palette is not None:
            color = palette[mm]
        else:
            color = None
        sns.lineplot(qiepian, x=x, y=y, ax=ax, 
                    marker=mkr, label=name, color=color,
                    linewidth=linewidth, markersize=markersize, alpha=alpha, **kwargs)
    lizify(ax)


def barplot(data, *, x=None, y=None, hue=None, ax=None, hatches=None, palette=None,
            linewidth=1.5, alpha=1, saturation=1, edgecolor='k', **kwargs):
    sns.barplot(data, x=x, y=y, hue=hue, ax=ax, palette=palette, 
                linewidth=linewidth, alpha=alpha, saturation=saturation, edgecolor=edgecolor, **kwargs)
    if hatches is None:
        return
    if hatches is True:
        hatches = ['..', '||', '--', '//', '++', '\\\\', 'xx', '**']
    for hues, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legend_handles):
        handle.set_hatch(hatch)
        for hue in hues:
            hue.set_hatch(hatch)


def set_mean_bar_label(ax, fontsize=17, fontweight='bold', rotation=90, padding=10, **kwargs):
    for container in ax.containers:
        labels = ['%.2f' % v for v in container.datavalues]
        for i in range(len(container) - 1):
            labels[i] = ''
        ax.bar_label(container, labels=labels, fontsize=fontsize, fontweight=fontweight, 
                     rotation=rotation, padding=padding, **kwargs)


def trimm_mean(df, by, value, lo, hi, outelier_rate=1.5):
    fltr = df.groupby(by).apply(
        lambda x: x[(x[value] <= x[value].min() * outelier_rate)
                    & (x[value] >= x[value].quantile(lo)) 
                    & (x[value] <= x[value].quantile(hi))])
    fltr = fltr.reset_index(drop=True).groupby(by).mean().reset_index()
    return fltr

def usage_and_exit():
    print()
    print('Usage: python plot.py <metric>')
    print('metric: [throughput, GPU-util, mem-util]')

def plot_throughput():
    for filename in THROUGHPUT_DATA_LIST:
        filepath = os.path.join(DATA_DIR, filename)
        filename_without_extension = filename.rsplit(".", 1)[0]
        throughputs = pd.read_csv(filepath, index_col=0)
        groups = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'Avg.']
        data = {'Scheduler': [], 'Group': [], 'Values': []}
        
        print(throughputs)
        for row in throughputs:
            for idx, value in enumerate(throughputs[row]):
                data['Scheduler'].append(scheduler_map[row])
                data['Group'].append(groups[idx])
                data['Values'].append(value)

        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_FIG_WIDTH, 4))
        barplot(data=data, x='Group', y='Values', palette=lancet_light[:5].tolist(), hue='Scheduler', ax=ax)
        plt.xlabel('')
        plt.ylabel('Improvement')
        plt.ylim(ymin=0, ymax=3.5)
        plt.xticks(rotation=0)
        plt.legend(title='', ncol=3, loc='upper left')
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1)  # 添加红色虚线
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'multiGPU_{filename_without_extension}.pdf'), bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(FIG_DIR, f'multiGPU_{filename_without_extension}.png'), bbox_inches='tight', dpi=600)

def plot_gpu_util():
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(GPU_UTIL_SUF):
            print(filename)
            dir_name = filename.split('.')[0]
            if not os.path.exists(os.path.join(DATA_DIR, dir_name)):
                os.mkdir(os.path.join(DATA_DIR, dir_name))

            name = filename.rsplit(".", 1)[0]
            name = name.replace('v100', 'A100')
            gpu_util_path = os.path.join(RESULT_DIR, filename)
            output_path = os.path.join(DATA_DIR, dir_name, f"{name}_gpu.png")
            gpu_util = pd.read_csv(gpu_util_path, index_col=0)
            
            x = gpu_util.index
            data = {'x': [(t - x[0]) / 1e9 for t in x]}
            for col in gpu_util.columns:
                data[col] = gpu_util[col].values
                
            df = pd.DataFrame(data)
            df = df[(df['x'] >= PROFILE_PREPARE_TIME) & (df['x'] <= (df['x'].max() - PROFILE_END_TIME))]
            df['x'] = df['x'] - PROFILE_PREPARE_TIME
            df_melt = pd.melt(df, ['x'])
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x='x', y='value', hue='variable', data=df_melt, linewidth=2, markersize=1)
            plt.xlabel('Time (s)')
            plt.ylabel('Utilization (%)')
            plt.title(f'GPU Utilization of {name}')
            plt.xticks(rotation=0)
            plt.ylim(ymin=0, ymax=120)
            plt.xlim(xmin=0)
            ax.axhline(y=100, color='black', linestyle='--', linewidth=2)
            plt.legend(title='', ncol=2, loc='upper left')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

def plot_sm_active_util():
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(SM_ACTIVE_SUF):
            print(filename)
            dir_name = filename.split('.')[0]
            if not os.path.exists(os.path.join(DATA_DIR, dir_name)):
                os.mkdir(os.path.join(DATA_DIR, dir_name))

            name = filename.rsplit(".", 1)[0]
            name = name.replace('v100', 'A100')
            gpu_util_path = os.path.join(RESULT_DIR, filename)
            output_path = os.path.join(DATA_DIR, dir_name, f"{name}_sm_active.png")
            gpu_util = pd.read_csv(gpu_util_path, index_col=0)
            
            x = gpu_util.index
            data = {'x': [(t - x[0]) / 1e9 for t in x]}
            for col in gpu_util.columns:
                data[col] = gpu_util[col].values

            df = pd.DataFrame(data)
            df = df[(df['x'] >= PROFILE_PREPARE_TIME) & (df['x'] <= (df['x'].max() - PROFILE_END_TIME))]
            df['x'] = df['x'] - PROFILE_PREPARE_TIME
            df_melt = pd.melt(df, ['x'])
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x='x', y='value', hue='variable', data=df_melt, linewidth=2, markersize=1)
            plt.xlabel('Time (s)')
            plt.ylabel('Utilization (%)')
            plt.title(f'SM Activation of {name}')
            plt.xticks(rotation=0)
            plt.ylim(ymin=0, ymax=120)
            plt.xlim(xmin=0)
            ax.axhline(y=100, color='black', linestyle='--', linewidth=2)
            plt.legend(title='', ncol=2, loc='upper left')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

def plot_sm_occupancy_util():
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(SM_OCCUPANCY_SUF):
            print(filename)
            dir_name = filename.split('.')[0]
            if not os.path.exists(os.path.join(DATA_DIR, dir_name)):
                os.mkdir(os.path.join(DATA_DIR, dir_name))

            name = filename.rsplit(".", 1)[0]
            name = name.replace('v100', 'A100')
            gpu_util_path = os.path.join(RESULT_DIR, filename)
            output_path = os.path.join(DATA_DIR, dir_name, f"{name}_sm_occupancy.png")
            gpu_util = pd.read_csv(gpu_util_path, index_col=0)
            
            x = gpu_util.index
            data = {'x': [(t - x[0]) / 1e9 for t in x]}
            for col in gpu_util.columns:
                data[col] = gpu_util[col].values
                
            df = pd.DataFrame(data)
            df = df[(df['x'] >= PROFILE_PREPARE_TIME) & (df['x'] <= (df['x'].max() - PROFILE_END_TIME))]
            df['x'] = df['x'] - PROFILE_PREPARE_TIME
            df_melt = pd.melt(df, ['x'])
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x='x', y='value', hue='variable', data=df_melt, linewidth=2, markersize=1)
            plt.xlabel('Time (s)')
            plt.ylabel('Utilization (%)')
            plt.title(f'SM Occupancy of {name}')
            plt.xticks(rotation=0)
            plt.ylim(ymin=0, ymax=120)
            plt.xlim(xmin=0)
            ax.axhline(y=100, color='black', linestyle='--', linewidth=2)
            plt.legend(title='', ncol=2, loc='upper left')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

def plot_mem_util():
    pass

def plot_motivation_1():
    motivation_1_files = ['motivation-1-ms.MultiGPU.1-sched_sm_occupancy.csv', 'motivation-1.MultiGPU.2-sched_sm_occupancy.csv', 'motivation-1-ms.MultiGPU.2-sched_sm_occupancy.csv']
    device = 'device_1'
    for filename in motivation_1_files:
        if not os.path.exists(os.path.join(RESULT_DIR, filename)):
            print(f"File {filename} does not exist")
            assert False
            
    ms_path = os.path.join(RESULT_DIR, motivation_1_files[0])
    mt_path = os.path.join(RESULT_DIR, motivation_1_files[1])
    better_path = os.path.join(RESULT_DIR, motivation_1_files[2])
    ms = pd.read_csv(ms_path, index_col=0)
    mt = pd.read_csv(mt_path, index_col=0)
    better = pd.read_csv(better_path, index_col=0)

    ms_x = ms.index
    ms_data = {'x': [(t - ms_x[0]) / 1e9 for t in ms_x]}
    for col in ms.columns:
        ms_data[col] = ms[col].values
    ms_df = pd.DataFrame(ms_data)
    ms_df = ms_df[(ms_df['x'] >= PROFILE_PREPARE_TIME) & (ms_df['x'] <= (ms_df['x'].max() - PROFILE_END_TIME))]
    ms_df['x'] = ms_df['x'] - PROFILE_PREPARE_TIME
    ms_df_melt = pd.melt(ms_df, ['x'])
    ms_df_melt = ms_df_melt[ms_df_melt['variable'] == device]
    ms_df_melt['variable'] = 'multi-stream'
    ms_df_melt['value'] = ms_df_melt['value'] * 100
    
    mt_x = mt.index
    mt_data = {'x': [(t - mt_x[0]) / 1e9 for t in mt_x]}
    for col in mt.columns:
        mt_data[col] = mt[col].values
    mt_df = pd.DataFrame(mt_data)
    mt_df = mt_df[(mt_df['x'] >= PROFILE_PREPARE_TIME) & (mt_df['x'] <= (mt_df['x'].max() - PROFILE_END_TIME))]
    mt_df['x'] = mt_df['x'] - PROFILE_PREPARE_TIME
    mt_df_melt = pd.melt(mt_df, ['x'])
    mt_df_melt = mt_df_melt[mt_df_melt['variable'] == device]
    mt_df_melt['variable'] = 'multi-task'
    mt_df_melt['value'] = mt_df_melt['value'] * 100
    
    better_x = better.index
    better_data = {'x': [(t - better_x[0]) / 1e9 for t in better_x]}
    for col in better.columns:
        better_data[col] = better[col].values
    better_df = pd.DataFrame(better_data)
    better_df = better_df[(better_df['x'] >= PROFILE_PREPARE_TIME) & (better_df['x'] <= (better_df['x'].max() - PROFILE_END_TIME))]
    better_df['x'] = better_df['x'] - PROFILE_PREPARE_TIME
    better_df_melt = pd.melt(better_df, ['x'])
    better_df_melt = better_df_melt[better_df_melt['variable'] == device]
    better_df_melt['variable'] = 'hybrid'
    better_df_melt['value'] = better_df_melt['value'] * 100
    
    combined_df = pd.concat([ms_df_melt, mt_df_melt, better_df_melt], ignore_index=True)

    palette = np.concat([np.array(['#EECA40']), lancet_light[1:3]])
    hue_order = ['multi-stream', 'multi-task', 'hybrid']
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_FIG_WIDTH, 4))    
    lineplot(data=combined_df, x='x', y='value', hue='variable', hue_order=hue_order, 
            ax=ax, alpha=1, markers=['', '', ''], linewidth=2, markersize=10, palette=palette.tolist())
    ax.grid(False)
    plt.xlabel('Time (s)')
    plt.ylabel('SM Occupancy (\%)')
    plt.xticks(rotation=0)
    plt.ylim(ymin=0, ymax=40)
    plt.xlim(xmin=0)
    plt.legend(title='', ncol=1, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'motivation_1.png'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIG_DIR, 'motivation_1.pdf'), bbox_inches='tight', dpi=600)

def plot_motivation_2():
    motivation_2_files = ['motivation-2-ms.MultiGPU.2-sched_sm_occupancy.csv', 'motivation-2-ms-mem.MultiGPU.3-sched_sm_occupancy.csv']
    device = 'device_1'
    for filename in motivation_2_files:
        if not os.path.exists(os.path.join(RESULT_DIR, filename)):
            print(f"File {filename} does not exist")
            assert False
    
    baseline_path = os.path.join(RESULT_DIR, motivation_2_files[0])
    better_path = os.path.join(RESULT_DIR, motivation_2_files[1])
    baseline = pd.read_csv(baseline_path, index_col=0)
    better = pd.read_csv(better_path, index_col=0)
    
    baseline_x = baseline.index
    baseline_data = {'x': [(t - baseline_x[0]) / 1e9 for t in baseline_x]}
    for col in baseline.columns:
        baseline_data[col] = baseline[col].values
    baseline_df = pd.DataFrame(baseline_data)
    baseline_df = baseline_df[(baseline_df['x'] >= PROFILE_PREPARE_TIME) & (baseline_df['x'] <= (baseline_df['x'].max() - PROFILE_END_TIME))]
    baseline_df['x'] = baseline_df['x'] - PROFILE_PREPARE_TIME
    baseline_df_melt = pd.melt(baseline_df, ['x'])
    baseline_df_melt = baseline_df_melt[baseline_df_melt['variable'] == device]
    baseline_df_melt['variable'] = 'hybrid'
    baseline_df_melt['value'] = baseline_df_melt['value'] * 100
    
    better_x = better.index
    better_data = {'x': [(t - better_x[0]) / 1e9 for t in better_x]}
    for col in better.columns:
        better_data[col] = better[col].values
    better_df = pd.DataFrame(better_data)
    better_df = better_df[(better_df['x'] >= PROFILE_PREPARE_TIME) & (better_df['x'] <= (better_df['x'].max() - PROFILE_END_TIME))]
    better_df['x'] = better_df['x'] - PROFILE_PREPARE_TIME
    better_df_melt = pd.melt(better_df, ['x'])
    better_df_melt = better_df_melt[better_df_melt['variable'] == device]
    better_df_melt['variable'] = 'hybrid w/ mem.'
    better_df_melt['value'] = better_df_melt['value'] * 100
    
    combined_df = pd.concat([baseline_df_melt, better_df_melt], ignore_index=True)

    hue_order = ['hybrid', 'hybrid w/ mem.']
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_FIG_WIDTH, 4))    
    lineplot(data=combined_df, x='x', y='value', hue='variable', hue_order=hue_order, 
            ax=ax, alpha=1, markers=['', ''], linewidth=2, markersize=10, palette=lancet_light[2:4].tolist())
    ax.grid(False)
    plt.xlabel('Time (s)')
    plt.ylabel('SM Occupancy (\%)')
    plt.xticks(rotation=0)
    plt.ylim(ymin=0, ymax=40)
    plt.xlim(xmin=0)
    x_min = combined_df['x'].min()
    x_max = combined_df['x'].max()

    x_ticks = np.arange(0, math.floor(x_max) + 1, 3)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(tick)) for tick in x_ticks])
    plt.legend(title='', ncol=1, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'motivation_2.png'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIG_DIR, 'motivation_2.pdf'), bbox_inches='tight', dpi=600)


def plot_single_task():
    single_task_data = pd.read_csv(SINGLE_TASK_DATA_PATH)
    data = {
        'Benchmark': single_task_data['Benchmark'].tolist(),
        'Original Time (ms)': single_task_data['Original Time (ms)'].tolist(),
        'Taskflow Time(ms)': single_task_data['Taskflow Time(ms)'].tolist(),
        'MS Time (ms)': single_task_data['MS Time (ms)'].tolist(),
        'MS Mem Time (ms)': single_task_data['MS Mem Time (ms)'].tolist()
    }

    df = pd.DataFrame(data)
    df['Serial'] = df['Original Time (ms)'] / df['Original Time (ms)']
    df['Taskflow'] = df['Original Time (ms)'] / df['Taskflow Time(ms)']
    df['MultiGPU_ms'] = df['Original Time (ms)'] / df['MS Time (ms)']
    df['MultiGPU_ms_mem'] = df['Original Time (ms)'] / df['MS Mem Time (ms)']
    
    avg_baseline_speedup = 1
    avg_taskflow_speedup = df['Taskflow'].mean()
    avg_ms_speedup = df['MultiGPU_ms'].mean()
    avg_ms_mem_speedup = df['MultiGPU_ms_mem'].mean()
    avg_row = pd.DataFrame([['Average', np.nan, np.nan, np.nan, np.nan, avg_baseline_speedup, avg_taskflow_speedup, avg_ms_speedup, avg_ms_mem_speedup]], 
                           columns=['Benchmark', 'Original Time (ms)', 'Taskflow Time(ms)', 'MS Time (ms)', 'MS Mem Time (ms)', 'Serial', 'Taskflow', 'MultiGPU_ms', 'MultiGPU_ms_mem'])
    df = pd.concat([df, avg_row], ignore_index=True)
    df = df[['Serial', 'Taskflow', 'MultiGPU_ms', 'MultiGPU_ms_mem']]
    
    groups = ['VEC', 'B\&S', 'ML', 'IMG', 'DL', 'M1', 'M2', 'Avg.']
    data = {'Scheduler': [], 'Group': [], 'Values': []}
    print(df)
    for row in df:
        for idx, value in enumerate(df[row]):
            data['Scheduler'].append(scheduler_map[row])
            data['Group'].append(groups[idx])
            data['Values'].append(value)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_FIG_WIDTH, 4))
    plalette = np.concat([lancet_light[:2], lancet_light[3:5]])
    barplot(data=data, x='Group', y='Values', palette=plalette.tolist(), hue='Scheduler', ax=ax)
    plt.xlabel('')
    plt.ylabel('Speedup')
    # plt.ylim(ymin=0, ymax=3.5)
    plt.ylim(ymin=0, ymax=2)
    plt.xticks(rotation=0)
    plt.legend(title='', ncol=2, loc='upper left')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1)  # black dashed line
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'single_task.pdf'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIG_DIR, f'single_task.png'), bbox_inches='tight', dpi=600)

def plot_multigpu_util():
    all_data = []
    i = 0
    for filename in MULTIGPU_UTIL_DATA_LIST:
        filepath = os.path.join(DATA_DIR, filename)
        filename_without_extension = filename.rsplit(".", 1)[0]
        multigpu_util = pd.read_csv(filepath, index_col=0)
        groups = multigpu_util.index
        data = {'Scheduler': [], 'Group': [], 'Values': []}
        for row in multigpu_util:
            for idx, value in enumerate(multigpu_util[row]):
                data['Scheduler'].append(scheduler_map[row])
                if i == 1:
                    data['Group'].append(groups[idx])
                else:
                    data['Group'].append(' ' + groups[idx] + ' ')
                data['Values'].append(value)
        all_data.append(pd.DataFrame(data))
        i += 1

    combined_data = pd.concat(all_data)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_FIG_WIDTH, 4))
    barplot(data=combined_data, x='Group', y='Values', hue='Scheduler', ax=ax, palette=lancet_light[:5].tolist())

    num_groups = len(groups)
    ax.grid(False)

    x_ticks = ax.get_xticks()
    ax.text(num_groups / 2 - 0.5, -0.25, 'W4', ha='center', va='center', transform=ax.get_xaxis_transform())
    ax.text(num_groups + (num_groups / 2) - 0.5, -0.25, 'W8', ha='center', va='center', transform=ax.get_xaxis_transform())
    ax.plot([x_ticks[0] - 0.5, x_ticks[0] - 0.5], [0, -0.3], transform=ax.get_xaxis_transform(), clip_on=False,
        color='black', linestyle='-', linewidth=2)
    ax.plot([x_ticks[-1] + 0.5, x_ticks[-1] + 0.5], [0, -0.3], transform=ax.get_xaxis_transform(), clip_on=False,
            color='black', linestyle='-', linewidth=2)
    ax.plot([num_groups - 0.5, num_groups - 0.5], [0, -0.3], transform=ax.get_xaxis_transform(), clip_on=False,
            color='black', linestyle='-', linewidth=2)
    ax.set_xlim(x_ticks[0] - 0.5, x_ticks[-1] + 0.5)

    plt.ylabel('Improvement')
    plt.ylim(ymin=0, ymax=7.2)
    plt.xticks(range(6), rotation=0)
    plt.xlabel('')
    plt.legend(title='', ncol=3, loc='upper left')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1)  # black dashed line
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'combined_multigpu_util.pdf'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIG_DIR, 'combined_multigpu_util.png'), bbox_inches='tight', dpi=600)
    plt.show()

def plot_different_stream():
    indices = ['b5', 'b14', 'b15']
    time_data = pd.read_csv(DIFFERENT_STREAM_TIME_DATA, index_col=0, header=0)
    memory_data = pd.read_csv(DIFFERENT_STREAM_MEMORY_DATA, index_col=0, header=0)

    time_data.index.name = 'Benchmark'
    memory_data.index.name = 'Benchmark'

    for i in range(1, 11):
        time_data[f'MS_Mem_{i}'] = time_data['Original'] / time_data[f'MS_Mem_{i}']
        memory_data[f'{i}'] = memory_data[f'{i}'] / memory_data['Original']
    time_data['Original'] = 1
    time_data = time_data[['Original', 'MS_Mem_1', 'MS_Mem_2', 'MS_Mem_3', 'MS_Mem_4', 'MS_Mem_5', 'MS_Mem_6', 'MS_Mem_7', 'MS_Mem_8', 'MS_Mem_9', 'MS_Mem_10']]
    
    time_data = time_data.loc[indices]
    memory_data = memory_data.loc[indices]
    print(time_data)
    print(memory_data)
    groups = ['B\&S', 'M1', 'M2']
    data = {'Benchmark': [], 'Group': [], 'Speedup': [], 'Memory': []}
    benchmark_map = {'b5': 'B\&S', 'b14': 'M1', 'b15': 'M2'}
    for benchmark in time_data.index:
        for i in range(1, 11):
            data['Benchmark'].append(benchmark_map[benchmark])
            data['Group'].append(i)
            data['Speedup'].append(time_data.loc[benchmark, f'MS_Mem_{i}'])
            data['Memory'].append(memory_data.loc[benchmark, f'{i}'])
    data = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COLUMN_FIG_WIDTH, 5), gridspec_kw={'hspace': 0.5})

    # Plot speedup
    lineplot(data=data, x='Group', y='Speedup', hue='Benchmark', hue_order=groups, 
            ax=ax1, markers=['o', '^', 'X'], alpha=1, linewidth=3, markersize=10)
    ax1.grid(False)
    # ax1.set_xlabel('\# Streams')
    ax1.set_xlabel('')
    ax1.set_ylabel('Speedup')
    ax1.set_ylim(ymin=0, ymax=4.7)
    ax1.set_xticks(range(1, 11))
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax1.legend(loc='upper left', ncol=3, fontsize=14)
    ax1.text(0.5, -0.38, '(a) Speedup', ha='center', va='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')

    # Plot memory reduction ratio
    lineplot(data=data, x='Group', y='Memory', hue='Benchmark', hue_order=groups, 
            ax=ax2, markers=['o', '^', 'X'], alpha=1, linewidth=3, markersize=10)
    ax2.grid(False)
    # ax2.set_xlabel('\# Streams')
    ax2.set_xlabel('')
    ax2.set_ylabel('Memory Ratio')
    ax2.set_ylim(ymin=0, ymax=1.1)
    ax2.set_xticks(range(1, 11))
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax2.legend_.remove()
    ax2.text(0.5, -0.38, '(b) Memory Ratio', ha='center', va='center', transform=ax2.transAxes, fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'different_stream_combined.png'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIG_DIR, 'different_stream_combined.pdf'), bbox_inches='tight', dpi=600)
    plt.close()

def plot_different_memory():
    data = {'Scheduler': [], 'Group': [], 'Values': []}
    groups = ['20 GB', '30 GB', '40 GB']
    
    for idx, filename in enumerate(THROUGHPUT_DATA_LIST):
        filepath = os.path.join(DATA_DIR, filename)
        filename_without_extension = filename.rsplit(".", 1)[0]
        throughputs = pd.read_csv(filepath, index_col=0)
        
        avg_values = throughputs.loc['Avg.']
        for scheduler, value in avg_values.items():
            data['Scheduler'].append(scheduler_map[scheduler])
            data['Group'].append(groups[idx])
            data['Values'].append(value)
    
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_FIG_WIDTH, 5))
    barplot(data=data, x='Group', y='Values', palette=lancet_light[:5].tolist(), hue='Scheduler', ax=ax)
    plt.xlabel('')
    plt.ylabel('Improvement')
    plt.ylim(ymin=0, ymax=3.5)
    plt.xticks(rotation=0)
    plt.legend(title='', ncol=3, loc='upper left')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'different_memory.pdf'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIG_DIR, 'different_memory.png'), bbox_inches='tight', dpi=600)
    
def plot_different_number_GPU():
    data = {'Scheduler': [], 'Group': [], 'Values': []}
    data_4090 = {'Scheduler': [], 'Group': [], 'Values': []}
    groups = ['1', '2', '3', '4']
    
    for idx, filename in enumerate(DIFFERENT_NUM_GPU_DATA_LIST):
        filepath = os.path.join(DATA_DIR, filename)
        throughputs = pd.read_csv(filepath, index_col=0)
        
        avg_values = throughputs.loc['Avg.']
        for scheduler, value in avg_values.items():
            data['Scheduler'].append(scheduler_map[scheduler])
            data['Group'].append(groups[idx])
            data['Values'].append(value)
    
    for idx, filename in enumerate(DIFFERENT_NUM_GPU_4090_DATA_LIST):
        filepath = os.path.join(DATA_DIR, filename)
        throughputs = pd.read_csv(filepath, index_col=0)
        
        avg_values = throughputs.loc['Avg.']
        for scheduler, value in avg_values.items():
            data_4090['Scheduler'].append(scheduler_map[scheduler])
            data_4090['Group'].append(groups[idx])
            data_4090['Values'].append(value)
    
    df = pd.DataFrame(data)
    scheduler_order = ['SA', 'CASE', r'{\sc HuntK}', r'{\sc HuntKT}', r'{\sc HuntKTm}']
    table = df.pivot_table(index='Group', columns='Scheduler', values='Values', aggfunc='mean')
    table = table[scheduler_order]
    print(table)
    df = pd.DataFrame(data_4090)
    scheduler_order = ['SA', 'CASE', r'{\sc HuntK}', r'{\sc HuntKT}', r'{\sc HuntKTm}']
    table = df.pivot_table(index='Group', columns='Scheduler', values='Values', aggfunc='mean')
    table = table[scheduler_order]
    print(table)
    
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_FIG_WIDTH * 2, 4))  # 横向拼接两幅图

    # A100 improvement
    barplot(data=data, x='Group', y='Values', palette=lancet_light[:5].tolist(), hue='Scheduler', ax=axes[0])
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Improvement')
    axes[0].set_ylim(ymin=0, ymax=5.5)
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels(groups, rotation=0)
    axes[0].axhline(y=1, color='black', linestyle='--', linewidth=1)
    # axes[0].legend_.remove()  # 移除图例
    axes[0].legend(title='', ncol=3, loc='upper right')
    axes[0].text(0.5, -0.25, '(a) A100 system', ha='center', va='center', transform=axes[0].transAxes, fontsize=20, fontweight='bold')

    # 4090 improvement
    barplot(data=data_4090, x='Group', y='Values', palette=lancet_light[:5].tolist(), hue='Scheduler', ax=axes[1])
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].set_ylim(ymin=0, ymax=5.5)
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels(groups, rotation=0)
    axes[1].axhline(y=1, color='black', linestyle='--', linewidth=1)
    axes[1].legend_.remove()  # 移除图例
    axes[1].text(0.5, -0.25, '(b) 4090 system', ha='center', va='center', transform=axes[1].transAxes, fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'different_number_GPU_combined.pdf'), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(FIG_DIR, 'different_number_GPU_combined.png'), bbox_inches='tight', dpi=600)

def plot_single_stream():
    for filename in SINGLE_STREAM_DATA_LIST:
        filepath = os.path.join(DATA_DIR, filename)
        filename_without_extension = filename.rsplit(".", 1)[0]
        throughputs = pd.read_csv(filepath, index_col=0)
        groups = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'Avg.']
        data = {'Scheduler': [], 'Group': [], 'Values': []}
        
        print(throughputs)
        for row in throughputs:
            for idx, value in enumerate(throughputs[row]):
                data['Scheduler'].append(scheduler_map[row])
                data['Group'].append(groups[idx])
                data['Values'].append(value)

        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_FIG_WIDTH, 4))
        barplot(data=data, x='Group', y='Values', palette=lancet_light[:4].tolist(), hue='Scheduler', ax=ax)
        plt.xlabel('')
        plt.ylabel('Improvement')
        plt.ylim(ymin=0, ymax=2.7)
        plt.xticks(rotation=0)
        plt.legend(title='', ncol=2, loc='upper left')
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'multiGPU_{filename_without_extension}_single_stream.pdf'), bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(FIG_DIR, f'multiGPU_{filename_without_extension}_single_stream.png'), bbox_inches='tight', dpi=600)

plot_funcs = {
    'throughput': plot_throughput,
    'GPU-util': plot_gpu_util,
    'mem-util': plot_mem_util,
    'sm-active-util': plot_sm_active_util,
    'sm-occupancy-util': plot_sm_occupancy_util,
    'motivation-1': plot_motivation_1,
    'motivation-2': plot_motivation_2,
    'single-task': plot_single_task,
    'multiGPU-util': plot_multigpu_util,
    'different-stream': plot_different_stream,
    'different-memory': plot_different_memory,
    'different-num-GPU': plot_different_number_GPU,
    'single-stream': plot_single_stream
}

if __name__ == '__main__':
    set_style()
    if len(sys.argv) != 2:
        usage_and_exit()
    if sys.argv[1] not in plot_funcs:
        usage_and_exit()

    plot_funcs[sys.argv[1]]()