import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing import event_accumulator

def radar_plot():
    # dataset = pd.DataFrame(data=[[125, 116, 78, 77, 50, 76],
    #                              [142, 115, 96, 8, 50, 92],
    #                              [145, 141, 123, 125, 50, 99],
    #                              [133, 111, 73, 68, 50, 88],
    #                              [103,  96, 66, 35, 50, 69],
    #                              [142, 100, 115, 52, 50, 93],
    #                              [144, 195, 166, 196, 50, 167],
    #                              [117, 154, 125, 88, 50, 128],
    #                              [108, 140, 118, 53, 50, 66],
    #                              [125, 124, 25, 32, 50, 86],
    #                              [128, 107, 107, 48, 50, 71],
    #                              [135, 131, 126, 129, 50, 142]],
    # index=['     Human1', '     Human2','  Human3','Human4','Human5  ',
    #     'Human6     ', 'Human7     ','Human8     ','Human9  ','Human10', 
    #     '  Human11', '     Human12'],
    # columns=['Oracle','Hypernet','ER','EWC','Packnet', 'Naive'])

    # dataset = pd.DataFrame(data=[[0, 116, 78, 77, 50,0],
    #                              [0, 115, 96, 50, 50,0],
    #                              [0, 141, 123, 125, 50,0],
    #                              [0,111, 73, 68, 50,0],
    #                              [0,96, 66, 50, 50,0],
    #                              [0,100, 115, 52, 50,0],
    #                              [0,195, 166, 196, 50,0],
    #                              [0,154, 125, 88, 50,0],
    #                              [0,140, 118, 53, 50,0],
    #                              [0,124, 50, 50, 50,0],
    #                              [0,107, 107, 50, 50,0],
    #                              [0,131, 126, 129, 50,0]],
    # index=['     Human1', '     Human2','  Human3','Human4','Human5  ',
    #     'Human6     ', 'Human7     ','Human8     ','Human9  ','Human10', 
    #     '  Human11', '     Human12'],
    # columns=['Oracle','Hypernet','ER','EWC','Packnet', 'Naive'])

    # dataset = pd.DataFrame(data=[[125, 116, 0, 0, 0, 76],
    #                              [142, 115, 0, 0, 0, 92],
    #                              [145, 141, 0, 0, 0, 99],
    #                              [133, 111, 0, 0, 0, 88],
    #                              [103,  96, 0, 0, 0, 69],
    #                              [142, 100, 0, 0, 0, 93],
    #                              [144, 195, 0, 0, 0, 167],
    #                              [117, 154, 0, 0, 0, 128],
    #                              [108, 140, 0, 0, 0, 66],
    #                              [125, 124, 0, 0, 0, 86],
    #                              [128, 107, 0, 0, 0, 71],
    #                              [135, 131, 0, 0, 0, 142]],
    # index=['     Human1', '     Human2','  Human3','Human4','Human5  ',
    #     'Human6     ', 'Human7     ','Human8     ','Human9  ','Human10', 
    #     '  Human11', '     Human12'],
    # columns=['Oracle','Hypernet','ER','EWC','Packnet', 'Naive'])

    # dataset = pd.DataFrame(data=[[116, 122, 118, 118, 76],
    #                              [115, 107, 115, 114, 92],
    #                              [141, 140, 149, 145, 99],
    #                              [111, 99,  104, 104, 88],
    #                              [96,  83,  71,  66, 69],
    #                              [100, 116, 103, 111, 93],
    #                              [195, 153, 152, 143, 167],
    #                              [154, 131, 104, 133, 128],
    #                              [140, 128, 118, 136, 66],
    #                              [124, 117, 117, 125, 86]],
    # index=['     Human1', '     Human2','  Human3','Human4','Human5  ',
    #     'Human6     ', 'Human7     ','Human8     ','Human9  ','Human10'],
    # columns=['ID', 'episode32','episode64','episode160','Naive'])

    dataset = pd.DataFrame(data=[[0, 122, 118, 118, 0],
                                 [0, 107, 115, 114, 0],
                                 [0, 140, 149, 145, 0],
                                 [0, 99,  104, 104, 0],
                                 [0, 83,  71,  66,  0],
                                 [0, 116, 103, 111, 0],
                                 [0, 153, 152, 143, 0],
                                 [0, 131, 104, 133, 0],
                                 [0, 128, 118, 136, 0],
                                 [0, 117, 117, 125, 0]],
    index=['     Human1', '     Human2','  Human3','Human4','Human5  ',
        'Human6     ', 'Human7     ','Human8     ','Human9  ','Human10'],
    columns=['ID', 'episode32','episode64','episode160','Naive'])

    radar_labels=dataset.index
    nAttr = len(dataset.index)
    data=dataset.values #数据值
    data_labels=dataset.columns
    # 设置角度
    angles=np.linspace(0,2*np.pi,nAttr,
    endpoint= False)
    data=np.concatenate((data, [data[0]]))
    angles=np.concatenate((angles, [angles[0]]))
    # 设置画布
    fig=plt.figure(facecolor="white",figsize=(10,6))
    ax = plt.subplot(111, polar=True)
    # 绘图
    plt.plot(angles, data,'o-', linewidth=2.0, alpha= 0.6)
    # 填充颜色
    plt.fill(angles, data, alpha=0.25)
    plt.thetagrids(angles[:-1]*180/np.pi, radar_labels, 1.2)
    plt.figtext(0.52, 0.95,'Performance with distinct human proxy models',ha='center', size=15)
    # 设置图例
    ax.set_rlim(40, 200)
    legend=plt.legend(data_labels,
    loc=(1.1, 0.05),
    labelspacing=0.1)
    plt.setp(legend.get_texts(),
    fontsize='large')
    plt.grid(True)
    plt.savefig('radar map.png')
    
def bar_plot():
    # plt.figure(figsize=(16, 6))
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.array([0, 3, 6, 9])
    y = [[127, 127, 127, 127],
         [127, 127, 127, 127],
         [103, 103, 103, 103],
         [75, 75, 75, 75],
         [0, 0, 0, 0],
         [98, 98, 98, 98]]
    c = ['slategrey', 'deepskyblue', 'saddlebrown', 'firebrick', 'y', 'orange']

    bar_width = 0.25
    tick_label1 = ['Oracle', 'Hypernet', 'ER', 'EWC', 'Packnet', 'Naive']
    tick_label2 = ['Ring', 'unident_s8', 'random3', 'random4']
    ax.set_xticks(list(x + 2 *bar_width))
    ax.set_xticklabels(tick_label2, fontsize=18)
    for i in range(len(y)):
        ax.bar(x + i * (bar_width +0.02), y[i], bar_width, color=c[i], align='center', label=tick_label1[i], alpha=0.75)

    plt.ylim([60, 130])  
    # plt.xlabel("maps", fontsize=15)
    plt.ylabel("Average reward per episode", fontsize=18)
    plt.title("Performance with human proxy model", fontsize=18)
    plt.legend()
    plt.savefig('bar.png')


def plot_training(smoothing=True):
    
    stats = []
    times = []
    color = ['slategrey', 'deepskyblue', 'saddlebrown', 'firebrick', 'y', 'orange']
    interval = 5000000
    tasks = 12

    tb_multi = [['./results/tb_logs/offpg__2023-03-27_15-17-35/events.out.tfevents.1679930259.common-z6j6pznv', 12]]
    
    # tb_hyper = [['./results/tb_logs/offpg__2023-03-29_05-31-07/events.out.tfevents.1680067867.common-d6gh1lmt', 2],
    #             ['./results/tb_logs/offpg__2023-03-30_06-03-39/events.out.tfevents.1680156219.common-d6gh1lmt', 3],
    #             ['./results/tb_logs/offpg__2023-03-30_13-17-45/events.out.tfevents.1680182265.common-d6gh1lmt', 12]]
    # tb_hyper = [['./results/tb_logs/offpg__2023-03-29_05-31-27/events.out.tfevents.1680067887.common-d6gh1lmt', 0],
    #             ['./results/tb_logs/offpg__2023-03-31_22-44-58/events.out.tfevents.1680273900.common-z6j6pznv', 12]]
    tb_hyper = [['./results/tb_logs/offpg__2023-03-29_05-31-27/events.out.tfevents.1680067887.common-d6gh1lmt', 0],
                ['./results/tb_logs/offpg__2023-03-31_22-45-17/events.out.tfevents.1680273920.common-z6j6pznv', 12]]

    # tb_naive = [['./results/tb_logs/offpg__2023-03-29_05-29-14/events.out.tfevents.1680067754.common-d6gh1lmt', 12]]
    tb_naive = [['./results/tb_logs/offpg__2023-03-31_07-53-34/events.out.tfevents.1680249214.common-d6gh1lmt', 12]]
    
    # tb_ER = [['./results/tb_logs/offpg__2023-03-29_05-29-28/events.out.tfevents.1680067768.common-d6gh1lmt', 0], 
    #          ['../CRL_human_AI_DOP_baseline/results/tb_logs/offpg__2023-03-30_13-28-19/events.out.tfevents.1680182899.common-d6gh1lmt', 6],
    #          ['../CRL_human_AI_DOP_baseline/results/tb_logs/offpg__2023-04-01_10-02-43/events.out.tfevents.1680314565.common-z6j6pznv', 12],]
   
    tb_ER = [['./results/tb_logs/offpg__2023-03-29_05-29-28/events.out.tfevents.1680067768.common-d6gh1lmt', 0], 
             ['../CRL_human_AI_DOP_baseline/results/tb_logs/offpg__2023-03-30_10-19-13/events.out.tfevents.1680171555.common-z6j6pznv', 8],
             ['../CRL_human_AI_DOP_baseline/results/tb_logs/offpg__2023-04-01_10-03-01/events.out.tfevents.1680314583.common-z6j6pznv', 12],]   
    
    tb_EWC = [['../CRL_human_AI_DOP_baseline/results/tb_logs/offpg__2023-03-30_13-36-07/events.out.tfevents.1680183367.common-d6gh1lmt', 7],
              ['../CRL_human_AI_DOP_baseline/results/tb_logs/offpg__2023-03-31_22-08-54/events.out.tfevents.1680271735.common-z6j6pznv', 8],
              ['../CRL_human_AI_DOP_baseline/results/tb_logs/offpg__2023-04-01_09-12-06/events.out.tfevents.1680311528.common-z6j6pznv', 12]]
    
    tb_total = [tb_multi, tb_hyper, tb_naive, tb_ER, tb_EWC]
    names = ['Oracle', 'Hypernet', 'Naive', 'ER', 'EWC']
    for index in range(len(tb_total)):
        tb_files = tb_total[index]
        st = [[] for _ in range(tasks)]
        ti = [[] for _ in range(tasks)]
        st[0].append(0)
        ti[0].append(0)
        for tb_file, end_task in tb_files:
            ea=event_accumulator.EventAccumulator(tb_file) 
            ea.Reload()
            reward_total = {}
            for t_i in range(tasks):
                task_name = 'test_task{}_return'.format(t_i)
                reward_total[task_name] = \
                    ea.scalars.Items(task_name)
                for info in reward_total[task_name]:
                    if info.step > tasks * interval:
                        break
                    if info.step > (end_task + 1) * interval:
                        continue
                    if info.step < t_i * interval:
                        continue
                    if info.step in ti[t_i]:
                        st[t_i][ti[t_i].index(info.step)] = info.value
                    else:
                        ti[t_i].append(info.step)
                        st[t_i].append(info.value)
        stats.append(st)
        times.append(ti)
    # import pdb; pdb.set_trace()

    for t in range(tasks):

        fig = plt.figure(figsize=(20, 5))
        for j in range(tasks):
            plt.plot(np.ones(200) * interval * (j + 1), np.arange(0, 200), '--', c='darkgrey')
        for i in range(len(stats)):
            if stats[i][t] == []:
                continue
            arr = np.array(stats[i][t]).reshape(-1, len(stats[i][t]))
            time = times[i][t]
            for _ in range(arr.shape[0]):
                mu=arr.mean(0)
                # mu = savgol_filter(mu, 9, 2) if smoothing else mu
                mu = smooth_single(mu, 10)
                std=arr.std(0)
                std = 12
                plt.plot(time, mu, '-', color=color[i], linewidth=2.0, label=names[i])
                plt.fill_between(time, mu-std, mu+std, color=color[i], alpha=0.1)
        #     print(mu)
        # import pdb; pdb.set_trace()
            
        plt.ylabel('Task {}'.format(t), fontsize=25)
        plt.xlabel('Environment Steps', fontsize=25)
        plt.xlim([-1000, max(time) + 1000])
        plt.ylim([35, 150])
        plt.legend(fontsize=15)
        # plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
        filename = f'task{t}.png'
        plt.savefig(filename)
        plt.close()

def smooth_single(data, smooth_range):
    new_data = np.zeros_like(data)
    for i in range(0, data.shape[-1]):
        if i < smooth_range:
            new_data[i] = 1. * np.sum(data[:i + 1], axis=0) / (i + 1)
        else:
            new_data[i] = 1. * np.sum(data[i - smooth_range + 1:i + 1], axis=0) / smooth_range
    return new_data

if __name__ == '__main__':
    
    radar_plot()
    # bar_plot()
    # plot_training()
    
    