from omnisafe.utils.plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 指定日志目录路径
    log_dirs = [
        r"C:\Users\28906\Desktop\cl3\runs\COptiDICE-{SafetyPointCircle1-v0}-SafetyPointCircle1-v0-mixed-beta0.25",
        r"C:\Users\28906\Desktop\cl3\runs\COptiDICE-{SafetyPointCircle1-v0}-SafetyPointCircle1-v0-mixed-beta0.5",
        r"C:\Users\28906\Desktop\cl3\runs\COptiDICE-{SafetyPointCircle1-v0}-SafetyPointCircle1-v0-mixed-beta0.75"
    ]

    # 创建Plotter对象
    plotter = Plotter()

    # 创建画布和子图对象
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 定义不同的颜色
    colors = ['red', 'green', 'blue']

    # 存储reward和cost的均值和标准差
    beta_values = []
    reward_means = []
    reward_stds = []
    cost_means = []
    cost_stds = []

    for i, log_dir in enumerate(log_dirs):
        # 获取数据集
        datasets = plotter.get_datasets(log_dir)

        # 绘制回报曲线
        plotter.plot_data(
            sub_figures=axes,
            data=datasets,
            xaxis="TotalSteps",
            value="Metrics/EpRet",
            condition=None,
            label=f"Beta {log_dir.split('-')[-1][4:]}",
            smooth=1,
            color=colors[i]
        )

        # 绘制成本曲线
        plotter.plot_data(
            sub_figures=axes,
            data=datasets,
            xaxis="TotalSteps",
            value="Metrics/EpCost",
            condition=None,
            label=f"Beta {log_dir.split('-')[-1][4:]}",
            smooth=1,
            color=colors[i]
        )

        # 提取Beta值
        beta_value = float(log_dir.split('-')[-1][4:])
        beta_values.append(beta_value)

        # 计算reward和cost的均值和标准差
        reward_mean = np.mean([d['Metrics/EpRet'] for d in datasets])
        reward_std = np.std([d['Metrics/EpRet'] for d in datasets])
        cost_mean = np.mean([d['Metrics/EpCost'] for d in datasets])
        cost_std = np.std([d['Metrics/EpCost'] for d in datasets])

        reward_means.append(reward_mean)
        reward_stds.append(reward_std)
        cost_means.append(cost_mean)
        cost_stds.append(cost_std)

    # 绘制reward均值和标准差图像
    axes[2].errorbar(beta_values, reward_means, yerr=reward_stds, fmt='o', capsize=5)
    axes[2].set_xlabel('Beta')
    axes[2].set_ylabel('Reward Mean')
    axes[2].set_title('Reward Mean and Std')

    # 绘制cost均值和标准差图像
    axes[3].errorbar(beta_values, cost_means, yerr=cost_stds, fmt='o', capsize=5)
    axes[3].set_xlabel('Beta')
    axes[3].set_ylabel('Cost Mean')
    axes[3].set_title('Cost Mean and Std')

    # 显示图像
    plt.tight_layout()
    plt.show()