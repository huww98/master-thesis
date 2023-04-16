import matplotlib
matplotlib.use('pgf')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R

plt.rcParams["pgf.texsystem"] = "xelatex"
plt.rcParams["pgf.preamble"] = R"""\usepackage[zihao=-4]{ctex}
\setmainfont{Times New Roman}"""

FIG_PATH = Path('build/figures')

class HalfLocator(ticker.Locator):
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        'Return the locations of the ticks'
        vmin = np.floor(vmin * 2) / 2
        vmax = np.ceil(vmax * 2) / 2
        return np.arange(vmin, vmax + 0.5, 0.5)

def problem():
    x, y = 8, 3
    k = 2.
    a_x = 2.1
    X = np.arange(x)
    Y = np.arange(y)
    X = X[None, :]
    Y = Y[:, None]

    FG = np.array([1, 1, 1])
    BG = np.array([0.6, 0.6, 0.7])

    target = np.where((X + 0 * Y < 3.5)[..., None], FG, BG)

    fig, (axes_target, axes_model, axes_loss, axes_grad) = \
        plt.subplots(4, 3, figsize=(6.3, 4), sharex=True, sharey=False)
    fig.set_layout_engine('compressed')

    for ax in axes_target:
        ax.imshow(target, interpolation='none', origin='lower')
        ax.grid(True, which='minor')
    for axes in [axes_target, axes_model]:
        for ax in axes:
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_minor_locator(HalfLocator())
                axis.set_major_locator(ticker.MultipleLocator(1))
    axes_target[0].set_ylabel(R'目标$\mathcal{I}$')

    a = k * a_x + 0.5
    model = np.where(k * X - Y < a, 1, 0)[..., None]
    axes_model[0].imshow(model, interpolation='none', origin='lower', cmap='gray')

    n_sample = 256
    sample = (np.arange(n_sample) + 0.5) / n_sample - 0.5

    for ax in axes_model:
        ax.grid(True, which='minor')
        ax.add_line(plt.Line2D([a_x, a_x + y/k], [-0.5, y-0.5], color='red', linewidth=1))
        ax.arrow(2.7, 0, 0.5, 0, color='red', linewidth=1, head_width=0.2, head_length=0.2)
        ax.annotate('$a_x$', xy=(a_x, -0.5), xycoords='data',
                    xytext=(1,-1), textcoords='offset points', va='top',
                    color='red')
    axes_model[0].set_ylabel(R'渲染$\hat{\mathcal{I}}$')

    X_off = np.linspace(-0.5, x-0.5, 256)
    model_batch = np.where(k * (X - X_off[:, None, None]) - Y < 0.5, 1, 0)[..., None]
    loss = np.sum(np.abs(model_batch - target), axis=(1, 2, 3))
    axes_loss[0].step(X_off, loss, color='red')
    axes_grad[0].plot([-0.5, x-0.5], [0, 0], color='red')

    Xo = X_off[:, None, None, None, None]
    def plot_to(idx):
        axes_model[idx].imshow(model, interpolation='none', origin='lower', cmap='gray')
        loss = np.sum(np.abs(model_batch - target), axis=(1, 2, 3))
        axes_loss[idx].plot(X_off, loss, color='red')
        grad = np.gradient(loss, X_off)
        axes_grad[idx].axhline(0, color=[0.7]*3, linewidth=1)
        axes_grad[idx].plot(X_off, grad, color='red')

    model = k * (X[..., None, None] + sample) - (Y[..., None, None] + sample[:, None]) < a
    model = np.mean(model, axis=(-1, -2))[..., None]
    model_batch = k * ((X[..., None, None] - Xo) + sample) - (Y[..., None, None] + sample[:, None]) < 0.5
    model_batch = np.mean(model_batch, axis=(-1, -2))[..., None]
    plot_to(1)

    model = FG * model + BG * (1 - model)
    model_batch = FG * model_batch + BG * (1 - model_batch)
    plot_to(2)


    axes_loss[0].set_ylabel(R'损失函数$\mathcal{L}$')

    axes_grad[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axes_grad[0].set_ylabel(R'梯度')
    axes_grad[0].set_xlabel('(a) 无可见性梯度')


    axes_grad[1].set_xlabel('(b) 错误可见性梯度\n黑色背景模型')


    axes_grad[2].set_xlabel('(c) 理想可见性梯度\n理想背景模型')

    for i in range(len(axes_model)):
        trans = transforms.blended_transform_factory(
            axes_grad[i].transData, axes_grad[i].transAxes)
        con = patches.ConnectionPatch(
            xyA=(a_x, -0.5), xyB=(a_x, 0),
            coordsA='data', coordsB=trans,
            axesA=axes_model[i], axesB=axes_grad[i],
            color='gray', linewidth=1, linestyle='--'
        )
        fig.add_artist(con)

    fig.align_ylabels([axes_target[0], axes_model[0], axes_loss[0], axes_grad[0]])
    fig.savefig(FIG_PATH / 'problem.pgf')

def one_dim_loss():
    fig, (ax_func, ax_loss1, ax_lossn, ax_lossa) = \
        plt.subplots(1, 4, figsize=(6.3, 1.5))
    fig.set_layout_engine('compressed')
    # hide y ticks
    for ax in [ax_func, ax_loss1, ax_lossn, ax_lossa]:
        ax.yaxis.set_major_locator(ticker.NullLocator())

    theta = 0.7
    ax_func.stairs([1, 0.1], [0, 0.6, 1], baseline=None, linestyle='--', label='$\mathcal{I}$')
    ax_func.plot([0,theta], [0.8, 0.8], color='red', label='$\hat{\mathcal{I}}$')
    ax_func.legend()

    trans = transforms.blended_transform_factory(
        ax_func.transData, ax_func.transAxes)
    ax_func.add_artist(patches.ConnectionPatch(
        xyA=(theta, 0.8), xyB=(theta, 0),
        coordsA='data', coordsB=trans,
        color='gray', linewidth=1, linestyle='--'
    ))
    ax_func.annotate(
        R'$\theta$', xy=(theta, 0), xycoords='data',
        xytext=(0,-1), textcoords='offset points', va='top', ha='center',
        color='red')

    ax_loss1.plot([0, 0.6, 1], [0, 0.2*0.6, 0.2*0.6+0.7*0.4], color='red')

    ax_lossn.plot([0, 0.6, 1], [0.2, 0.2, 0.2*0.6+0.7*0.4], color='red')

    alpha = 0.1
    ax_lossa.plot([0, 0.6, 1], [0.2, 0.2 - 0.6*alpha, 0.2*0.6+0.7*0.4 - 1*alpha], color='red')

    ax_func.set_xlabel(R'(a) 目标与前景模型')
    ax_loss1.set_xlabel(R'(b) $\mathcal{L}_n(\theta)$的分子')
    ax_lossn.set_xlabel(R'(c) $\mathcal{L}_n(\theta)$的第一项')
    ax_lossa.set_xlabel(R'(d) $\mathcal{L}_n(\theta)$')

    fig.savefig(FIG_PATH / 'one_dim_loss.pgf')

def l2_loss():
    fig, ((ax_target, ax_loss), (ax_model, ax_grad)) = \
        plt.subplots(2, 2, figsize=(6.3, 2), sharex=True)
    fig.set_layout_engine('compressed')

    for ax in [ax_target, ax_model]:
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_minor_locator(HalfLocator())
            axis.set_major_locator(ticker.MultipleLocator(1))
        ax.grid(True, which='minor', axis='both')
    # ax_model.sharey(ax_target)
    # ax_model.yaxis.set_tick_params(labelleft=False)
    ax_model.set_ylim(-.5, 2.5)

    a_x = 2.1
    x,y = 6,3
    X = np.arange(x)
    Y = np.arange(y)
    X = X[None, :]
    Y = Y[:, None]

    target = X + 0 * Y < 2.5
    ax_target.imshow(target, cmap='gray', interpolation='none', origin='lower')

    n_sample = 256
    sample = (np.arange(n_sample) + 0.5) / n_sample - 0.5
    model = (X[..., None] + sample) - 0 * (Y[..., None]) < a_x
    model = np.mean(model, axis=(-1))
    ax_model.imshow(model, cmap='gray', interpolation='none', origin='lower')
    ax_model.arrow(2.7, 0, 0.5, 0, color='red', linewidth=1, head_width=0.2, head_length=0.2)
    ax_model.add_line(plt.Line2D([a_x, a_x], [-1, y], color='red', linewidth=1, clip_on=False))

    X_off = np.linspace(-0.5, x-0.5, 256)
    Xo = X_off[..., None, None, None]

    model_batch = ((X[..., None] - Xo) + sample) - 0 * (Y[..., None]) < 0
    model_batch = np.mean(model_batch, axis=(-1))
    loss = np.mean((model_batch - target)**2, axis=(-1, -2))
    ax_loss.plot(X_off, loss, color='red')

    grad = np.gradient(loss, X_off)
    ax_grad.plot(X_off, grad, color='red')
    ax_grad.axhline(0, color=[0.7]*3, linewidth=1)

    ax_target.set_ylabel(R'目标')
    ax_model.set_ylabel(R'渲染结果')
    ax_loss.set_ylabel(R'L2损失')
    ax_grad.set_ylabel(R'梯度')

    fig.align_ylabels([ax_target, ax_model])
    fig.align_ylabels([ax_loss, ax_grad])

    fig.savefig(FIG_PATH / 'l2_loss.pgf')

def sdf():
    img = cv2.imread('data/sdf.exr', cv2.IMREAD_UNCHANGED)[::-1, ..., ::-1]
    border = np.load('data/border.npy')
    sdf_f(img[..., 0], border)
    sdf_grad(img[..., 1:], border)

def sdf_f(img, border):
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.5))
    fig.set_layout_engine('compressed')

    ax_img = ax.imshow(img, interpolation='none', origin='lower', extent=[0, 1, 0, 1])
    for b in border:
        ax.add_line(plt.Line2D(b[:,0], b[:,1], color='gray', linewidth=1))
    fig.colorbar(ax_img, ax=ax)

    fig.savefig(FIG_PATH / 'sdf.pgf')

def sdf_grad(img, border):
    fig, ax = plt.subplots(1, 1, figsize=(2.9, 2.5))
    fig.set_layout_engine('compressed')

    img = (img + 1) / 2
    h, w, c = img.shape
    img3 = np.zeros((h, w, 3))
    img3[..., :2] = img[..., :2]
    ax.imshow(img3, interpolation='none', origin='lower', extent=[0, 1, 0, 1])
    for b in border:
        ax.add_line(plt.Line2D(b[:,0], b[:,1], color='gray', linewidth=1))
    ax.yaxis.set_major_locator(ticker.NullLocator())

    fig.savefig(FIG_PATH / 'sdf_grad.pgf')

def HDRI_stats():
    fig, (ax_hist, ax_var) = plt.subplots(1, 2, figsize=(6.3, 2.8))
    ax_stats = ax_hist.twinx()
    fig.set_layout_engine('compressed')
    data = np.load('data/HDRI_stats.npz')
    i = 4

    # Histogram
    BIN_SIZE = 32
    weights = data['count'][i].reshape(-1, BIN_SIZE).sum(axis=-1)
    bins = np.arange(0, data['count'].shape[1] + 1, BIN_SIZE)
    ax_hist.hist(bins[:-1], bins=bins, weights=weights,
                 color='#BBB', histtype='stepfilled')
    ax_hist.set_ylim(0, 10000)
    ax_hist.yaxis.set_major_locator(ticker.NullLocator())
    ax_hist.set_xlabel(R'像素值（曝光时间1/15秒）')

    ax_stats.set_ylabel(R'像素值（曝光时间1/8秒）')
    ax_stats.set_xlim(0, 12000)
    ax_stats.yaxis.set_label_position('left')
    ax_stats.yaxis.set_ticks_position('left')
    BIN_SIZE = 16
    weights = data['count'][i].reshape(-1, BIN_SIZE).sum(axis=-1)
    valid = weights > 10
    weights = weights[valid]
    means = data['sum'][i].reshape(-1, BIN_SIZE).sum(axis=-1)[valid] / weights
    x = np.arange(data['count'].shape[1]) * data['count'][i]
    x = x.reshape(-1, BIN_SIZE).sum(axis=-1)[valid] / weights
    ax_stats.plot(x, means, linewidth=2)

    origin = (data['black'][i], data['black'][i + 1])
    slope = data['exp'][i + 1] / data['exp'][i]
    ax_stats.axline(origin, slope=slope,
                    color='red', linestyle='--', linewidth=1,
                    label=f'估计：$r_i={slope:.2f}$')
    ax_stats.axline(origin, slope=data['r'][i],
                    color='orange', linestyle='--', linewidth=1,
                    label=f'拟合：$r^*_i={data["r"][i]:.2f}$')
    ax_stats.legend()
    ax_stats.set_title(R'(a) 相邻曝光照片的像素值关联', y=-0.38)

    std = [1.299498100287206, 1.7574200114417322, 1.8000141805437535, 2.7799237496688822]
    ax_var.plot(
        ["100", "200", "400", "800"],
        np.square(std),
        "o-", linewidth=2 )
    ax_var.set_xlabel(R'ISO')
    ax_var.set_ylabel(R'$\sigma_i^2$')
    ax_var.set_ylim(bottom=0)
    ax_var.set_title(R'(b) ISO与观测噪音方差', y=-0.38)

    fig.savefig(FIG_PATH / 'HDRI_stats.pgf')

def corner_fit():
    fig, (ax_angle, ax_d, ax_gain) = plt.subplots(1,3, figsize=(6.3, 2.0))
    fig.set_layout_engine('compressed')

    data = np.load('data/corner_data.npz')
    fit = np.load('data/corner_fit_14.npz')

    rxyz = data['rxyz']
    # ZYX euler angle to angle with xy plane
    angle = R.from_euler('XYZ', rxyz).as_matrix()[:,2,2]
    angle = np.rad2deg(np.arccos(angle))
    d = data['xyz'][:,2]

    angle_mask = (d > -.3) & (d < .2)
    reproj_err_o = np.linalg.norm(fit['opencv'], axis=-1).mean(axis=0)
    ax_angle.scatter(angle[angle_mask], reproj_err_o[angle_mask], s=1, label='OpenCV')

    reproj_err_s = np.linalg.norm(fit['saddle'], axis=-1).mean(axis=0)
    ax_angle.scatter(angle[angle_mask], reproj_err_s[angle_mask], s=1, label='本文')

    ax_angle.set_xlabel(R'标定板与成像平面夹角（度）')
    ax_angle.set_ylabel(R'重投影误差（像素）')
    ax_angle.set_ylim(-0.03, 0.9)
    ax_angle.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax_angle.legend()

    d_mask = (angle < 70)
    ax_d.scatter(d[d_mask], reproj_err_o[d_mask], s=1, label='OpenCV')
    ax_d.scatter(d[d_mask], reproj_err_s[d_mask], s=1, label='本文')
    ax_d.set_xlabel(R'标定板与焦平面距离（米）')
    ax_d.set_ylim(-0.1, 2.6)
    ax_d.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_d.legend()

    gain_exp_2 = list(range(7, 15)) + ['inf']
    gain_exp_2 = gain_exp_2[::-1]
    gain_mask = d_mask & angle_mask
    reproj_err_s = []
    reproj_err_o = []
    for g in gain_exp_2:
        fit_g = np.load(f'data/corner_fit_{g}.npz')
        reproj_err_s.append(np.linalg.norm(fit_g['saddle'][:, gain_mask], axis=-1).mean(axis=(0,1)))
        reproj_err_o.append(np.linalg.norm(fit_g['opencv'][:, gain_mask], axis=-1).mean(axis=(0,1)))

    ax_gain.plot(gain_exp_2, reproj_err_o, 'o-', label='OpenCV')
    ax_gain.plot(gain_exp_2, reproj_err_s, 'o-', label='本文')
    ax_gain.set_xlabel(R'增益（log2最大电子数）')
    ax_gain.legend()

    fig.savefig(FIG_PATH / 'corner_fit.pgf')

    IMG_ID = [23]
    GAIN = [14, 10]
    for g in GAIN:
        fit = np.load(f'data/corner_fit_{g}.npz')
        for iid in IMG_ID:
            img = cv2.imread(f'data/corners/{iid:04d}.exr', cv2.IMREAD_UNCHANGED)[::-1, ..., 0]
            gain = 2**g
            img = np.random.poisson(img * gain) / gain
            for t in ['opencv', 'saddle']:
                fig, ax = plt.subplots(1,1, figsize=(1.57, 1.57))
                ax.set_axis_off()
                # set frame color
                ax.set_position([0, 0, 1, 1])
                ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='none', origin='lower')
                truth = data['img'][iid]
                off_o = fit[t][:, iid] + truth
                ax.scatter(truth[0], truth[1], s=4, c='green')
                ax.scatter(off_o[:, 0], off_o[:, 1], s=1, c='red')
                fig.savefig(FIG_PATH / f'corner_gain-{g}_{iid:04d}_{t}.pgf')

STAB_DATA = [
    ("264029004613", 1121, 0.32),
    ("284021000282", 1250, 0.31),
    ("284021000925", 1106, 0.31),
    ("284021006242", 1143, 0.45),
    ("284021006877", 1283, 0.47),
    ("284021006904", 1190, 0.29),
    ("284021006911", 1250, 0.55),
    ("284021006920", 1361, 0.37),
    ("284021009519", 1271, 0.37),
    ("294021000859", 1178, 1.04),
    ("294021002041", 1217, 0.39),
    ("304021000278", 1360, 0.43),
]

def stab_ablation():
    stab_data = sorted(STAB_DATA, key=lambda x: x[0])
    fig,ax = plt.subplots(1,1, figsize=(3.3, 2.0))
    fig.set_layout_engine('compressed')
    ax.set_ylabel(R'重投影误差（像素）')
    x = np.arange(len(stab_data)) + 1
    ax.scatter(
        x,
        [d[2] for d in stab_data],
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylim(0, 1.1)

    fig.savefig(FIG_PATH / 'stab_ablation.pgf')

def landmark():
    data = np.load('data/lms68.npz')

    fig,ax = plt.subplots(1,1, figsize=(1.55, 1.55))
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    # transparent background
    fig.patch.set_alpha(0)
    lower = np.array([79.5, 439.5])
    upper = lower + 384
    bound = np.array([lower, upper])
    bound = bound / 1024 * 2 - 1
    ax.set_xlim(bound[0, 0], bound[1, 0])
    ax.set_ylim(bound[1, 1], bound[0, 1])
    ax.scatter(data['pred'][:, 0], data['pred'][:, 1], s=1, c='red')
    ax.scatter(data['gt'][:, 0], data['gt'][:, 1], s=1, c='green')

    fig.savefig(FIG_PATH / 'landmark.pgf')

DATA = {
    '黑色背景': {
        'iter': [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,],
        'loss': [4.225361,4.176862,4.165521,4.159070,4.154953,4.151855,4.149079,4.146403,4.143839,4.141201,4.138553,4.135942,4.133227,4.130594,4.127884,4.125270,4.122549,4.119808,4.116897,4.114006,],
        'rot': [0.012015,0.015639,0.013280,0.012030,0.012231,0.013207,0.014449,0.015725,0.017047,0.018722,0.020409,0.022046,0.023718,0.025348,0.027103,0.028545,0.030013,0.031603,0.033237,0.034773,],
    },
    '数学期望背景': {
        'iter': [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,],
        'loss': [2.656832,2.600445,2.588517,2.584687,2.583514,2.582843,2.582439,2.582088,2.581746,2.581423,2.581124,2.580792,2.580472,2.580146,2.579816,2.579477,2.579119,2.578755,2.578379,2.578011,],
        'rot': [0.012015,0.014355,0.009666,0.007035,0.006140,0.005846,0.005795,0.005867,0.006023,0.006241,0.006527,0.006879,0.007291,0.007733,0.008147,0.008562,0.008977,0.009360,0.009784,0.010182,]
    },
    '本章方法': {
        'iter': [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,],
        'loss': [0.066409,0.019071,0.010587,0.007624,0.006929,0.006665,0.006471,0.006344,0.006272,0.006230,0.006212,0.006171,0.006145,0.006134,0.006122,0.006120,0.006125,0.006111,0.006092,0.006097,],
        'rot': [0.012017,0.014279,0.008664,0.004788,0.003239,0.002316,0.001721,0.001330,0.001049,0.000846,0.000703,0.000614,0.000535,0.000481,0.000430,0.000389,0.000355,0.000342,0.000326,0.000310,],
    },
}

def loss_vs_rot():
    fig, (ax_loss, ax_rot) = plt.subplots(1,2, figsize=(6.3, 2.5))
    fig.set_layout_engine('compressed')

    ax_loss.set_ylabel(R'归一化后损失')
    ax_loss.set_xlabel(R'迭代次数')
    ax_rot.set_ylabel(R'旋转误差（°）')
    ax_rot.set_xlabel(R'迭代次数')

    for name, data in DATA.items():
        loss = data['loss']
        loss = (np.array(loss) - loss[-1]) / (loss[0] - loss[-1])
        ax_loss.plot(data['iter'], loss, label=name)
        rot = data['rot']
        rot = np.rad2deg(rot)
        ax_rot.plot(data['iter'], rot, label=name)

    ax_loss.legend()
    ax_rot.legend()

    fig.savefig(FIG_PATH / 'loss_vs_rot.pgf')

def main():
    FIG_PATH.mkdir(parents=True, exist_ok=True)
    loss_vs_rot()
    HDRI_stats()
    problem()
    one_dim_loss()
    l2_loss()
    sdf()
    corner_fit()
    stab_ablation()
    landmark()

if __name__ == '__main__':
    main()
