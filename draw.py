import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
from pathlib import Path

plt.rcParams["pgf.texsystem"] = "xelatex"
plt.rcParams["pgf.preamble"] = R"\usepackage{ctex}"

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
    axes_grad[0].set_xlabel('a) 无可见性梯度')


    axes_grad[1].set_xlabel('b) 错误可见性梯度\n黑色背景模型')


    axes_grad[2].set_xlabel('c) 理想可见性梯度\n理想背景模型')

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

    ax_func.set_xlabel(R'a) 目标与前景模型')
    ax_loss1.set_xlabel(R'b) $\mathcal{L}_n(\theta)$的分子')
    ax_lossn.set_xlabel(R'c) $\mathcal{L}_n(\theta)$的第一项')
    ax_lossa.set_xlabel(R'd) $\mathcal{L}_n(\theta)$')

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

def main():
    FIG_PATH.mkdir(parents=True, exist_ok=True)
    problem()
    one_dim_loss()
    l2_loss()

if __name__ == '__main__':
    main()
