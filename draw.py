import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
from pathlib import Path

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

def no_aa():
    x, y = 8, 3
    X = np.arange(x)
    Y = np.arange(y)
    X = X[None, :]
    Y = Y[:, None]

    target = np.where((X + 0 * Y < 3.5)[..., None], [1,1,1], [0.5,0.5,0.6])

    fig, (axes_target, axes_model, axes_loss, axes_grad) = \
        plt.subplots(4, 3, figsize=(6.3, 4), sharex=True, sharey=False)
    fig.set_layout_engine('compressed')

    for ax in axes_target:
        ax.imshow(target, interpolation='none', origin='lower', cmap='gray')
        ax.grid(True, which='minor')
    for axes in [axes_target, axes_model]:
        for ax in axes:
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_minor_locator(HalfLocator())
                axis.set_major_locator(ticker.MultipleLocator(1))
    axes_target[0].set_ylabel('拟合目标')

    model = np.where(2 * X - Y < 5.1, 1, 0)[..., None]
    axes_model[0].imshow(model, interpolation='none', origin='lower', cmap='gray')

    for ax in axes_model:
        ax.grid(True, which='minor')
        ax.add_line(plt.Line2D([2.3, 3.8], [-0.5, 2.5], color='red', linewidth=1))
        ax.arrow(2.7, 0, 0.5, 0, color='red', linewidth=1, head_width=0.2, head_length=0.2)
        ax.annotate('$a_x$', xy=(2.3, -0.5), xycoords='data',
                    xytext=(1,-1), textcoords='offset points', va='top',
                    color='red')
    axes_model[0].set_ylabel('渲染结果')

    X_off = np.linspace(-0.5, 7.5, 1000)
    model = np.where(2 * (X - X_off[:, None, None]) - Y < 0.5, 1, 0)[..., None]

    loss = np.sum(np.abs(model - target), axis=(1, 2, 3))
    axes_loss[0].step(X_off, loss, color='red')
    axes_loss[0].set_ylabel('损失函数')

    axes_grad[0].plot([-0.5, 7.5], [0, 0], color='red')
    axes_grad[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axes_grad[0].set_ylabel('梯度')
    axes_grad[0].set_xlabel('无可见性梯度')


    axes_grad[1].set_xlabel('错误可见性梯度')


    axes_grad[2].set_xlabel('理想可见性梯度')

    for i in range(len(axes_model)):
        trans = transforms.blended_transform_factory(
            axes_grad[i].transData, axes_grad[i].transAxes)
        con = patches.ConnectionPatch(
            xyA=(2.3, -0.5), xyB=(2.3, 0),
            coordsA='data', coordsB=trans,
            axesA=axes_model[i], axesB=axes_grad[i],
            color='gray', linewidth=1, linestyle='--'
        )
        fig.add_artist(con)

    fig.align_ylabels([axes_target[0], axes_model[0], axes_loss[0], axes_grad[0]])
    fig.savefig(FIG_PATH / 'no_aa.pgf')

def main():
    FIG_PATH.mkdir(parents=True, exist_ok=True)
    no_aa()

if __name__ == '__main__':
    main()
