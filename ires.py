from typing import final
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

moment_funcs = {
    "mean": lambda x: np.mean(x, axis=-1),
    "var":  lambda x: np.var(x, axis=-1),
    "std":  lambda x: np.std(x, axis=-1),
    "skew":  lambda x: np.array(stats.skew(x, axis=-1)),
    "kur":  lambda x: np.array(stats.kurtosis(x, axis=-1)),
    "m6":  lambda x: np.array(stats.moment(x, moment=6, axis=-1)),
    "m7":  lambda x: np.array(stats.moment(x, moment=7, axis=-1)),
}

def _line_moment_func(i, o, w, s, f_s,f):
    o[...] = f(i[np.arange(w) + np.arange(f_s*s)[::s, None]])

# def _moment_func(i, o, m):
#     o = stats.moment(i, moment=m, axis=0)

class IRES:

    moment_names = {
        "mean": "Mean",
        "var": "Variance",
        "std": "Standard Deviation",
        "skew": "Skewness",
        "kur": "Kurtosis",
        "m6": "Moment 6",
        "m7": "Moment 7",
    }

    default_moments = ["mean", "var", "std", "skew", "kur"]

    def __init__(self, data, win, stride, normalized=True, moments=default_moments, in_place=False):#, batch=False, batch_dim=-1):
        self.moments = moments
        self.data = data
        expended = False

        if data.ndim == 1:
            data = data[None, :]
            expended = True

        # sub_windows = np.lib.stride_tricks.sliding_window_view(np.arange(data.shape[-1]), win)[::stride]
        # print(sub_windows.shape)

        final_shape = ((data.shape[-1]-win)//stride)
        if final_shape <= 0:
            raise Exception("Window/Stride too large")
        
        moment_data = np.empty(data.shape[:-1] + (len(self.moments), final_shape)) # initialize empty array with shape (# samples, # moments, output shape length)
        
        
        for l in list(np.ndindex(data.shape[:-1])):
            for n, m in enumerate(moments):
                if m in moment_funcs:
                    _line_moment_func(data[l], moment_data[l + (n,)], win, stride, final_shape, moment_funcs[m])

                # for l in range(np.product(data.shape[:-1])):
                #     _line_moment_func(data[np.unravel_index(l, data.shape[:-1])], moment_data[np.unravel_index(l, data.shape[:-1])+(n, None)], win, stride, final_shape, moment_funcs[m])

                
        self.normalized = normalized

        if normalized:
            moment_data = (moment_data - moment_data.min(axis=-1)[..., None]) / (moment_data.max(axis=-1) - moment_data.min(axis=-1))[..., None]

        #Remove extra dimention added to make calcs easier
        if expended:
            moment_data = np.squeeze(moment_data, axis=0)

        if in_place:
            return moment_data
        else:
            self.moment_data = moment_data

    def vis(self, index=None, name=None, raw=True, figsize=(10, 10), interpolation=2, moment_color=True, raw_color=False, cmap='rainbow'):
        num_plots = len(self.moments)
        multi = self.moment_data.ndim > 2

        if isinstance(index, int):
            index = (index, )
        if index is None:
            index = (0, ) * (self.moment_data.ndim - 2)
        if multi and len(index) != self.moment_data.ndim - 2:
            raise Exception(f"Index dims must match data dims ({self.moment_data.ndim - 2}); {len(index)} was given.")

        if raw:
            num_plots = num_plots + 1

        fig, ax = plt.subplots(num_plots, gridspec_kw=dict(height_ratios=np.ones(num_plots)), figsize=figsize)

        if name is not None:
            fig.suptitle(f"{name}", fontsize=15)

        moment_data = None
        plot_data = None
        if multi:
            moment_data = self.moment_data[(...,) + index + (slice(None), slice(None))]
            plot_data = self.data[(...,) + index + (slice(None),)]
        else:
            moment_data = self.moment_data
            plot_data = self.data

        for i, (m, moment) in enumerate(list(zip(moment_data, self.moments))[::-1]):
            ax[i].set_xlim(0, len(m)-1)

            interp = np.interp(np.linspace(0, len(m), int(len(m)*interpolation)), np.arange(len(m)), m)
            if moment_color:
                ax[i].imshow(interp[None, :], extent=[0, len(m), m.min(), m.max()], cmap=cmap, aspect='auto', interpolation='spline16', interpolation_stage="rgba")
            # if shadow:
            #     s = np.tile(np.linspace(interp.max(), interp.min(), len(interp)), (len(interp), 1)).T
            #     s[s>interp] = np.nan
            #     ax[i].imshow(s, extent=[0, len(m), m.min(), m.max()], cmap=shadow_cmap, aspect='auto', interpolation='spline16', interpolation_stage="rgba", alpha=shadow_alpha)
            ax[i].plot(np.linspace(0, len(m), len(interp)), interp, color="black")
            ax[i].set_ylabel(self.moment_names[moment], rotation=22.5, labelpad=0, va="center", ha="right")
            ax[i].set_yticklabels('')
            m_range = max(m)-min(m)
            ax[i].set_yticks([min(m) + (m_range * .1), min(m) + (m_range / 2), max(m) - (m_range * .1)])
            ax[i].set_yticklabels([f"{m.min():.2f}", f"{m.mean():.2f}", f"{m.max():.2f}"])
            ax[i].yaxis.tick_right()
            ax[i].set_ylim(min(m), max(m))
            ax[i].set_xticklabels([])
        
        if raw:
            ax[-1].set_xlim(0, len(plot_data)-1)
            interp = np.interp(np.linspace(0,len(plot_data), int(len(plot_data)*interpolation)), np.arange(len(plot_data)), plot_data)
            if raw_color:
                ax[-1].imshow(interp[None, :], extent=[0, len(plot_data), plot_data.min(), plot_data.max()], cmap=cmap, aspect='auto')
            ax[-1].plot(np.linspace(0, len(plot_data), len(interp)), interp, color="black")
            ax[-1].set_ylabel("Raw", rotation=22.5, labelpad=0, va="center", ha="right")
            ax[-1].set_yticklabels('')
            data_range = max(plot_data)-min(plot_data)
            ax[-1].set_yticks([min(plot_data) + (data_range * .1), min(plot_data) + (data_range / 2), max(plot_data) - (data_range * .1)])
            ax[-1].set_yticklabels([f"{plot_data.min():.2f}", f"{plot_data.mean():.2f}", f"{plot_data.max():.2f}"])
            ax[-1].yaxis.tick_right()
            ax[-1].set_ylim(min(plot_data), max(plot_data))
            ax[-1].set_xticklabels([])
        
        fig.tight_layout()
        fig.align_labels()

        plt.show()

        # return self

    #moment data syntatic sugar helper funcs
    @property
    def shape(self):
        return self.moment_data.shape
    def __repr__(self):
        return repr(self.moment_data)
    def __getitem__(self, i):
        return self.moment_data[i]
    def __len__(self):
        return len(self.moment_data)

