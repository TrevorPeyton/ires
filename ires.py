import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def from_data(data, win, stride, *args, **kwargs):
    return IRESObject(data, win, stride, *args, **kwargs)

def _line_moment_func(i, o, w, s, f_s, f):
    o[...] = f(i[np.arange(w) + np.arange(f_s*s)[::s, None]])

_moment_names = {
        "mean": "Mean",
        "var": "Variance",
        "std": "Standard Deviation",
        "skew": "Skewness",
        "kur": "Kurtosis",
        # "m6": "Moment 6",
        # "m7": "Moment 7",
    }

_moment_funcs = {
    "mean": lambda x: np.mean(x, axis=-1),
    "var":  lambda x: np.var(x, axis=-1),
    "std":  lambda x: np.std(x, axis=-1),
    "skew":  lambda x: np.array(stats.skew(x, axis=-1)),
    "kur":  lambda x: np.array(stats.kurtosis(x, axis=-1)),
    # "m6":  lambda x: np.array(stats.moment(x, moment=6, axis=-1)),
    # "m7":  lambda x: np.array(stats.moment(x, moment=7, axis=-1)),
}

class IRESObject:

    def _line_moment_func(i, o, w, s, f_s, f):
        o[...] = f(i[np.arange(w) + np.arange(f_s*s)[::s, None]])

    def __init__(self, data, win, stride, normalized=True, moments=None, moment_funcs=_moment_funcs, moment_names=_moment_names, *args, **kwargs):
        if moments is None:
            self.moments = list(moment_funcs.keys())
        else:
            self.moments = moments
        self.data = data
        self.moment_funcs = moment_funcs
        self.moment_names = moment_names
        self.normalized = normalized

        #expand 1d data to match nd data
        expended = False
        if data.ndim == 1:
            data = data[None, :]
            expended = True

        #calculate shape after win_size and stride
        final_shape = ((data.shape[-1]-win)//stride)
        if final_shape <= 0:
            raise Exception("Window/Stride too large")
        moment_data = np.empty(data.shape[:-1] + (len(self.moments), final_shape)) # initialize empty array with shape (# samples, # moments, output shape length)
        
        #loop through all dimensions just modifying last dimension
        for l in list(np.ndindex(data.shape[:-1])):
            for n, m in enumerate(self.moments):
                if m in self.moment_funcs:
                    _line_moment_func(data[l], moment_data[l + (n,)], win, stride, final_shape, self.moment_funcs[m])

        if self.normalized:
            moment_data = (moment_data - moment_data.min(axis=-1)[..., None]) / (moment_data.max(axis=-1) - moment_data.min(axis=-1))[..., None]

        #Return in original shape if expanded
        if expended:
            moment_data = np.squeeze(moment_data, axis=0)

        self.moment_data = moment_data

    def vis(self, index=None, name=None, raw=True, figsize=(10, 10), interpolation=2, moment_color=True, raw_color=False, cmap='rainbow', in_place=True):
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
            m_range = max(m)-min(m)
            if moment_color:
                ax[i].imshow(interp[None, :], extent=[0, len(m), min(m)-(m_range * .05), max(m)+(m_range * .05)], cmap=cmap, aspect='auto', interpolation='spline16', interpolation_stage="rgba")
            # if shadow:
            #     s = np.tile(np.linspace(interp.max(), interp.min(), len(interp)), (len(interp), 1)).T
            #     s[s>interp] = np.nan
            #     ax[i].imshow(s, extent=[0, len(m), m.min(), m.max()], cmap=shadow_cmap, aspect='auto', interpolation='spline16', interpolation_stage="rgba", alpha=shadow_alpha)
            ax[i].plot(m, color="black")
            ax[i].set_ylabel(self.moment_names[moment], rotation=22.5, labelpad=0, va="center", ha="right")
            ax[i].set_yticklabels('')
            ax[i].set_yticks([min(m), min(m) + (m_range / 2), max(m)])
            ax[i].set_yticklabels([f"{min(m):.2f}", f"{m.mean():.2f}", f"{max(m):.2f}"])
            ax[i].yaxis.tick_right()
            ax[i].set_ylim(min(m)-(m_range * .05), max(m)+(m_range * .05))
            ax[i].set_xticklabels([])
        
        if raw:
            ax[-1].set_xlim(0, len(plot_data)-1)
            interp = np.interp(np.linspace(0,len(plot_data), int(len(plot_data)*interpolation)), np.arange(len(plot_data)), plot_data)
            data_range = max(plot_data)-min(plot_data)
            if raw_color:
                ax[-1].imshow(interp[None, :], extent=[0, len(plot_data), min(plot_data)-(data_range * .05), max(plot_data)+(data_range * .05)], cmap=cmap, aspect='auto')
            ax[-1].plot(plot_data, color="black")
            ax[-1].set_ylabel("Raw", rotation=22.5, labelpad=0, va="center", ha="right")
            ax[-1].set_yticklabels('')
            ax[-1].set_yticks([min(plot_data), min(plot_data) + (data_range / 2), max(plot_data)])
            ax[-1].set_yticklabels([f"{plot_data.min():.2f}", f"{plot_data.mean():.2f}", f"{plot_data.max():.2f}"])
            ax[-1].yaxis.tick_right()
            ax[-1].set_ylim(min(plot_data) - (data_range * .05), max(plot_data) + (data_range * .05))
            ax[-1].set_xticklabels([])
        
        fig.tight_layout()
        fig.align_labels()

        plt.show()

        if not in_place:
            return self

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

