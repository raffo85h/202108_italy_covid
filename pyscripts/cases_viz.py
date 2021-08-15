import numpy as np
import pandas as pd
from scipy.constants import golden
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


__all__ = ['NationalCases', 'RegionalCases']


class BaseCases():

    def __init__(self, df, is_region, const=None):
        self.df = df
        self.col = 'denominazione_regione' if is_region else 'variable'
        self.const = const

    def _get_plt_multi_metrics_single_region_df(self, *args, **kwargs):
        raise NotImplementedError()

    def _plot_small_multiples(
        self, plt_df, start_date, cols, col_and_hue_order, n_colors, title, col_wrap=None, single_color=False
    ):
        pal = get_pal(n_colors)
        if single_color:
            pal = pal[-1:]
        with sns.axes_style("whitegrid"):
            g = sns.FacetGrid(
                data=plt_df, col="variable", hue="variable",
                col_order=col_and_hue_order, col_wrap=col_wrap,
                hue_order=col_and_hue_order,
                palette=pal, height=2.5, aspect=1.2,
            )
            g.map(sns.lineplot, 'data', 'value', linewidth=4, zorder=99)
            for ax in g.axes.flatten():
                t = ax.get_title().split(' = ')[1]
                ax.text(.99, .9, t, transform=ax.transAxes, fontweight="bold", ha='right', zorder=1000)
                sns.lineplot(
                    data=plt_df, x="data", y="value", units="variable",
                    estimator=None, color=".6", linewidth=1, ax=ax,
                )
                try:
                    ax.legend_.remove()
                except Exception as e:
                    print(f'exception when removing legend: {t}')
        g.set_titles('')
        g.set_axis_labels(x_var='', y_var='')
        g.set_xdates(start_date, self.const.END_DATE, fmt='%b', freq='1MS')
        g.despine(left=True)
        g.fig.tight_layout(h_pad=0)
        g.set_suptitle(title)
        g.set_subtitle(self.const.FOOTNOTE, x=.99, y=-.025, fs=12, ha='right')
        return g

    def plot_small_multiples_many_metrics(
        self, start_date, cols, n, title, col_wrap=None, region=None,
        is_norm=True, renorm=False
    ):
        plt_df = self._get_plt_multi_metrics_single_region_df(start_date, cols, region, renorm)
        renamed_cols = list(map(lambda x: x.replace('_norm', '').replace('_reg', '').replace('_', ' '), cols))
        g = self._plot_small_multiples(plt_df, start_date, cols, renamed_cols, n, title, col_wrap=col_wrap)
        if is_norm:
            g.set(ylim=(0, 1), yticks=np.arange(0, 1.01, .2))
        return g


class NationalCases(BaseCases):
    def __init__(self, df, const):
        super().__init__(df, is_region=False, const=const)

    def _get_plt_multi_metrics_single_region_df(self, start_date, cols, region=None, renorm=False):
        tmp_df = (
            self.df
            .query('data>=@start_date')
            [['data'] + cols]
        )
        if renorm:
            print('Country | Renormalize!')
            for c in cols:
                tmp_df[c] = tmp_df[c] / tmp_df[c].max()
        return (
            tmp_df
            .melt(id_vars='data')
            .assign(variable=lambda x: x.variable.str.replace('_norm', '').str.replace('_reg', '').str.replace('_', ' '))
        )


class RegionalCases(BaseCases):
    def __init__(self, df, const):
        super().__init__(df, is_region=False, const=const)
        self.regions = sorted(self.df.denominazione_regione.unique())
        self.n_regions = self.df.denominazione_regione.nunique()

    def _get_plt_multi_metrics_single_region_df(self, start_date, cols, region, renorm):
        if region is None:
            raise ValueError('Expected a region, got `None`')
        tmp_df = (
            self.df
            .query('denominazione_regione==@region')
            .query('data>=@start_date')
            [['data'] + cols]
        )
        if renorm:
            print('Region | Renormalize!')
            for c in cols:
                tmp_df[c] = tmp_df[c] / tmp_df[c].max()
        return (
            tmp_df
            .melt(id_vars='data')
            .assign(variable=lambda x: x.variable.str.replace('_norm', '').str.replace('_reg', '').str.replace('_', ' '))
        )

    def _get_plt_single_metric_all_regions_df(self, start_date, col):
        assert isinstance(col, str) and col in self.df.columns
        if not isinstance(col, str):
            col = col[0]
        return (
            self.df
            .query('data>=@start_date')
            [['data', 'denominazione_regione', col]]
            .rename(columns={'denominazione_regione': 'variable', col: 'value'})
        )

    def plot_small_multiples_single_metric_all_regions(self, start_date, metric, title, col_wrap=7, n_y_ticks=4):
        plt_df = self._get_plt_single_metric_all_regions_df(start_date, metric)
        g = self._plot_small_multiples(plt_df, start_date, metric, self.regions, self.n_regions, title, col_wrap=col_wrap)
        max_y = np.ceil(self.df.query('data>=@start_date')[metric].max())
        g.set(ylim=(-0.02, max_y), yticks=np.arange(0, max_y+.1, np.ceil(max_y/n_y_ticks)))
        return g


def extend_class(cls):
    """Source: https://gist.github.com/victorlei/5968685"""
    return lambda f: (setattr(cls,f.__name__,f) or f)

@extend_class(sns.FacetGrid)
def set_xdates(self, start, end, fmt='%B-%Y', freq='MS'):
    dt = pd.date_range(start, end, freq=freq)
    self.set(xticks=dt, xticklabels=dt.map(lambda x: x.strftime(fmt)))
    return self

@extend_class(sns.FacetGrid)
def set_suptitle(self, s, x=0, y=1.1, fontsize=18, ha='left'):
    self.fig.suptitle(s, x=x, y=y, ha=ha, fontweight='bold', fontsize=fontsize)
    return self

@extend_class(sns.FacetGrid)
def set_subtitle(self, s, x=0, y=1, c='darkgrey', fs=18, fw='bold', ha='left'):
    if isinstance(c, int):
        c = sns.color_palette('colorblind')[c]
    self.fig.text(x, y, s, color=c, fontsize=fs, fontweight=fw, ha=ha)

@extend_class(sns.FacetGrid)
def set_formatter(self, axis='xaxis', denom=1e6, fmt='{}'):
    if denom > 1:
        f = lambda x, pos: fmt.format(int(x/denom))
    else:
        f = lambda x, pos: fmt.format(x)
    for ax in self.axes.flat:
        getattr(ax, axis).set_major_formatter(FuncFormatter(f))
    return self

def get_pal(n):
    return sns.cubehelix_palette(n_colors=n, start=3, rot=.6, gamma=.6, hue=1, light=.8, dark=0)
