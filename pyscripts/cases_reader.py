import pandas as pd


__all__ = ['NationalCasesReader', 'RegionalCasesReader']


class CasesReader():
    REG_DEMOGRAPHY_FN = '../../COVID-19/dati-statistici-riferimento/popolazione-istat-regione-range.csv'
    DROP_COLS = [
        'stato', 'note', 'note_test', 'note_casi', 'casi_da_screening',
        'casi_da_sospetto_diagnostico', 'variazione_totale_positivi',
        'dimessi_guariti', 'totale_casi', 'totale_positivi_test_molecolare',
        'totale_positivi_test_antigenico_rapido', 'tamponi_test_molecolare',
        'tamponi_test_antigenico_rapido', 'tamponi', 'casi_testati',
        'totale_positivi', 'isolamento_domiciliare', 'terapia_intensiva'
    ]
    MAX_COLS = [
        'ricoverati_con_sintomi', 'totale_ospedalizzati', 'nuovi_positivi',
        'deceduti', 'deceduti_2w', 'ingressi_terapia_intensiva'
    ]

    def __init__(self, fn, is_region, rolling=7, drop_col=None, max_cols=None):
        self.fn = fn
        self.is_region = is_region
        self.rolling = rolling
        self.rolling_mean_f = lambda x: x.rolling(rolling, 1).mean()
        self.drop_col = drop_col or self.DROP_COLS
        self.max_cols = max_cols or self.MAX_COLS

    def run_pipeline(self):
        self._read_regional_demography()
        self._read_base()
        self._build_smooth()
        self._build_death_2w_lag()
        self._join_deaths_2w()
        self._build_max_to_normalize()
        self._build_normalized_metrics()
        self._post_proc()
        return self

    def _read_regional_demography(self):
        self.reg_demography_df = pd.read_csv(self.REG_DEMOGRAPHY_FN)
        if not self.is_region:
            self.reg_demography_df = (
                self.reg_demography_df
                .assign(denominazione_regione='ita')
            )
        self.reg_demography_df = (
            self.reg_demography_df
            .groupby('denominazione_regione')
            [['totale_generale']]
            .sum()
            .reset_index()
        )
        return self

    def __get_pct_change_f(self, col):
        return lambda x: x.sort_values('data').groupby('denominazione_regione')[col].pct_change(7)

    def __get_smooth_f(self, col):
        return lambda df: df.groupby(['denominazione_regione'])[col].transform(self.rolling_mean_f)

    def _read_base(self):
        self.cases_df = (
            pd.read_csv(self.fn)
            .assign(data=lambda x: x.data.str[:len('2020-02-24')])
            .assign(data=lambda x: pd.to_datetime(x.data))
            .drop(self.drop_col, axis=1)
        )
        if not self.is_region:
            self.cases_df = self.cases_df.assign(denominazione_regione='ita')

        self.cases_df = (
            self.cases_df
            .assign(deceduti=lambda df: df.groupby(['denominazione_regione']).deceduti.diff(1))
            .assign(pct_diff_positive=self.__get_pct_change_f('nuovi_positivi'))
            .assign(pct_diff_deceduti=self.__get_pct_change_f('deceduti'))
        )
        return self

    def _build_smooth(self):
        self.cases_df = (
            self.cases_df
            # Get population per region
            .merge(self.reg_demography_df, on='denominazione_regione', how='inner')
            # Normalize daily-positive per region (*100k)
            .assign(nuovi_positivi_reg=lambda x: x.nuovi_positivi * 100_000 / x.totale_generale)
            .assign(deceduti_reg=lambda x: x.deceduti * 100_000 / x.totale_generale)
            .assign(ingressi_terapia_intensiva_reg=lambda x: x.ingressi_terapia_intensiva * 100_000 / x.totale_generale)
            .assign(ricoverati_con_sintomi_reg=lambda x: x.ricoverati_con_sintomi * 100_000 / x.totale_generale)
            .assign(totale_ospedalizzati_reg=lambda x: x.totale_ospedalizzati * 100_000 / x.totale_generale)
            # 7-days moving average
            # Overall
            .assign(nuovi_positivi=self.__get_smooth_f('nuovi_positivi'))
            .assign(deceduti=self.__get_smooth_f('deceduti'))
            .assign(ingressi_terapia_intensiva=self.__get_smooth_f('ingressi_terapia_intensiva'))
            .assign(ricoverati_con_sintomi=self.__get_smooth_f('ricoverati_con_sintomi'))
            .assign(totale_ospedalizzati=self.__get_smooth_f('totale_ospedalizzati'))
            # normalized out of 100_000
            .assign(nuovi_positivi_reg=self.__get_smooth_f('nuovi_positivi_reg'))
            .assign(deceduti_reg=self.__get_smooth_f('deceduti_reg'))
            .assign(ingressi_terapia_intensiva_reg=self.__get_smooth_f('ingressi_terapia_intensiva_reg'))
            .assign(ricoverati_con_sintomi_reg=self.__get_smooth_f('ricoverati_con_sintomi_reg'))
            .assign(totale_ospedalizzati_reg=self.__get_smooth_f('totale_ospedalizzati_reg'))
            .drop(['totale_generale'], axis=1)
        )
        return self

    def _build_death_2w_lag(self):
        self.deaths_2w_lag_df = (
            self.cases_df[['data', 'denominazione_regione', 'deceduti']]
            .assign(data_2w=lambda x: x.data - pd.DateOffset(weeks=2))
            .drop('data', axis=1)
            .rename(columns={'data_2w': 'data', 'deceduti': 'deceduti_2w'})
        )
        return self

    def _join_deaths_2w(self):
        self.cases_df = (
            self.cases_df
            .merge(self.deaths_2w_lag_df, on=['data', 'denominazione_regione'], how='left')
        )

    def _build_max_to_normalize(self):
        rename_dct = dict(
            zip(
                self.max_cols,
                map(lambda x: x + '_max', self.max_cols)
            )
        )
        self.max_per_region_df = (
            self.cases_df
            .groupby('denominazione_regione')
            [self.max_cols]
            .max()
            .reset_index()
            .rename(columns=rename_dct)
        )
        return self

    def _build_normalized_metrics(self):
        self.cases_df = (
            self.cases_df
            .merge(self.max_per_region_df, on='denominazione_regione', how='inner')
        )
        for col in self.max_cols:
            self.cases_df = (
                self.cases_df
                .assign(**{col + '_norm': lambda x: x[col] / x[col + '_max']})
            )
        self.cases_df = (
            self.cases_df
            .drop(map(lambda x: x + '_max', self.max_cols), axis=1)
            .sort_values(['data', 'denominazione_regione'])
            .reset_index(drop=True)
        )
        return self

    def _post_proc(self):
        if not self.is_region:
            self.cases_df = self.cases_df.drop('denominazione_regione', axis=1)
        return self


class NationalCasesReader(CasesReader):
    NATIONAL_CASES_FN = '../../COVID-19/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'

    def __init__(self, rolling=7, drop_col=None, max_cols=None):
        super().__init__(
            self.NATIONAL_CASES_FN, False,
            rolling=rolling, drop_col=drop_col,
            max_cols=max_cols
        )


class RegionalCasesReader(CasesReader):
    REGIONAL_CASES_FN = '../../COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv'

    def __init__(self, rolling=7, drop_col=None, max_cols=None):
        super().__init__(
            self.REGIONAL_CASES_FN, True,
            rolling=rolling, drop_col=drop_col,
            max_cols=max_cols
        )
