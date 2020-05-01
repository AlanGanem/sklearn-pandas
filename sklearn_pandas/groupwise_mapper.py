from .dataframe_mapper import DataFrameMapper
from tqdm.notebook import tqdm
tqdm.pandas()


class GroupWiseTransformer:
    '''
    DataFrameMapper wrapper to perform transformations in a group-wise (pandas groupby) fashion
    '''

    def __init__(self):
        return

    def fit(self, df, feature_def, group_columns, **dfmapperargs):
        #define groupby object
        groupby_object = df.groupby(group_columns)
        #get group names
        groups = [g for g, _ in groupby_object]
        # create a dict mapping a DataFrameMapper instance to each group in groups
        scalers = {group: DataFrameMapper(feature_def, df_out=True, **dfmapperargs) for group in groups}

        # apply fit for each group
        for group in tqdm(scalers):
            group_df = groupby_object.get_group(group)
            scalers[group].fit(group_df)

        #fitted columns
        self.columns = [i[0][0] if i[0].__class__ == list else i[0] for i in scalers[group].features]
        self.group_columns = group_columns
        self.scalers = scalers
        return self

    def transform(self, df):
        #create copy to avoid inplace operations
        # TODO: create 'inplace' parameter
        df = df.copy()
        df.loc[:, self.columns] = self._apply_method(df[self.group_columns + self.columns], 'transform')[self.columns]
        return df

    def fit_transform(self, df, feature_def, group_columns, **dfmapperargs):
        self.fit(df, feature_def, group_columns, **dfmapperargs)
        return self.transform(df)

    def inverse_transform(self, df):
        # create copy to avoid inplace operations
        # TODO: create 'inplace' parameter
        df = df.copy()
        df.loc[:, self.columns] = self._apply_method(df[self.group_columns + self.columns], 'inverse_transform')[
            self.columns]
        return df

    def _apply_method(self, df, method):
        '''
        internal method to progress_apply transformer method under pandas groupby syntax
        :param df:
        :param method:
        :return:
        '''
        groupby_object = df.groupby(self.group_columns)
        return groupby_object.progress_apply(lambda x: self._robust_apply(x, method=method)).reset_index(drop=True)

    def _robust_apply(self, df, method, handler='ignore'):
        '''
        internal method to apply transformer method handling group exceptions
        :param df:
        :param method: Str describing the transformer method to apply ('Transform', for instance)
        :param handler: how to handle new groups (unseen in fit)
        :return: method applied df
        '''
        try:
            group = tuple([df[i][0] for i in self.group_columns])
            return getattr(self.scalers[group], method)(df)

        except KeyError: #in case a new group appears

            if handler == 'coerce':
                # return a NaN filled df
                df_copy = df.copy()
                df_copy.loc[:] = np.nan
                return df_copy

            elif handler == 'ignore':
                #return df with the same values
                return df

            elif handler == 'raise':
                raise

            else:
                raise ValueError(
                    'handler should be one of "coerce","ignore" or "raise"')