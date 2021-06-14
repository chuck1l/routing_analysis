import pandas as pd
import numpy as np


def prepare_data(data):

    df = data.copy()

    if 'Unnamed: 0' in list(df.columns):
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Create new bins since fmv_buckets have na's
    bin_conditions = [
        df['miles'].between(0, 3),
        df['miles'].between(4, 6),
        df['miles'].between(7, 10),
        df['miles'] > 10
    ]
    bin_values = ['bin_one', 'bin_two', 'bin_three', 'bin_four']

    df['bin_labels'] = np.select(
        bin_conditions, bin_values)

    # Deal with na's in midpoint_target_market
    df['midpoint_target_market'] = df.groupby(
        ['call_center_state', 'bin_labels', 'los_code']
    )['cost_usd'].transform(lambda x: x.fillna(x.mean()))

    # Number of days ride is scheduled prior to ride date (advanced notice 2)
    df['date_ride'] = pd.to_datetime(df['date_ride'], format='%Y-%m-%d')
    df['ridecreate_datetime'] = pd.to_datetime(
        df['ridecreate_datetime'], format='%Y-%m-%d'
    )

    df['days_in_advance'] = (
        df['date_ride'] - df['ridecreate_datetime']
    ).dt.days

    # Create a month colummn to trend re-reroute percentage 
    df['month'] = df['date_ride'].dt.month
    
    mon_cond = [
        df['month'] == 1, df['month'] == 2, df['month'] == 3,
        df['month'] == 4, df['month'] == 5, df['month'] == 6,
        df['month'] == 7, df['month'] == 8, df['month'] == 9,
        df['month'] == 10, df['month'] == 11, df['month'] == 12,
    ]
    mon_vals = [
        'January', 'February', 'March', 'April', 'May',
        'June', 'July', 'August', 'September', 'October',
        'November', 'December'
    ]
    df['month'] = np.select(mon_cond, mon_vals)


    trend_df = df.copy()
    # Minimum of 28 days to be included in the count/average re-routes
    qualified_months = trend_df.groupby('month', as_index=False)['date_ride'].agg('nunique')
    month_mask = qualified_months['month'][qualified_months.date_ride >= 28].to_list()

    trend_df = trend_df[trend_df['month'].isin(month_mask)]
    # Return this dataframe as well, plot in the important features function
    trend_df = trend_df.groupby('month', as_index=False).agg(
        reroute_count=('reroute_24hrs', 'sum'),
        reroute_avg=('reroute_24hrs', 'mean')
    )

    df.drop(['date_ride', 'ridecreate_datetime', 'month'], axis=1, inplace=True)

    df['pu_county_code'] = df['pu_county_code'].astype(str)
    df['provider_code'] = df['provider_code'].astype(str)

    df_dummified = pd.get_dummies(df)

    return df_dummified, trend_df

if __name__ == '__main__':
    pass
    