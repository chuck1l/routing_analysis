import pandas as pd
import numpy as np
import os
from feature_engineering import prepare_data
from parameter_search import source_best_params
from sns_plot import create_cm_plot
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                'axes.titlepad': 20})
plt.style.use('ggplot')
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def feature_importance(data, state):

    df = data.copy()
    df, reroute_trends = prepare_data(df)
    print(f'{state} data are prepared, starting feature analysis.')

    # Create a directory to save market level resutls
    path = f'../img/{state}'
    if not os.path.isdir(path):
        os.mkdir(path)

    cols = df.columns
    df[cols] = df[cols].astype(float).round(2)

    print(df.shape, f'{state}:', df['reroute_24hrs'].mean())
    df = df.dropna()
    print(df.shape, f'{state}:', df['reroute_24hrs'].mean())

    # Create the X and y train and test
    y = df['reroute_24hrs']
    X = df.drop('reroute_24hrs', axis=1).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

    #best_params = source_best_params(X_train, y_train)

    labels = pd.Series(X_train.columns, name='features')

    xgb_model = XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1,
        eta=0.015, 
        n_estimators=600, 
        max_depth=20, 
        min_child_weight=0.5, 
        gamma=0.7, 
        subsample=0.9, 
        colsample_bytree=0.8, 
        reg_alpha=1e-05, 
        reg_lambda=0.13 
    )

    xgb_model.fit(X_train, y_train)
    y_hat = xgb_model.predict(X_test)

    # Plot the confusion matrix for results
    print(f'Plotting results for {state}')

    confusion = metrics.confusion_matrix(y_test, y_hat)

    create_cm_plot(confusion, state)

    importances = pd.Series(xgb_model.feature_importances_, name='Feature Importance')
    features_df = pd.concat([labels, importances], axis=1)
    features_df = features_df.sort_values(by='Feature Importance', ascending=False)
    features_df.reset_index(drop=True, inplace=True)
    features_df['feature_count'] = np.arange(1, features_df.shape[0]+1)
    features_df['importance_cumsum'] = features_df['Feature Importance'].values.cumsum()

    num_of_features = (abs(features_df['importance_cumsum'] - 0.80).idxmin()) + 1

    features_df.plot(x='feature_count', y='importance_cumsum')
    plt.title(f'Cumulative Sum of Feature Importance "{state}"')
    plt.xlabel('Number of Features', fontsize=13, c='k')
    plt.ylabel('Sum (Approaching 1 or 100%)', fontsize=13, c='k')
    plt.tight_layout()
    plt.legend('')
    plt.hlines(.8, xmin=0, xmax=num_of_features, colors='b', linestyles='dashed')
    plt.vlines(num_of_features, ymin=0, ymax=.8, colors='b', linestyles='dashed', label=num_of_features)
    plt.savefig(f'../img/{state}/{state}_cumsum_plot.png', dpi=500, orientation='landscape');

    # Keep only the top 80%, if they're greater than 1% contribution
    sum_mask = (features_df['importance_cumsum'] <= .80) & (
        features_df['Feature Importance'] >= .01)

    features_top = features_df[sum_mask]
    feat_col_names = features_top['features'].to_list()

    ft_total_avg = df[feat_col_names].mean(axis=0).reset_index()
    ft_total_avg.rename(columns={'index': 'features', 0: 'total_avg'}, inplace=True)
    ft_reroute_avg = df[feat_col_names][df['reroute_24hrs'] == 1].mean(axis=0).reset_index()
    ft_reroute_avg.rename(columns={'index': 'features', 0: 'reroute_avg'}, inplace=True)
    ft_correlation = df.corr()
    ft_correlation = ft_correlation.loc[feat_col_names, 'reroute_24hrs']
    ft_correlation = ft_correlation.reset_index() 
    ft_correlation.rename(columns={'index': 'features', 'reroute_24hrs': 'reroute_corr'}, inplace=True)

    top_feature_analysis = pd.merge(features_top, ft_total_avg, how='left', on='features')
    top_feature_analysis = pd.merge(top_feature_analysis, ft_reroute_avg, how='left', on='features')
    top_feature_analysis = pd.merge(top_feature_analysis, ft_correlation, how='left', on='features')

    round_cols = list(top_feature_analysis.columns)
    
    round_cols.remove('features')
    top_feature_analysis[round_cols] = top_feature_analysis[round_cols].round(2)

    top_feature_analysis['features'] = top_feature_analysis['features'].str.replace('_', ' ')

    # Plot feature importance - save figure
    top_feature_analysis.sort_values(by='Feature Importance', ascending=True).plot.barh(
        x='features', y='Feature Importance')
    plt.title(f'Top Features for "{state}"')
    plt.xlabel('Feature Importance Score', fontsize=13, c='k')
    plt.ylabel('Features', fontsize=13, c='k')
    plt.tight_layout()
    plt.legend('')
    plt.savefig(f'../img/{state}/{state}_feature_importance.png', dpi=500, orientation='landscape');
    # Plot ratios for total data, and reroute occurances - save figure
    top_feature_analysis.sort_values(by='Feature Importance', ascending=True)[
        ['features', 'total_avg', 'reroute_avg']].plot.barh(x='features', stacked=False)
    plt.title(f'Ratios Total vs Re-Routes "{state}"')
    plt.xlabel('Percent Occurance in Data', fontsize=13, c='k')
    plt.ylabel('Features', fontsize=13, c='k')
    plt.tight_layout()
    plt.savefig(f'../img/{state}/{state}_feature_ratios.png', dpi=500, orientation='landscape');
    # Plot the correlation between top features and target
    top_feature_analysis.sort_values(by='Feature Importance', ascending=True)[
        ['features', 'reroute_corr']].plot.barh(x='features', y='reroute_corr')
    plt.title(f'Feature Correlation "{state}"')
    plt.xlabel('Correlation', fontsize=13, c='k')
    plt.ylabel('Features', fontsize=13, c='k')
    plt.tight_layout()
    plt.legend('')
    plt.savefig(f'../img/{state}/{state}_feature_correlation.png', dpi=500, orientation='landscape');

    # Plot the re-reoute average and raw count
    reroute_trends.set_index('month', inplace=True)
    # Set fonts
    title_fonts = {
        'color': 'darkblue',
        'weight': 'normal', 
        'size': 13
    }
    label_fonts = {
        'color': 'darkred',
        'weight': 'normal', 
        'size': 12
    }
    fig = plt.figure()
    subplot_avg = fig.add_subplot(1,2,1)
    reroute_trends['reroute_avg'].plot(ax=subplot_avg, grid=True)
    subplot_avg.set_xlabel('Month of Year', fontdict=label_fonts, labelpad=10)
    subplot_avg.set_ylabel(f'{state} Re-Route Average', fontdict=label_fonts)
    subplot_avg.set_title(f'{state} Re-Route Average By Month', fontdict=title_fonts)

    subplot_tot = fig.add_subplot(1,2,2)
    reroute_trends['reroute_count'].plot(ax=subplot_tot, grid=True)
    subplot_tot.set_xlabel('Month of Year', fontdict=label_fonts, labelpad=10)
    subplot_tot.set_ylabel(f'{state} Re-Route Count', fontdict=label_fonts)
    subplot_tot.set_title(f'{state} Re-Route Count By Month', fontdict=title_fonts)
    plt.savefig(f'../img/{state}_reroute_avg_cnt.png', dpi=500, orientation='landscape');
    
    return None


if __name__ == '__main__':
    state = 'OK'
    feature_importance(state)