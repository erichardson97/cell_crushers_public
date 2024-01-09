performance = {'repeat':[], 'fold':[], 'score':[], 'baseline':[]}
for repeat in glob('/mnt/bioadhoc/Users/erichard/cell_crushers/data/cv_folds/*'):
    folds = pd.read_pickle(repeat)
    for fold in folds:
        train = folds[fold]['Train']#[feature_list]
        test = folds[fold]['Test']#[feature_list]
        train = ds.data.iloc[train].copy()[feature_list+['Target']]
        train[train.columns] = StandardScaler().fit_transform(train)
        test = ds.data.iloc[test].copy()[feature_list+['Target']]
        test[test.columns] = StandardScaler().fit_transform(test)
        baseline = spearmanr(test['Titre_IgG_PT'],test["Target"])[0]
        model = EnsembleModel(RandomForestRegressor, {'n_estimators':100},feature_groups, models, model_kwargs, feature_order,coef='feature_importances_')
        model.fit(train[feature_order].values, train['Target'].values)
        out = model.predict(test[feature_order].values)
        score = spearmanr(out,test['Target'])[0]
        performance['repeat'].append(repeat)
        performance['score'].append(score)
        performance['fold'].append(fold)
        performance['baseline'].append(baseline)
        print(score,baseline)
pd.DataFrame(performance)
