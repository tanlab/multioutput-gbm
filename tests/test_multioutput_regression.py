import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pygbm import GradientBoostingRegressor


def test_atp1d():
    df = pd.read_csv('atp1d.csv')
    target = df.loc[:, df.columns.str.startswith('LBL')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42, shuffle=True)
    gb = GradientBoostingRegressor(
        l2_regularization=0.003391634274257872,
        min_samples_leaf=10,
        learning_rate=0.1088115324113492,
        max_iter=199,
        n_iter_no_change=20
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    scalery = StandardScaler()
    scalery.fit(y_train)
    y_Train = scalery.transform(y_train)
    y_Test = scalery.transform(y_test)
    gb.fit(train_scaled, y_Train)
    y_preds = gb.predict_multi(test_scaled)
    r2 = r2_score(y_Test, y_preds, multioutput='uniform_average')
    print(r2)

def test_atp7d():
    df = pd.read_csv('atp7d.csv')
    target = df.loc[:, df.columns.str.startswith('LBL')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42, shuffle=True)
    gb = GradientBoostingRegressor(
        l2_regularization=0.880826520747869,
        min_samples_leaf=12,
        learning_rate=0.22445307581959334,
        max_iter=279,
        n_iter_no_change=23,
    )
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # train_scaled = scaler.transform(X_train)
    # test_scaled = scaler.transform(X_test)
    # scalery = StandardScaler()
    # scalery.fit(y_train)
    # y_Train = scalery.transform(y_train)
    # y_Test = scalery.transform(y_test)
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test)
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)


def test_edm():
    df = pd.read_csv('edm.csv')
    target = df.loc[:, ['DFlow', 'DGap']]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42, shuffle=True)
    gb = GradientBoostingRegressor(
        l2_regularization=0.880826520747869,
        min_samples_leaf=12,
        learning_rate=0.22445307581959334,
        max_iter=279,
        n_iter_no_change=23,
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    scalery = StandardScaler()
    scalery.fit(y_train)
    y_Train = scalery.transform(y_train)
    y_Test = scalery.transform(y_test)
    gb.fit(train_scaled, y_Train)
    y_preds = gb.predict_multi(test_scaled)
    r2 = r2_score(y_Test, y_preds, multioutput='uniform_average')
    print(r2)


def test_scm1d():
    df = pd.read_csv('scm1d.csv')
    target = df.loc[:, df.columns.str.contains('L')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=1658.0/8145.0, random_state=42, shuffle=True)
    gb = GradientBoostingRegressor(
        l2_regularization=0.07054193143238725,
        min_samples_leaf=23,
        learning_rate=0.12336530854190006,
        max_iter=1999,
        n_iter_no_change=None,
    )
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # train_scaled = scaler.transform(X_train)
    # test_scaled = scaler.transform(X_test)
    # scalery = StandardScaler()
    # scalery.fit(y_train)
    # y_Train = scalery.transform(y_train)
    # y_Test = scalery.transform(y_test)
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test)
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)


def test_scm20d():
    df = pd.read_csv('scm20d.csv')
    target = df.loc[:, df.columns.str.contains('L')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    c=1503.0/7463.0
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=c, random_state=42, shuffle=True)
    gb = GradientBoostingRegressor(
        l2_regularization=0.8640187696889217,
        min_samples_leaf=19,
        learning_rate=0.1164232801613771,
        max_iter=1998,
        n_iter_no_change=None,
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    scalery = StandardScaler()
    scalery.fit(y_train)
    y_Train = scalery.transform(y_train)
    y_Test = scalery.transform(y_test)
    gb.fit(train_scaled, y_Train)
    y_preds = gb.predict_multi(test_scaled)
    r2 = r2_score(y_Test, y_preds, multioutput='uniform_average')
    print(r2)


def test_wq():
    df = pd.read_csv('water-quality.csv')
    target = df.loc[:, df.columns.str.startswith('x')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42, shuffle=True)
    gb = GradientBoostingRegressor(
        l2_regularization=0.07509314619453317,
        min_samples_leaf=15,
        learning_rate=0.01948991297099692,
        max_iter=300,
        n_iter_no_change=17
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    scalery = StandardScaler()
    scalery.fit(y_train)
    y_Train = scalery.transform(y_train)
    y_Test = scalery.transform(y_test)
    gb.fit(train_scaled, y_Train)
    y_preds = gb.predict_multi(test_scaled)
    r2 = r2_score(y_Test, y_preds, multioutput='uniform_average')
    print(r2)
