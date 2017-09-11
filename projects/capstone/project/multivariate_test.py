import pandas as pd
# For undersampling
from imblearn.under_sampling import RandomUnderSampler
from multivariate_util import *
# helping to remove outliers
from scipy.stats import iqr

feats = ['V15', 'V18', 'V11', 'V13', 'V8', 'Class']

df = pd.read_csv('./creditcard.csv')
df = df[feats]

# separating the predictors and the labels
X, y = df.iloc[:,:-1], df.iloc[:,-1]

legit, fraud = X[df.Class==0].copy(), X[df.Class==1].copy()

for idx, feat in enumerate(legit.columns):
    q75, q25 = np.percentile(legit[feat], [75, 25])
    iqr_ = iqr(legit[feat]) * 1.5

    greater = np.array(legit[feat] <= q25 - iqr_, dtype=bool)
    legit.loc[greater, feat] = np.nan

    lower = np.array(legit[feat] >= q75 + iqr_, dtype=bool)
    legit.loc[lower, feat] = np.nan
legit = pd.concat([legit, fraud], axis=0)
legit = pd.concat([legit, y], axis=1)
legit = legit.dropna()
# checking if there is a missing value
print legit.isnull().sum()
print '\n\n'

# separating the predictors and the labels
X, y = legit.iloc[:,:-1], legit.iloc[:,-1]

rus = RandomUnderSampler(ratio={0:246*60, 1:246}, random_state=0, return_indices=True)
X_resampled, y_resampled, idxes = rus.fit_sample(X, y)

t_clf = MultivariateTMixture(n_components=4, random_state=0)
t_clf.fit(X_resampled)

t_clf.iterate(X_resampled)
t_clf.iterate(X_resampled)
t_clf.iterate(X_resampled)
t_clf.iterate(X_resampled)
t_clf.iterate(X_resampled)

print 'finished'
