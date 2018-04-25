from Terminal import color
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools as it
import operator
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import style
import scikitplot as skplt
import operator
style.use('ggplot')


list_1 = []
list_accuracy = []
list_auc = []
max_auc = []
max_acy = []
max_auc_f = []
max_acy_f = []
#file = open('LDA.txt','a')
#exp,trial,userjudgement,time_skew,time_kurtosis,time_mean,time_variance,t_sd,time_maxblock,timefortrial,gazedistAll,gazedistX,gazedistY,trans_mean,trans_var,trans_maxblock,trans_skew,trans_kurtosis,trans_sd,correctlabel
#features_dict = {'time_skew': 0,'time_kurtosis': 0,'time_mean': 0,'t_sd': 0,'timefortrial': 0,'gazedistall': 0,'gazedistX': 0,
#                 'gazedistY': 0,'trans_mean': 0,'trans_maxblock': 0, 'trans_skew': 0,'trans_kurtosis': 0,'trans_sd': 0}
features_dict = {'trans_mean' : 0,'trans_sd' : 0,'trans_skew' : 0,'trans_kurtosis' : 0,'trans_maxblock' : 0,'gazedistall' : 0,
                'gazedistX' : 0,'gazedistY' : 0,'time_mean' : 0,'t_sd' : 0,'time_skew' : 0,'time_kurtosis' : 0,'timefortrial' : 0}
features_list = ['trans_mean','trans_sd','trans_skew','trans_kurtosis','trans_maxblock','gazedistall',
                'gazedistX','gazedistY','time_mean','t_sd','time_skew','time_kurtosis','timefortrial']
#features_list = ['time_skew','time_kurtosis','time_mean','t_sd','timefortrial','gazedistall','gazedistX',
#                 'gazedistY','trans_mean','trans_maxblock','trans_skew','trans_kurtosis','trans_sd']
features_combination = []
################################################################################################
print ('3 v 4')
################################################################################################
for r in range(len(features_dict)):
    list_auc[:] = []
    list_1[:] = []
    list_accuracy[:] = []
    features_combination = list(it.combinations(features_list,r))
    for feature in features_combination:
        df = pd.read_csv('final-output.csv')
        df.drop(['exp'], 1, inplace=True)
        df.drop(['trial'], 1, inplace=True)
        df.drop(['userjudgement'], 1, inplace=True)
        df.drop(['trans_var'], 1, inplace=True)
        df.drop(['time_variance'], 1, inplace=True)
        #df.drop(['trans_maxblock'], 1, inplace=True)
        ################################################################################################
        df['correctlabel'] = df['correctlabel'].map({3: 1, 4: -1})
        ################################################################################################
        df = df[np.isfinite(df['correctlabel'])]
        #print ([x for x in features_list if x not in list(feature)])
        list_1.append(",".join([x for x in features_list if x not in list(feature)]))
        df.drop(list(feature),1,inplace = True)
        X = np.array(df.drop(['correctlabel'], 1))
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        y = np.array(df['correctlabel'])

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)

        #accuracy = clf.score(X_test, y_test)
        Y_predict = clf.predict(X_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, Y_predict, pos_label=1)

        auc = metrics.auc(fpr, tpr)

        #list_accuracy.append(float(accuracy))
        list_auc.append(float(auc))
    #index_acy, value_acy = max(enumerate(list_accuracy), key=operator.itemgetter(1))
    index_auc, value_auc = max(enumerate(list_auc), key=operator.itemgetter(1))
    max_auc.append(float(value_auc))
    #max_acy.append(float(value_acy))
    max_auc_f.append(list_1[index_auc])
    #max_acy_f.append(list_1[index_acy])
    for ele in list_1[index_auc].split(','):
        features_dict[ele] += 1

ind_auc ,m_auc = max(enumerate(max_auc), key=operator.itemgetter(1))


print (str(max_auc_f[ind_auc]) + '   ' + str(len(max_auc)))

factor=1.0/sum(features_dict.itervalues())
for k in features_dict:
    features_dict[k] = features_dict[k]*factor
#print (features_dict)
threshold = 0.75 * max(features_dict.iteritems(), key=operator.itemgetter(1))[1];
x = []
thresh_list = []
for i in range(len(features_dict)):
    x.append(i)
    thresh_list.append(threshold)



non_accept_features = [x for x in features_dict if features_dict[x] < threshold]

fig = plt.figure()
ax = fig.add_subplot(111)

################################################################################################
fig.canvas.set_window_title('LDA-3-vs-4-BE')
################################################################################################
fig.set_size_inches(8,8)
# draw the bar for the features
y = ['1','2','3','4','5','6','7','8','9','10','11','12','13']
#plt.bar(y, list(features_dict.values()), align='center' , color = 'grey', )
plt.bar(range(len(features_dict)), features_dict.values(), align='center',color = 'grey')
plt.xticks(range(len(features_dict)), y)
plt.xlabel('Features')
plt.ylabel('Features Probability')
plt.title('Backward Elemination Graph')
#plt.xticks(range(len(features_dict)),y)
#draw threshold line
axis2 =  plt.plot( y, thresh_list, color='black', linestyle='-')
################################################################################################
plt.savefig("graph-final/LDA/LDA-3-vs-4-BE.png")
################################################################################################
#plt.xlabel('Feature Numbers ')
#plt.ylabel('Probability of the feature to be selected')
#plt.show()





df1 = pd.read_csv('final-output.csv')
df1.drop(['exp'], 1, inplace=True)
df1.drop(['trial'], 1, inplace=True)
df1.drop(['userjudgement'], 1, inplace=True)
df1.drop(['trans_var'], 1, inplace=True)
df1.drop(['time_variance'], 1, inplace=True)
################################################################################################
df1['correctlabel'] = df1['correctlabel'].map({3: 1, 4: -1})
################################################################################################
df1 = df1[np.isfinite(df1['correctlabel'])]
#print ([x for x in features_list if x not in list(feature)])
df1.drop(list(non_accept_features),1,inplace = True)
X = np.array(df1.drop(['correctlabel'], 1))
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
y = np.array(df1['correctlabel'])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
Y_predict = clf.predict_proba(X_test)
#fpr, tpr, threshold = metrics.roc_curve(y_test, Y_predict, pos_label=1)
#auc = metrics.auc(fpr, tpr)

fig = plt.figure()
axis = fig.add_subplot(111)

axis.set_ylim(0,1)
################################################################################################
fig.canvas.set_window_title('LDA-3-vs-4-ROC')
################################################################################################
fig.set_size_inches(8,8)
#plt.plot(fpr, tpr,label = "Accuracy: " + str(accuracy) + '\n' + "AUC: " + str(auc),color='black')
skplt.metrics.plot_roc_curve(y_test, Y_predict,title = 'ROC Curve\n' + 'Accuracy: ' + str(accuracy) +
                                                       '\n 3: 1 & 4: -1')
#axis.legend(label = "Accuracy: " + str(accuracy) + '\n' + "AUC: " + str(auc),
#                              color='black' )
################################################################################################
plt.savefig("graph-final/LDA/LDA-3-vs-4-ROC.png")
################################################################################################
