from Terminal import color

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
import itertools as it
import operator
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.naive_bayes import GaussianNB
from matplotlib import style
plt.style.use('grayscale')


list_1 = []
list_accuracy = []
list_auc = []
max_auc = []
max_acy = []
max_auc_f = []
max_acy_f = []
################################################################################################
file = open('data-good/GNB/GNB-2-vs-4.txt','a')
################################################################################################
#exp,trial,userjudgement,time_skew,time_kurtosis,time_mean,time_variance,t_sd,time_maxblock,timefortrial,gazedistAll,gazedistX,gazedistY,trans_mean,trans_var,trans_maxblock,trans_skew,trans_kurtosis,trans_sd,correctlabel
features_list = ['time_skew','time_kurtosis','time_mean','time_variance','t_sd','time_maxblock','timefortrial','gazedistAll','gazedistX',
                 'gazedistY','trans_mean','trans_var','trans_maxblock','trans_skew','trans_kurtosis','trans_sd']
features_combination = []
################################################################################################
print ('2 v 4')
################################################################################################
big_auc = 0;
for r in range(len(features_list)):
    list_auc[:] = []
    list_1[:] = []
    list_accuracy[:] = []
    features_combination = list(it.combinations(features_list,r))
    for feature in features_combination:
        df = pd.read_csv('output_updates.csv')
        df.drop(['exp'], 1, inplace=True)
        df.drop(['trial'], 1, inplace=True)
        df.drop(['userjudgement'], 1, inplace=True)
        ################################################################################################
        df['correctlabel'] = df['correctlabel'].map({2: -1, 4: 1})
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
        clf = GaussianNB()
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        Y_predict = clf.predict(X_test)
        fpr, tpr, threshold = metrics.roc_curve(y_test, Y_predict, pos_label=1)

        auc = metrics.auc(fpr, tpr)
        # code for ROC curve
        if auc > big_auc:
            big_fpr = fpr;
            big_tpr = tpr;
        # code ends for ROC curve
        list_accuracy.append(float(accuracy))
        list_auc.append(float(auc))
    index_acy, value_acy = max(enumerate(list_accuracy), key=operator.itemgetter(1))
    index_auc, value_auc = max(enumerate(list_auc), key=operator.itemgetter(1))
    max_auc.append(float(value_auc))
    max_acy.append(float(value_acy))
    max_auc_f.append(list_1[index_auc])
    max_acy_f.append(list_1[index_acy])


# code for ploting ROC curve
plt.plot(big_fpr,big_tpr,label = "GNB")
################################################################################################
plt.savefig("graphs-good/GNB/ROC-2-vs-4.png")
################################################################################################
plt.show()


# Code bellow is for ploting AUC and accuracy in the graph

ind_auc ,m_auc = max(enumerate(max_auc), key=operator.itemgetter(1))

print (str(ind_auc) + '   ' + str(len(max_auc)))

#a,b,c,d = zip([list(x) for x in zip(*sorted(zip(max_auc, max_acy,max_auc_f,max_acy_f), key=lambda pair: pair[0]))])
#a = [list(x) for x in a]
#b = [list(x) for x in b]
#c = [list(x) for x in c]
#d = [list(x) for x in d]


file.writelines(["".join(str(x) + '\n' + str(y) + '\n' + str(z) + '\n' + str(m) + '\n') for x,y,z,m in zip(max_auc, max_acy,max_auc_f,max_acy_f)])
fig = plt.figure()
################################################################################################
fig.canvas.set_window_title('GNB-2-vs-4')
################################################################################################
fig.set_size_inches(8,6)
ax = fig.add_subplot(111)

#plt.title("KNN Comparision AUC vs Accuracy")

#print (max_auc)
ax.set_ylim(0,1)
plt.xticks(range(len(max_auc)),range(len(max_auc)))

x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y= ['16','15','14','13','12','11','10','9','8','7','6','5','4','3','2','1']
plt.bar(x, list(max_auc), color = 'grey', label='AUC')

plt.xticks(range(len(max_auc)),y)


line1, = ax.plot(max_acy, label='accuracy', color = 'black')
#plt.xlabel('X Axis Number of Features')
#plt.ylabel('Y Axis')

ax.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
#ax.annotate('Max Value, Auc: ' + str(m_auc) + '\n features: ' + str(max_auc_f[ind_auc]), xy=(ind_auc + 0.05, m_auc),
#            xytext=(0, m_auc - 0.2), arrowprops=dict(facecolor='darkgray', shrink=0.05),
#           )

#manager = plt.get_current_fig_manager()
#manager.resize(*manager.window.maxsize())

################################################################################################
plt.savefig("graphs-good/GNB/GNB-2-vs-4.png")
################################################################################################
plt.show()
