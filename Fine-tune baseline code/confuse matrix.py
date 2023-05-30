import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,('experience','knowledge','education','project','others'))
    plt.yticks(tick_marks,('experience','knowledge','education','project','others'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.savefig('test_bert.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()

# seed = 42
# np.random.seed(seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
# print("Accuracy of cross validation, mean %.2f, std %.2f\n" % (result.mean(), result.std()))

# 显示混淆矩阵
def plot_confuse(model,data):
    all_preds = np.array([])
    all_true = np.array([])
    for x_true, y_true in data:

        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        
        all_preds = np.concatenate((all_preds, y_pred))
        all_true = np.concatenate((all_true, y_true))
    conf_mat = confusion_matrix(all_true, all_preds)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(y_true)+1))