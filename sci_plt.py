from matplotlib import pyplot as plt
import numpy as np
from wordcloud import WordCloud
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns

# paper style
def plot_hyperparameter_tuning(grid):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 获取参数值并确保它们是数值类型
    param_values = grid.cv_results_['param_C']
    if hasattr(param_values, 'data'):
        param_values = param_values.data
    param_values = np.array([float(v) for v in param_values])
    
    train_scores = np.array(grid.cv_results_['mean_train_score'])
    val_scores = np.array(grid.cv_results_['mean_test_score'])
    train_std = np.array(grid.cv_results_['std_train_score'])
    val_std = np.array(grid.cv_results_['std_test_score'])
    
    # 左图：折线图
    ax1.fill_between(param_values, 
                    train_scores - train_std, 
                    train_scores + train_std, 
                    alpha=0.3, color='white')
    ax1.fill_between(param_values, 
                    val_scores - val_std, 
                    val_scores + val_std, 
                    alpha=0.3, color='lightgray')
    
    ax1.plot(param_values, train_scores, label='Train Score', 
             color='black', marker='o', linewidth=2)
    ax1.plot(param_values, val_scores, label='Validation Score', 
             color='darkgray', marker='s', linewidth=2)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization Parameter (C)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Cross-validation Performance', fontsize=14)
    ax1.legend(frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 右图：热力图
    cv_scores = np.array(grid.cv_results_['split0_test_score'])
    for i in range(1, 5):
        cv_scores = np.vstack((cv_scores, 
                             grid.cv_results_[f'split{i}_test_score']))
    
    im = ax2.imshow(cv_scores, aspect='auto', cmap='Greys', 
                    vmin=np.min(cv_scores)-0.05, vmax=np.max(cv_scores)+0.05)
    plt.colorbar(im, ax=ax2)
    
    ax2.set_xticks(range(len(param_values)))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels([f'{v:.3f}' for v in param_values])
    ax2.set_yticklabels([f'Fold {i+1}' for i in range(5)])
    
    # 添加数值标注
    for i in range(cv_scores.shape[0]):
        for j in range(cv_scores.shape[1]):
            val = cv_scores[i, j]
            val_norm = (val - np.min(cv_scores)) / (np.max(cv_scores) - np.min(cv_scores))
            text_color = 'white' if val_norm < 0.5 else 'black'
            ax2.text(j, i, f'{val:.3f}',
                    ha='center', va='center',
                    color=text_color,
                    fontweight='bold')
    
    ax2.set_xlabel('Regularization Parameter (C)', fontsize=12)
    ax2.set_title('Cross-validation Scores by Fold', fontsize=14)
    
    plt.tight_layout()
    plt.show()

def plot_word_importance(vectorizer, X_train, n_words=15):
    # 计算TF-IDF最大值
    max_value = X_train.max(axis=0).toarray().ravel()
    
    # 获取特征名称（适配新版本 sklearn）
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
    except AttributeError:
        try:
            feature_names = np.array(vectorizer.get_feature_names())
        except AttributeError:
            print("无法获取特征名称，请检查 sklearn 版本")
            return
    
    # 获取排序后的索引
    sorted_by_tfidf = max_value.argsort()
    
    # 选择最高和最低的n个词
    top_indices = sorted_by_tfidf[-n_words:]
    bottom_indices = sorted_by_tfidf[:n_words]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)
    
    # 上半部分：最重要的词
    ax1 = fig.add_subplot(gs[0])
    y_pos = np.arange(n_words)
    
    # 绘制水平条形图
    bars1 = ax1.barh(y_pos, max_value[top_indices], 
                     color='black', alpha=0.8)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', 
                ha='left', va='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', 
                         alpha=0.7, pad=2))
    
    # 设置标签
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names[top_indices], fontsize=12)
    ax1.set_title('Most Important Terms', fontsize=14, pad=20)
    ax1.set_xlabel('TF-IDF Score', fontsize=12)
    
    # 添加网格线
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # 下半部分：最不重要的词
    ax2 = fig.add_subplot(gs[1])
    
    # 绘制水平条形图
    bars2 = ax2.barh(y_pos, max_value[bottom_indices], 
                     color='gray', alpha=0.6)
    
    # 添加数值标签
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', 
                ha='left', va='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', 
                         alpha=0.7, pad=2))
    
    # 设置标签
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names[bottom_indices], fontsize=12)
    ax2.set_title('Least Important Terms', fontsize=14, pad=20)
    ax2.set_xlabel('TF-IDF Score', fontsize=12)
    
    # 添加网格线
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    
    # 整体标题
    plt.suptitle('Term Importance Analysis based on TF-IDF Scores', 
                fontsize=16, y=1.02)
    
    # 添加注释
    fig.text(0.99, 0.01, 
             'Note: TF-IDF scores indicate term specificity and importance in the corpus',
             fontsize=10, style='italic', ha='right', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_word_cloud(vectorizer, max_value, n_words=100):
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        try:
            feature_names = vectorizer.get_feature_names()
        except AttributeError:
            print("无法获取特征名称，请检查 sklearn 版本")
            return
    
    # 创建词频字典
    word_importance = dict(zip(feature_names, max_value))
    
    # 选择前n个最重要的词
    sorted_words = dict(sorted(word_importance.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:n_words])
    
    # 创建词云
    plt.figure(figsize=(15, 8))
    
    wordcloud = WordCloud(
        background_color='white',
        width=1600,
        height=800,
        font_path='C:/Windows/Fonts/arial.ttf',  # 使用系统字体
        max_words=n_words,
        prefer_horizontal=0.7,  # 70%的词水平显示
        scale=3,  # 提高清晰度
        colormap='Greys',  # 使用黑白色调
        relative_scaling=0.5  # 调整词大小与权重的关系
    ).generate_from_frequencies(sorted_words)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # 添加标题
    plt.title('Term Importance Word Cloud (Based on TF-IDF Scores)', 
             fontsize=16, pad=20)
    
    # 添加注释
    plt.figtext(0.99, 0.01, 
                'Note: Word size indicates TF-IDF importance',
                fontsize=10, style='italic', ha='right')
    
    plt.tight_layout()
    plt.show()

def plot_logistic_importance(model, vectorizer, n_features=20):
    # 获取特征名称
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()
    
    # 获取系数
    coef = model.coef_[0]
    
    # 创建特征重要性对
    feature_importance = list(zip(feature_names, coef))
    
    # 按绝对值大小排序
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # 选择前n个最重要的特征
    top_features = feature_importance[:n_features]
    
    # 分离特征名和系数
    names = [x[0] for x in top_features]
    values = [x[1] for x in top_features]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建水平条形图
    colors = ['darkred' if x < 0 else 'darkblue' for x in values]
    y_pos = np.arange(len(names))
    
    plt.barh(y_pos, values, color=colors, alpha=0.8)
    
    # 添加特征名称
    plt.yticks(y_pos, names)
    
    # 添加零线
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 设置标题和标签
    plt.title('Top Feature Importance in Logistic Regression', fontsize=14, pad=20)
    plt.xlabel('Coefficient Value', fontsize=12)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加注释
    plt.figtext(0.99, 0.01,
                'Note: Blue positive, red negative',
                fontsize=10, style='italic', ha='right')
    
    # 为每个条形添加数值标签
    for i, v in enumerate(values):
        if v < 0:
            ha = 'right'
            x = v - 0.01
        else:
            ha = 'left'
            x = v + 0.01
        plt.text(x, i, f'{v:.3f}', 
                va='center', ha=ha,
                fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_logistic_wordcloud(model, vectorizer, n_words=100):
    # 获取特征名称
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()
    
    # 获取系数
    coef = model.coef_[0]
    
    # 创建词频字典（使用系数的绝对值）
    word_importance = dict(zip(feature_names, np.abs(coef)))
    
    # 创建正负相关的颜色映射
    color_func = lambda *args, **kwargs: 'darkred' if coef[feature_names.tolist().index(args[0])] < 0 else 'darkblue'
    
    # 选择前n个最重要的词
    sorted_words = dict(sorted(word_importance.items(), 
                             key=lambda x: abs(x[1]), 
                             reverse=True)[:n_words])
    
    # 创建词云
    plt.figure(figsize=(15, 8))
    
    wordcloud = WordCloud(
        background_color='white',
        width=1600,
        height=800,
        max_words=n_words,
        prefer_horizontal=0.7,  # 70%的词水平显示
        scale=3,  # 提高清晰度
        color_func=color_func,  # 使用自定义颜色函数
        relative_scaling=0.5  # 调整词大小与权重的关系
    ).generate_from_frequencies(sorted_words)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # 添加标题
    plt.title('Logistic Regression Coefficients Word Cloud', 
             fontsize=16, pad=20)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkblue', label='Positive Correlation'),
        Patch(facecolor='darkred', label='Negative Correlation')
    ]
    plt.legend(handles=legend_elements, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.05),
              ncol=2)
    
    # 添加注释
    plt.figtext(0.99, 0.01, 
                'Note: Word size indicates importance (absolute coefficient value)',
                fontsize=10, style='italic', ha='right')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys')
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_prob):
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='black', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_prob):
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='black', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, pad=20)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return pr_auc

def plot_performance_metrics(metrics_dict):
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    
    metrics_text = f"""
    Performance Metrics:
    
    Training Accuracy: {metrics_dict['train_accuracy']:.3f}
    Testing Accuracy:  {metrics_dict['test_accuracy']:.3f}
    
    Precision: {metrics_dict['precision']:.3f}
    Recall:    {metrics_dict['recall']:.3f}
    F1 Score:  {metrics_dict['f1']:.3f}
    
    ROC AUC:   {metrics_dict['roc_auc']:.3f}
    PR AUC:    {metrics_dict['pr_auc']:.3f}
    """
    plt.text(0.1, 0.1, metrics_text, fontsize=12, family='monospace')
    plt.title('Model Performance Summary', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def evaluate_logistic_regression(model, X_train, X_test, y_train, y_test):
    # 获取预测结果
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 获取预测概率
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # 计算各种指标
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
    
    # 绘制各个图表
    print("\n1. Confusion Matrix:")
    plot_confusion_matrix(y_test, y_test_pred)
    
    print("\n2. ROC Curve:")
    roc_auc = plot_roc_curve(y_test, y_test_prob)
    
    print("\n3. Precision-Recall Curve:")
    pr_auc = plot_precision_recall_curve(y_test, y_test_prob)
    
    # 创建指标字典
    metrics_dict = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    print("\n4. Performance Metrics Summary:")
    plot_performance_metrics(metrics_dict)
    
    # 打印详细分类报告
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return metrics_dict

def plot_history(history):
    # 设置科研风格
    plt.style.use('seaborn-paper')
    
    # 自定义黑白灰色调
    colors = ['#333333', '#666666']  # 深灰和中灰
    
    # 通用图表设置
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 2,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": '#cccccc'
    })
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    
    # 训练损失曲线及其阴影
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # 计算阴影区域（标准差）
    std_dev = np.std(train_loss) * 0.2
    plt.fill_between(epochs, 
                    [y - std_dev for y in train_loss],
                    [y + std_dev for y in train_loss],
                    alpha=0.2, color=colors[0])
    
    plt.plot(epochs, train_loss, 
            label='Training Loss',
            color=colors[0],
            linestyle='-',
            marker='o',
            markersize=5,
            markerfacecolor='white')
    
    # 验证损失曲线及其阴影
    std_dev = np.std(val_loss) * 0.2
    plt.fill_between(epochs,
                    [y - std_dev for y in val_loss],
                    [y + std_dev for y in val_loss],
                    alpha=0.2, color=colors[1])
    
    plt.plot(epochs, val_loss,
            label='Validation Loss',
            color=colors[1],
            linestyle='--',
            marker='s',
            markersize=5,
            markerfacecolor='white')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    
    # 训练准确率曲线及其阴影
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    # 计算阴影区域（标准差）
    std_dev = np.std(train_acc) * 0.2
    plt.fill_between(epochs,
                    [y - std_dev for y in train_acc],
                    [y + std_dev for y in train_acc],
                    alpha=0.2, color=colors[0])
    
    plt.plot(epochs, train_acc,
            label='Training Accuracy',
            color=colors[0],
            linestyle='-',
            marker='o',
            markersize=5,
            markerfacecolor='white')
    
    # 验证准确率曲线及其阴影
    std_dev = np.std(val_acc) * 0.2
    plt.fill_between(epochs,
                    [y - std_dev for y in val_acc],
                    [y + std_dev for y in val_acc],
                    alpha=0.2, color=colors[1])
    
    plt.plot(epochs, val_acc,
            label='Validation Accuracy',
            color=colors[1],
            linestyle='--',
            marker='s',
            markersize=5,
            markerfacecolor='white')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
