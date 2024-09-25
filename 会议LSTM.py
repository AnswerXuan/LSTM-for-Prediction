import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

#用前seq_len天的4个特征预测后mul天的adjclose这一项特征
#gpu,batch_size不能太大或者太小,1个epoch相当于把全部训练数据集训练了一遍,batch_size相当于一口气学几行(天).
# 学完一个batch更新一次神经网络权重,1个epoch学(总行数/batch_size)次,更新这么多次.多个epoch相当于一本书复习多遍,记得就牢了,融会贯通了

#para,可修改!!!!!!!!!!!!!!
seq_len = 20  # Sequence_length(10)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mulpre = 1 #多步预测预测的天数(3)!!!!!!!!!!!!!!!!!!!!!!!!!!

batch_size = 64#规定batch大小(64)
division_rate1 = 0.9#分割训练集占比,剩下的是测试集,原来是0.7
division_rate2 = 1.0

l = ['000001.SS', 'AAPL','BTC-USD','DJI','Gold_daily','GSPC','IXIC']

for i in l:

#path
    file_path = r'C:/lyx/learning/期刊论文/程序结果/LSTM/' + i#要保存图片和excel的那个文件夹,每次运行需要修改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    if(os.path.exists(file_path)):
        print('文件夹已存在')
    else:
        os.makedirs(file_path)

    picture_path = file_path + '/prediction_result.svg'#预测拟合图像保存路径,最后一项不用改


    excel_path = file_path +'/accu.xls'#最终的准确率和各项指标存放的路径和名称
    para_path = file_path + '/para.xls'#训练的神经网络参数
    picture_path4 = file_path + '/training_validation_loss_result'
    stock_name = i

    d = 0.2  # Dropout
    shape = [1, seq_len, 1]  # feature, window, output,和loaddata划分数据集还有buildmodeal有大关系!!!
    # neurons = [?, ?, mulpre]

    filename = 'C:/lyx/learning/会议论文/三支同时期数据/' + i + '.csv'
    epochs = 200
    decay = 0.2
    neurons = [200, 200, mulpre]
    accuracy = 0


    #gpu设置
    '''os.environ["CUDA_VISIBLE_DEVICES"]="0" # 使用编号为0号的GPU
    config = tf.compat.v1.ConfigProto
    
    config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
    session = tf.Session(config=config)
    
    # 设置session
    KTF.set_session(session )'''

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")
    from tensorflow.python.client import device_lib

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    #获取股票数据dataframe
    def get_stock_data(normalize=True):
        df = pd.read_csv(filename, usecols=['Adj Close'])
        if normalize:  # 归一化
            standard_scaler = preprocessing.StandardScaler()
            df['Adj Close'] = standard_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
        return df

    #先划分训练集测试集,再标准化归一化,避免数据泄露
    def load_data(df, seq_len , mul, normalize=True):
        amount_of_features = len(df.columns)  # columns是列索引,index是行索引
        data = df.values
        row1 = round(division_rate1 * data.shape[0])  # 70% split可改动!!!!!!!#round是四舍五入,0.9可能乘出来小数  #shape[0]是result列表中子列表的个数
        row2 = round(division_rate2 * data.shape[0])
        #训练集和测试集划分
        train = data[:int(row1), :]
        test = data[int(row1):int(row2), :]

        # 训练集和测试集归一化
        if normalize:
            standard_scaler = preprocessing.StandardScaler()
            train = standard_scaler.fit_transform(train.reshape(-1, 1))
            test = standard_scaler.fit_transform(test.reshape(-1, 1))

        X_train = []  # train列表中4个特征记录
        y_train = []
        X_test = []
        y_test = []
        train_samples=train.shape[0]-seq_len-mul+1
        test_samples = test.shape[0] - seq_len - mul + 1
        for i in range(0,train_samples,mul):  # maximum date = lastest date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
            X_train.append(train[i:i + seq_len,])#每个滑窗每天四个特征
            y_train.append(train[i+seq_len:i+seq_len+mul,-1])#-1即取最后一个特征

        for i in range(0, test_samples,mul):  # maximum date = lastest date - sequence length  #index从0到极限maximum,所有天数正好被滑窗采样完
            X_test.append(test[i:i + seq_len, ])  # 每个滑窗每天四个特征
            y_test.append(test[i + seq_len:i + seq_len + mul, -1])  # -1即取最后一个特征
        # X都对应全部4特征,y都对应adj close   #train都是前百分之90,test都是后百分之10
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print('train', train.shape)
        print(train)
        print('test', test.shape)
        print(test)
        print('X_train', X_train.shape)
        print('y_train', y_train.shape)
        print('X_test', X_test.shape)
        print('y_test', y_test.shape)
        print('df', df)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))  # (90%maximum, seq-1 ,4) #array才能reshape
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  # (10%maximum, seq-1 ,4) #array才能reshape
        print('X_train', X_train.shape)
        print('X_test', X_test.shape)
        return [X_train, y_train, X_test, y_test],row1,row2  # x是训练的数据，y是数据对应的标签,也就是说y是要预测的那一个特征!!!!!!


    #普通可视化
    # Draw Plot
    def plot_df(df, title="", xlabel='Date', ylabel='Adj Close', dpi=100):
        plt.figure(figsize=(16, 5), dpi=dpi)
        x = df.index
        y = df['Adj Close']
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()

    # Time series data source: fpp pacakge in R.
    #df = pd.read_csv(filename, parse_dates=['Date'],usecols=['Date','Open','High','Low','Adj Close', 'Volume'])

    '''head = df.head()
    print('head\n',head)
    plot_df(df, title='Adj Close of SP500ETF from 2010 to 2020.')'''



    def build_model(shape, neurons, d, decay):
        model = Sequential()

        model.add(LSTM(neurons[0], input_shape=(shape[1], shape[0]), return_sequences=True))
        model.add(Dropout(d))

        model.add(LSTM(neurons[1], return_sequences=False))
        model.add(Dropout(d))

        model.add(Dense(neurons[2]))

        adam = optimizers.Adam(decay=decay)
        # lr = lr\(1 + decay * iterations),adam自动逐渐减小学习率
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])  # loss函数可调节,可以增加准确率
        model.summary()
        return model


    def measure(d, shape, neurons, epochs, decay):
        tf.keras.backend.clear_session()#清理模块!!!!!!!!!!!!!!
        model = build_model(shape, neurons, d, decay)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
        trainScore = model.evaluate(X_train, y_train, verbose=0)  # 不输出日志信息
        print('Loss: %.5f MSE ' % trainScore[0])
        print('Accuracy: %.3f MSE ' % trainScore[1])
        return trainScore[0]




    '''def calculate_MAPE(pre,real):
        sum = 0
        for u in range(len(real)):  # for each data index in test data
            pr = pre[u][0]  # pr = prediction on day u
            re = real[u][0]
            sum += (abs(pr - re) / re) / re
        MAPE = (sum/len(real))*100
        return MAPE
    '''
    #计算涨跌趋势准确率
    def single_up_down_accuracy(pre, real):
        real_var = real[1:] - real[:len(real) - 1]  # 实际涨跌
        pre_var = pre[1:] - pre[:len(pre) - 1]  # 原始涨跌
        txt = np.zeros(len(real_var))
        for i in range(len(real_var - 1)):  # 计算数量
            txt[i] = (np.sign(real_var[i]) == np.sign(pre_var[i]))
        result = sum(txt) / len(txt)
        return result

    #计算multi天内涨跌趋势准确率
    def multi_up_down_accuracy(pre, real):
        real_var = real[mulpre:] - real[:len(real) - mulpre]  # 实际涨跌
        pre_var = pre[mulpre:] - pre[:len(pre) - mulpre]  # 原始涨跌
        txt = np.zeros(len(real_var))
        for i in range(len(real_var - 1)):  # 计算数量
            txt[i] = (np.sign(real_var[i]) == np.sign(pre_var[i]))
        result = sum(txt) / len(txt)
        return result

    #计算最终的准确率和各项指标,并且保存到文件夹excel中,(1-每日的相对误差率)计算算数平均值
    def calculate_accuracy(pre, real):
        accuracy = 0
        for u in range(len(real)):  # for each data index in test data
            pr = pre[u][0]  # pr = prediction on day u
            accuracy += ((1 - abs(pr - real[u]) / real[u])) / len(real)
            #percentage_diff.append((pr - y_train[u] / pr) * 100)

        single_trend_accuracy = single_up_down_accuracy(pre,real)
        multi_trend_accuracy = multi_up_down_accuracy(pre, real)

        # MAPE = np.mean(np.abs((pre - real) / real))
        MAPE = sklearn.metrics.mean_absolute_percentage_error(real,pre)
        #MAPE = calculate_MAPE(pre,real)
        RMSE = np.sqrt(np.mean(np.square(pre - real)))
        MSE = mean_squared_error(real, pre)
        MAE = np.mean(np.abs(pre - real))
        R2 = r2_score(pre, real)
        dict = {'single_trend_accuracy': single_trend_accuracy, 'multi_trend_accuracy': multi_trend_accuracy, 'accuracy': accuracy, 'MAPE': MAPE, 'RMSE': RMSE, 'MSE': MSE,'MAE': MAE, 'R2': R2}
        df = pd.DataFrame(dict)
        print('最终的准确率和指标如下\n',df)
        return df



    # 反归一化,这里有问题,用训练集和测试集一起反归一化了!!!!!!!!!!!!!!!!!!!!!!!
    def denormalize(normalized_value):
        df = pd.read_csv(filename)
        data = df['Adj Close'].values.reshape(-1, 1)  # 取原来没有归一化的adj数据作为样本
        row1 = round(division_rate1 * data.shape[0])  # 70% split可改动!!!!!!!#round是四舍五入,0.9可能乘出来小数  #shape[0]是result列表中子列表的个数
        row2 = round(division_rate2 * data.shape[0])
        # 训练集和测试集划分
        test = data[int(row1):int(row2), :]
        normalized_value = normalized_value.reshape(-1, 1)

        standard_scaler = preprocessing.StandardScaler()
        '反归一化'
        std = standard_scaler.fit_transform(test)  #利用m对data进行归一化，并储存df的归一化参数
        new = standard_scaler.inverse_transform(normalized_value)  # 利用m对normalized_value进行反归一化

        '归一化'
        '''m = min_max_scaler.fit_transform(train)  # 利用m对train进行归一化，并储存df的归一化参数!!
        new = min_max_scaler.transform(test)  # 利用m对test也进行归一化,注意这里是transform不能是fit_transform!!!1'''
        return new


    # 可视化  'plt原来是plt2'
    def plot_result(pre, real):
        real100 = real[0:101]
        pre100 = pre[0:101]
        plt.figure(figsize=(6.4, 4.8), dpi=2000)
        plt.plot(pre100, color='red', label='Prediction', linewidth=0.6)
        plt.plot(real100, color='blue', label='Actual', linewidth=0.6)
        plt.rcParams.update({'font.size': 20})
        plt.legend(loc='best')
        plt.title('The test result for {}'.format(stock_name), fontsize=20)
        plt.xlabel('Days', fontsize=20)
        plt.ylabel('Adjusted Closing Price', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        result = np.array(real100)
        min2 = min(result) - 5
        max2 = max(result) + 5
        plt.xlim([0, 101])
        plt.ylim([min2, max2])
        plt.savefig(picture_path, dpi=2000)  # 保存拟合曲线
        plt.clf()


    #plot_result(p, y_test) y_test(83,3)
    #plot_result(p, y_train)



    def training_validation_loss_plot(epochs,loss,val_loss):
        x_len = range(epochs)
        plt.plot(x_len, loss, 'r', label='training_loss')
        plt.plot(x_len, val_loss, 'b', label='validation_loss')
        plt.title('Training and validation loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(picture_path4)
        plt.clf()



    def save_data(accu,model):
        accu.to_excel(excel_path)#保存准确率和指标
        para = {'division_rate1': division_rate1, 'division_rate2': division_rate2,'seq_len': seq_len, 'mulpre': mulpre, 'batch_size' : batch_size}
        para = pd.DataFrame([para])
        para.to_excel(para_path)#保存网络参数
        '''model_path = str(file_path) + '/model.png'
        plot_model(model, to_file= model_path)#保存网络结构'''

    #数据整理
    df = get_stock_data()  # 修改了
    #print('df.shape',df.shape)
    [X_train, y_train, X_test, y_test],row1,row2 = load_data(df, seq_len, mulpre)
    #corr_heatmap(df)暂时不用热力图

    #IPSO得出final即最佳参数集


    #训练并且测试得出的最佳神经网络
    model = build_model(shape, neurons, d, decay)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)#batch_size不能太大或者太小!!!!!!!!!!!!!!!

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    training_validation_loss_plot(epochs,loss,val_loss)

    s = time.time()
    trainScore = model.evaluate(X_test, y_test, verbose=0)
    p = model.predict(X_test)
    e = time.time()

    timing = e - s
    print('时间', timing)

    print('pre.shape',p.shape)
    p = denormalize(p)
    y_test = denormalize(y_test)


    print('pre.shape',p.shape)#(735, 1)
    print('pre',p)
    print('real',y_test)
    print('real.shape',y_test.shape)#(735, 1)

    stock = i
    model2 = 'ARIMA'
    csv_path = 'C:/lyx/learning/期刊论文/程序结果/对比图表/' + stock +'/' + model2 + '.xls'
    df = pd.DataFrame(p)
    df.columns.name = None
    df.to_excel(csv_path,index=False,header=None)

    #计算所有指标并且保存
    accu = calculate_accuracy(p,y_test)
    save_data(accu,model)
    #用test测试最终效果, 并且绘制贴近图线
    #plot_result(p, y_test)



